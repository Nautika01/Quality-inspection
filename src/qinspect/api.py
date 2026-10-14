"""REST API (FastAPI) so MES / dashboards / other stations can use the inspector.

    uvicorn --factory qinspect.api:create_app --host 0.0.0.0 --port 8000
    QINSPECT_DB=/data/qinspect.db QINSPECT_SPEC=spec.json QINSPECT_MODEL=model.npz
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from . import __version__
from .anomaly import AnomalyModel, model_hash
from .db import InspectionDB, spec_hash
from .inspector import inspect_array
from .spc import capability, xbar_r
from .spec import PartSpec


def create_app(db_path: str | None = None, spec: PartSpec | None = None, model_path: str | None = None) -> FastAPI:
    db_path = db_path or os.environ.get("QINSPECT_DB", "qinspect.db")
    if spec is None:
        sp = os.environ.get("QINSPECT_SPEC")
        spec = PartSpec.from_json(sp) if sp else PartSpec()
    model_path = model_path or os.environ.get("QINSPECT_MODEL")
    model = AnomalyModel.load(model_path) if model_path else None
    mh = model_hash(model_path) if model_path else None

    app = FastAPI(title="qinspect", version=__version__)
    app.state.db = InspectionDB(db_path)
    app.state.spec = spec
    app.state.model = model

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "spec_hash": spec_hash(spec), "model_hash": mh}

    @app.get("/spec")
    def get_spec():
        from dataclasses import asdict

        return asdict(spec)

    @app.post("/inspect")
    async def inspect(file: UploadFile = File(...), lot_id: str = Form("LOT-API"), serial: str | None = Form(None),
                      station: str = Form("api")):
        data = await file.read()
        try:
            img = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"not an image: {exc}") from exc
        serial = serial or Path(file.filename or "unknown").stem
        res = inspect_array(img, serial, spec, model)
        rh = app.state.db.add_inspection(res, lot_id, spec, mh, station)
        return res.to_row() | {"lot_id": lot_id, "record_hash": rh,
                               "defects": [{"kind": g.defect.kind, "grade": g.grade, "length_mm": g.defect.length_mm,
                                            "bbox": g.defect.bbox} for g in res.defects]}

    @app.get("/lots")
    def lots():
        return app.state.db.lots()

    @app.get("/lots/{lot_id}/summary")
    def lot_summary(lot_id: str, subgroup: int = 5, baseline: int = 10):
        db: InspectionDB = app.state.db
        s = db.lot_summary(lot_id)
        if s["n"] == 0:
            raise HTTPException(404, "lot not found or empty")
        od = np.array([r["od_mm"] for r in db.inspections(lot_id) if r["od_mm"] is not None])
        if len(od) >= 2 * subgroup:
            ch = xbar_r(od, n=subgroup, baseline=baseline)
            s["spc_od"] = {"center": ch.center, "ucl": ch.ucl, "lcl": ch.lcl, "out_of_control": ch.out_of_control,
                           "violations": {str(k + 1): v for k, v in sorted(ch.violations.items())},
                           "capability": capability(od, spec.outer_diameter.lsl, spec.outer_diameter.usl, ch.sigma_within)}
        return s

    @app.get("/lots/{lot_id}/results")
    def lot_results(lot_id: str, limit: int | None = None):
        return app.state.db.inspections(lot_id, limit)

    @app.get("/serials/{serial}")
    def serial(serial: str):
        db: InspectionDB = app.state.db
        row = db.conn.execute("SELECT * FROM inspections WHERE serial=? ORDER BY id DESC LIMIT 1", (serial,)).fetchone()
        if not row:
            raise HTTPException(404, "serial not found")
        return dict(row) | {"defects": db.defects_for(row["id"])}

    @app.get("/events")
    def events(kind: str | None = None, limit: int = 100):
        return app.state.db.events(kind, limit)

    @app.get("/audit/verify")
    def audit_verify():
        return app.state.db.verify_chain()

    return app
