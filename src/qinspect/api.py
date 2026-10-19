"""REST API (FastAPI) so MES / dashboards / other stations can use the inspector.

    qinspect serve --db /data/qinspect.db            (or: uvicorn --factory qinspect.api:create_app)
    QINSPECT_DB, QINSPECT_SPEC, QINSPECT_MODEL, QINSPECT_API_KEY, QINSPECT_IMAGE_DIR, QINSPECT_MAX_UPLOAD_MB

Writes (`POST`) require the `X-API-Key` header when keys are configured; reads are open on
the plant network. Roles: QINSPECT_API_KEY = one admin key (every write), or
QINSPECT_API_KEYS="key1:inspector,key2:quality,key3:auditor" where
inspector -> /inspect, quality -> inspector + dispositions / lot close / freeze / ack,
auditor -> /audit/anchor, admin -> everything. Every record carries the station and operator supplied by the
caller, and the SHA-256 of the uploaded image (stored under QINSPECT_IMAGE_DIR for NG / ERROR
frames when set).
"""

from __future__ import annotations

import io
import os
import secrets
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import __version__
from .anomaly import AnomalyModel, model_hash
from .db import DISPOSITIONS, InspectionDB, image_sha256, spec_hash
from .inspector import VERDICT_PASS, inspect_array
from .sampling import AQLS, INSPECTION_LEVELS, draw_sample, iso2859_single_normal, lot_disposition
from .spc import SUBGROUP_SIZES, capability, xbar_r
from .spec import PartSpec
from .util import safe_name, utc_param
from .vision import to_gray8


def _window(since: str | None, until: str | None) -> tuple[str | None, str | None]:
    """Validate ISO-8601 query parameters (422 on a malformed value)."""
    try:
        s, u = utc_param(since, "since"), utc_param(until, "until")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if s and u and s >= u:
        raise HTTPException(422, "since must be earlier than until")
    return s, u


class DispositionIn(BaseModel):
    decision: str = Field(pattern="^(" + "|".join(DISPOSITIONS) + ")$")
    decided_by: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=500)
    mrb_ref: str | None = Field(default=None, max_length=64)


class LotCloseIn(BaseModel):
    decided_by: str = Field(min_length=1, max_length=64)
    aql: float = 1.0
    level: str = "II"
    seed: int | None = None
    note: str | None = Field(default=None, max_length=500)
    decision_override: str | None = Field(default=None, pattern="^(ACCEPT|REJECT|HOLD)$")


class FreezeIn(BaseModel):
    characteristic: str = Field(pattern="^(od_mm|pocket_mm|concentricity_mm)$")
    frozen_by: str = Field(min_length=1, max_length=64)
    subgroup: int = 5
    baseline: int | None = None


class AckIn(BaseModel):
    by: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=500)


ROLES = {"inspector": {"inspect"}, "quality": {"inspect", "quality"}, "auditor": {"audit"},
         "admin": {"inspect", "quality", "audit"}}


def parse_api_keys(single: str | None, multi: str | None) -> dict[str, str]:
    """{key: role} from QINSPECT_API_KEY (admin) and/or QINSPECT_API_KEYS='k:role,...'."""
    keys: dict[str, str] = {}
    if single:
        keys[single] = "admin"
    for item in (multi or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, role = item.partition(":")
        role = role or "admin"
        if role not in ROLES:
            raise ValueError(f"unknown API role {role!r}; use one of {sorted(ROLES)}")
        keys[key.strip()] = role
    return keys


def create_app(db_path: str | None = None, spec: PartSpec | None = None, model_path: str | None = None,
               api_key: str | None = None, image_dir: str | None = None, api_keys: dict[str, str] | None = None) -> FastAPI:
    db_path = db_path or os.environ.get("QINSPECT_DB", "qinspect.db")
    if spec is None:
        sp = os.environ.get("QINSPECT_SPEC")
        spec = PartSpec.from_json(sp) if sp else PartSpec()
    model_path = model_path or os.environ.get("QINSPECT_MODEL")
    model = AnomalyModel.load(model_path) if model_path else None
    mh = model_hash(model_path) if model_path else None
    if api_keys is None:
        api_keys = parse_api_keys(api_key if api_key is not None else os.environ.get("QINSPECT_API_KEY"),
                                  os.environ.get("QINSPECT_API_KEYS"))
    image_dir = image_dir or os.environ.get("QINSPECT_IMAGE_DIR")
    max_upload = int(float(os.environ.get("QINSPECT_MAX_UPLOAD_MB", "20")) * 1024 * 1024)

    app = FastAPI(title="qinspect", version=__version__)
    app.state.db = InspectionDB(db_path)
    app.state.spec = spec
    app.state.model = model

    def require(permission: str):
        def dep(x_api_key: str | None = Header(default=None)):
            if not api_keys:
                return  # open mode (no keys configured)
            role = None
            for key, r in api_keys.items():
                if x_api_key and secrets.compare_digest(x_api_key, key):
                    role = r
            if role is None:
                raise HTTPException(401, "missing or invalid X-API-Key")
            if permission not in ROLES[role]:
                raise HTTPException(403, f"role {role!r} may not {permission}")
        return dep

    require_inspect, require_quality, require_audit = require("inspect"), require("quality"), require("audit")

    def get_db() -> InspectionDB:
        return app.state.db

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "spec_hash": spec_hash(spec), "model_hash": mh,
                "auth": bool(api_keys), "roles": sorted(set(api_keys.values()))}

    @app.get("/spec")
    def get_spec():
        return spec.to_dict()

    @app.post("/inspect", dependencies=[Depends(require_inspect)])
    def inspect(file: UploadFile = File(...), lot_id: str = Form("LOT-API", min_length=1, max_length=64),
                serial: str | None = Form(None, max_length=64), station: str = Form("api", max_length=64),
                operator: str | None = Form(None, max_length=64), reinspection_of: int | None = Form(None)):
        data = file.file.read(max_upload + 1)
        if len(data) > max_upload:
            raise HTTPException(413, f"upload exceeds {max_upload // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                arr = np.asarray(im) if im.mode in ("I;16", "I;16B", "I;16L", "I", "F") else np.asarray(im.convert("L"))
            img = to_gray8(arr)
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(400, "not a decodable image") from None
        serial = safe_name(serial, safe_name(Path(file.filename or "").stem, "unknown"))
        res = inspect_array(img, serial, spec, model)
        img_hash = image_sha256(img)
        img_path = None
        if image_dir and res.verdict != VERDICT_PASS:
            Path(image_dir).mkdir(parents=True, exist_ok=True)
            img_path = str(Path(image_dir) / f"{serial}_{img_hash[:8]}.png")
            Image.fromarray(img).save(img_path)
        try:
            rh = app.state.db.add_inspection(res, lot_id, spec, mh, station, operator=operator, image_hash=img_hash,
                                             image_path=img_path, reinspection_of=reinspection_of)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return res.to_row() | {"lot_id": lot_id, "record_hash": rh, "image_sha256": img_hash,
                               "defects": [{"kind": g.defect.kind, "grade": g.grade, "length_mm": g.defect.length_mm,
                                            "bbox": g.defect.bbox, "rejecting": g.rejecting} for g in res.defects]}

    @app.get("/lots")
    def lots(db: InspectionDB = Depends(get_db)):
        return db.lots()

    @app.get("/lots/{lot_id}/summary")
    def lot_summary(lot_id: str, subgroup: int = Query(5, ge=min(SUBGROUP_SIZES), le=max(SUBGROUP_SIZES)),
                    baseline: int = Query(25, ge=2), since: str | None = None, until: str | None = None,
                    station: str | None = None, db: InspectionDB = Depends(get_db)):
        since, until = _window(since, until)
        s = db.lot_summary(lot_id, since=since, until=until, station=station)
        if s["n_inspections"] == 0:
            raise HTTPException(404, "lot not found or empty")
        lot_spec = db.lot_spec(lot_id) or spec
        rows = db.inspections(lot_id, since=since, until=until, station=station)
        s["spc"] = {}
        for ch, dim in (("od_mm", lot_spec.outer_diameter), ("pocket_mm", lot_spec.pocket_diameter)):
            if dim is None:
                continue
            vals = np.array([r[ch] for r in rows if r[ch] is not None], dtype=float)
            frozen = db.active_limits(lot_spec.part_number, ch).get(ch)
            need = frozen["n"] if frozen else 2 * subgroup  # frozen limits carry their own subgroup size
            if len(vals) < need:
                continue
            if frozen:
                from .spc import ControlLimits, evaluate

                ch_obj = evaluate(vals, ControlLimits.from_dict(frozen))
                src = f"frozen:n={frozen['n']}"
            else:
                ch_obj = xbar_r(vals, n=subgroup, baseline=baseline)
                src = f"phase1:{baseline}"
            s["spc"][ch] = {"limits_source": src, "center": ch_obj.center, "ucl": ch_obj.ucl, "lcl": ch_obj.lcl,
                            "out_of_control": ch_obj.out_of_control,
                            "violations": {str(k + 1): v for k, v in sorted(ch_obj.violations.items())},
                            "capability": capability(vals, dim.lsl, dim.usl, ch_obj.sigma_within)}
        s["lot_dispositions"] = db.lot_dispositions(lot_id)
        return s

    @app.get("/lots/{lot_id}/results")
    def lot_results(lot_id: str, limit: int | None = Query(None, ge=1, le=100000), since: str | None = None,
                    until: str | None = None, station: str | None = None, db: InspectionDB = Depends(get_db)):
        since, until = _window(since, until)
        return db.inspections(lot_id, limit, since=since, until=until, station=station)

    @app.get("/lots/{lot_id}/open-ng")
    def lot_open_ng(lot_id: str, db: InspectionDB = Depends(get_db)):
        return db.open_ng(lot_id)

    @app.post("/lots/{lot_id}/close", dependencies=[Depends(require_quality)])
    def lot_close(lot_id: str, body: LotCloseIn, db: InspectionDB = Depends(get_db)):
        """Acceptance sampling on the lot's latest-per-serial results, then record the lot disposition."""
        if body.aql not in AQLS or body.level not in INSPECTION_LEVELS:
            raise HTTPException(422, f"aql must be one of {AQLS}, level one of {INSPECTION_LEVELS}")
        rows = [r for r in db.inspections(lot_id) if r["verdict"] != "ERROR"]
        if not rows:
            raise HTTPException(404, "lot not found or empty")
        latest = {}
        for r in rows:
            latest[r["serial"]] = r
        serials = sorted(latest)
        try:
            plan = iso2859_single_normal(len(serials), body.aql, body.level)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        sample = draw_sample(serials, plan, body.seed)
        n_def = sum(latest[s]["verdict"] == "NG" for s in sample)
        n_crit = sum(latest[s]["worst_grade"] == "critical" for s in sample)
        disp = lot_disposition(plan, n_def, n_crit)
        decision = body.decision_override or disp["decision"]
        try:
            out = db.close_lot(lot_id, decision, plan.to_dict(), sample, n_def, n_crit, body.decided_by, body.note)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return out | {"sampling": disp, "n_serials": len(serials)}

    @app.get("/serials/{serial}")
    def serial(serial: str, db: InspectionDB = Depends(get_db)):
        hist = db.serial_history(serial)
        if not hist:
            raise HTTPException(404, "serial not found")
        return hist[-1] | {"history": hist}

    @app.get("/inspections/{inspection_id}")
    def inspection(inspection_id: int, db: InspectionDB = Depends(get_db)):
        row = db.inspection(inspection_id)
        if not row:
            raise HTTPException(404, "inspection not found")
        return row | {"dispositions": db.dispositions_for(inspection_id)}

    @app.post("/inspections/{inspection_id}/disposition", dependencies=[Depends(require_quality)])
    def disposition(inspection_id: int, body: DispositionIn, db: InspectionDB = Depends(get_db)):
        try:
            did = db.add_disposition(inspection_id, body.decision, body.decided_by, body.note, body.mrb_ref)
        except ValueError as exc:
            raise HTTPException(404 if "not found" in str(exc) else 422, str(exc)) from None
        return {"id": did, "inspection_id": inspection_id, "decision": body.decision}

    @app.get("/events")
    def events(kind: str | None = None, limit: int = Query(100, ge=1, le=10000), lot_id: str | None = None,
               unacked: bool = False, db: InspectionDB = Depends(get_db)):
        return db.events(kind, limit, lot_id=lot_id, unacked_only=unacked)

    @app.post("/events/{event_id}/ack", dependencies=[Depends(require_quality)])
    def ack(event_id: int, body: AckIn, db: InspectionDB = Depends(get_db)):
        if not db.ack_event(event_id, body.by, body.note):
            raise HTTPException(404, "event not found or already acknowledged")
        return {"id": event_id, "acked_by": body.by}

    @app.get("/spc/limits")
    def spc_limits(lot_id: str | None = None, part_number: str | None = None, db: InspectionDB = Depends(get_db)):
        """Active frozen limits for a part (default: the API's spec; or the part a lot is bound to)."""
        if lot_id:
            lot_spec = db.lot_spec(lot_id)
            if lot_spec is None:
                raise HTTPException(404, "lot not found")
            part_number = lot_spec.part_number
        return db.active_limits(part_number or spec.part_number)

    @app.post("/spc/limits/freeze", dependencies=[Depends(require_quality)])
    def spc_freeze(body: FreezeIn, lot_id: str = Query(...), db: InspectionDB = Depends(get_db)):
        """Freeze X-bar/R limits for a characteristic from a lot's Phase I data, filed under the lot's part."""
        if body.subgroup not in SUBGROUP_SIZES:
            raise HTTPException(422, f"subgroup must be one of {SUBGROUP_SIZES}")
        lot_spec = db.lot_spec(lot_id)
        if lot_spec is None:
            raise HTTPException(404, "lot not found")
        if body.characteristic != "od_mm" and not lot_spec.has_pocket:
            raise HTTPException(422, f"{body.characteristic}: the lot's part has no pocket")
        vals = np.array([r[body.characteristic] for r in db.inspections(lot_id) if r[body.characteristic] is not None])
        if len(vals) < 2 * body.subgroup:
            raise HTTPException(422, "not enough data for two subgroups")
        ch = xbar_r(vals, n=body.subgroup, baseline=body.baseline)
        lid = db.freeze_limits(lot_spec.part_number, body.characteristic, ch.limits.to_dict(), body.frozen_by, lot_id)
        return {"id": lid, "part_number": lot_spec.part_number, "characteristic": body.characteristic, **ch.limits.to_dict()}

    @app.get("/calibrations")
    def calibrations(db: InspectionDB = Depends(get_db)):
        return db.calibrations()

    @app.get("/reports/shift")
    def shift_report(lot_id: str, since: str, until: str, station: str | None = None,
                     db: InspectionDB = Depends(get_db)):
        """Per-shift figures for a time window [since, until); ISO-8601, any offset (normalised to UTC)."""
        since, until = _window(since, until)
        if not since or not until:
            raise HTTPException(422, "since and until are required")
        s = db.lot_summary(lot_id, since=since, until=until, station=station)
        alarms = db.events(prefix="alarm:", limit=None, lot_id=lot_id)
        s["alarms"] = [a for a in alarms if since <= a["ts"] < until]
        s["chain"] = db.verify_chain()
        return s

    @app.get("/audit/verify")
    def audit_verify(db: InspectionDB = Depends(get_db)):
        return db.verify_chain()

    @app.post("/audit/anchor", dependencies=[Depends(require_audit)])
    def audit_anchor(note: str | None = None, db: InspectionDB = Depends(get_db)):
        return db.anchor_head(note)

    return app
