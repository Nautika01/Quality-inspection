"""SQLite persistence with a tamper-evident audit trail (hash chain).

Every inspection row stores `prev_hash` and `record_hash = sha256(prev_hash || canonical
record)`. Changing or deleting any historical row breaks the chain, which
`verify_chain()` detects. Also recorded: inspection program version, spec hash and
anomaly-model hash, so each decision can be reproduced.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .inspector import InspectionResult
from .spec import PartSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS lots (
    lot_id TEXT PRIMARY KEY,
    part_number TEXT,
    spec_hash TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    serial TEXT NOT NULL,
    lot_id TEXT NOT NULL REFERENCES lots(lot_id),
    inspected_at TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reasons TEXT NOT NULL,
    od_mm REAL, pocket_mm REAL, concentricity_mm REAL,
    worst_grade TEXT,
    anomaly_score REAL,
    program_version TEXT,
    spec_hash TEXT,
    model_hash TEXT,
    station TEXT,
    prev_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_insp_lot ON inspections(lot_id, id);
CREATE INDEX IF NOT EXISTS idx_insp_serial ON inspections(serial);
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id),
    kind TEXT, grade TEXT, length_mm REAL, area_px INTEGER,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    lot_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""

GENESIS = "0" * 64


def spec_hash(spec: PartSpec) -> str:
    return hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()[:16]


def _canonical(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class InspectionDB:
    def __init__(self, path: str | Path = "qinspect.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- lots ---------------------------------------------------------------
    def ensure_lot(self, lot_id: str, spec: PartSpec) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO lots(lot_id, part_number, spec_hash) VALUES (?,?,?)",
            (lot_id, spec.part_number, spec_hash(spec)),
        )
        self.conn.commit()

    def lots(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT l.lot_id, l.part_number, l.created_at,
                      COUNT(i.id) AS n, SUM(i.verdict='PASS') AS n_pass
               FROM lots l LEFT JOIN inspections i ON i.lot_id=l.lot_id
               GROUP BY l.lot_id ORDER BY l.created_at"""
        ).fetchall()
        return [dict(r) for r in rows]

    # -- inspections --------------------------------------------------------
    def _last_hash(self) -> str:
        row = self.conn.execute("SELECT record_hash FROM inspections ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    @staticmethod
    def _record(res: InspectionResult, lot_id: str, sh: str, mh: str | None, station: str | None) -> dict:
        return {
            "serial": res.serial, "lot_id": lot_id, "inspected_at": res.inspected_at,
            "verdict": res.verdict, "reasons": ";".join(res.reasons),
            "od_mm": res.od_mm, "pocket_mm": res.pocket_mm, "concentricity_mm": res.concentricity_mm,
            "worst_grade": res.worst_grade, "anomaly_score": res.anomaly_score,
            "program_version": res.program_version, "spec_hash": sh, "model_hash": mh, "station": station,
            "defects": [
                {"kind": g.defect.kind, "grade": g.grade, "length_mm": g.defect.length_mm,
                 "area_px": g.defect.area_px, "bbox": list(g.defect.bbox)}
                for g in res.defects
            ],
        }

    def add_inspection(self, res: InspectionResult, lot_id: str, spec: PartSpec,
                       model_hash: str | None = None, station: str | None = None) -> str:
        self.ensure_lot(lot_id, spec)
        rec = self._record(res, lot_id, spec_hash(spec), model_hash, station)
        prev = self._last_hash()
        rh = hashlib.sha256((prev + _canonical(rec)).encode()).hexdigest()
        cur = self.conn.execute(
            """INSERT INTO inspections(serial, lot_id, inspected_at, verdict, reasons, od_mm, pocket_mm,
               concentricity_mm, worst_grade, anomaly_score, program_version, spec_hash, model_hash, station,
               prev_hash, record_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec["serial"], lot_id, rec["inspected_at"], rec["verdict"], rec["reasons"], rec["od_mm"],
             rec["pocket_mm"], rec["concentricity_mm"], rec["worst_grade"], rec["anomaly_score"],
             rec["program_version"], rec["spec_hash"], model_hash, station, prev, rh),
        )
        iid = cur.lastrowid
        for d in rec["defects"]:
            self.conn.execute(
                "INSERT INTO defects(inspection_id, kind, grade, length_mm, area_px, x0, y0, x1, y1) VALUES (?,?,?,?,?,?,?,?,?)",
                (iid, d["kind"], d["grade"], d["length_mm"], d["area_px"], *d["bbox"]),
            )
        self.conn.commit()
        return rh

    def inspections(self, lot_id: str | None = None, limit: int | None = None) -> list[dict]:
        q = "SELECT * FROM inspections" + (" WHERE lot_id=?" if lot_id else "") + " ORDER BY id"
        if limit:
            q = q.replace("ORDER BY id", "ORDER BY id DESC LIMIT %d" % int(limit))
        rows = self.conn.execute(q, (lot_id,) if lot_id else ()).fetchall()
        out = [dict(r) for r in rows]
        return out[::-1] if limit else out

    def defects_for(self, inspection_id: int) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM defects WHERE inspection_id=?", (inspection_id,))]

    def lot_summary(self, lot_id: str) -> dict:
        rows = self.inspections(lot_id)
        n = len(rows)
        n_pass = sum(r["verdict"] == "PASS" for r in rows)
        reasons: dict[str, int] = {}
        for r in rows:
            for x in filter(None, r["reasons"].split(";")):
                reasons[x] = reasons.get(x, 0) + 1
        return {
            "lot_id": lot_id, "n": n, "n_pass": n_pass, "n_ng": n - n_pass,
            "yield": n_pass / n if n else None,
            "ng_pareto": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        }

    # -- events / audit -----------------------------------------------------
    def add_event(self, kind: str, payload: dict, lot_id: str | None = None, ts: str | None = None) -> None:
        from datetime import datetime, timezone

        ts = ts or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute("INSERT INTO events(ts, lot_id, kind, payload) VALUES (?,?,?,?)",
                          (ts, lot_id, kind, json.dumps(payload, ensure_ascii=False)))
        self.conn.commit()

    def events(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        q = "SELECT * FROM events" + (" WHERE kind=?" if kind else "") + " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(q, ((kind, limit) if kind else (limit,))).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows][::-1]

    def verify_chain(self) -> dict:
        """Recompute every hash; report the first broken link if any."""
        prev = GENESIS
        n = 0
        for r in self.conn.execute("SELECT * FROM inspections ORDER BY id"):
            rec = {k: r[k] for k in ("serial", "lot_id", "inspected_at", "verdict", "reasons", "od_mm", "pocket_mm",
                                     "concentricity_mm", "worst_grade", "anomaly_score", "program_version",
                                     "spec_hash", "model_hash", "station")}
            rec["defects"] = [
                {"kind": d["kind"], "grade": d["grade"], "length_mm": d["length_mm"], "area_px": d["area_px"],
                 "bbox": [d["x0"], d["y0"], d["x1"], d["y1"]]}
                for d in self.defects_for(r["id"])
            ]
            expected = hashlib.sha256((prev + _canonical(rec)).encode()).hexdigest()
            if r["prev_hash"] != prev or r["record_hash"] != expected:
                return {"ok": False, "checked": n, "broken_at_id": r["id"], "serial": r["serial"]}
            prev = r["record_hash"]
            n += 1
        return {"ok": True, "checked": n, "head": prev}
