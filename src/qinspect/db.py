"""SQLite persistence with a tamper-evident audit trail (hash chain).

Every inspection row stores `prev_hash` and `record_hash = sha256(prev_hash || canonical
record)`. Changing or deleting any *historical* row breaks the chain, which
`verify_chain()` detects. Deleting the newest rows cannot be seen by the chain alone, so
the head is periodically written to `anchors` (and should be exported to an external
place: MES, printout, WORM storage); `verify_chain()` also checks the latest anchor.

Concurrency: one process may hold several connections and several processes may share
the file. Each write happens inside `BEGIN IMMEDIATE` so the head is read and the new
row inserted atomically; `UNIQUE(prev_hash)` makes a forked chain impossible to persist;
WAL mode lets dashboards read while a station writes.

Also recorded per row: inspection program version, spec hash, anomaly-model hash,
station, operator and the SHA-256 of the inspected image, so each decision can be
reproduced.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from pathlib import Path

from .inspector import VERDICT_ERROR, VERDICT_NG, VERDICT_PASS, InspectionResult, utc_now
from .spec import PartSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS lots (
    lot_id TEXT PRIMARY KEY,
    part_number TEXT,
    spec_hash TEXT,
    spec_json TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    serial TEXT NOT NULL,
    lot_id TEXT NOT NULL REFERENCES lots(lot_id),
    inspected_at TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('PASS','NG','ERROR')),
    reasons TEXT NOT NULL,
    od_mm REAL, pocket_mm REAL, concentricity_mm REAL,
    worst_grade TEXT,
    anomaly_score REAL,
    program_version TEXT,
    spec_hash TEXT,
    model_hash TEXT,
    station TEXT,
    operator TEXT,
    image_sha256 TEXT,
    image_path TEXT,
    reinspection_of INTEGER REFERENCES inspections(id),
    prev_hash TEXT NOT NULL UNIQUE,
    record_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_insp_lot ON inspections(lot_id, id);
CREATE INDEX IF NOT EXISTS idx_insp_serial ON inspections(serial);
CREATE INDEX IF NOT EXISTS idx_insp_time ON inspections(inspected_at);
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id),
    kind TEXT, grade TEXT, length_mm REAL, area_px INTEGER,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER
);
CREATE INDEX IF NOT EXISTS idx_defects_insp ON defects(inspection_id, id);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    lot_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    acked_by TEXT,
    acked_at TEXT,
    ack_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, id);
CREATE TABLE IF NOT EXISTS dispositions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id),
    decision TEXT NOT NULL CHECK (decision IN ('rework','scrap','use_as_is','mrb','reinspect')),
    decided_by TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    note TEXT,
    mrb_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_disp_insp ON dispositions(inspection_id);
CREATE TABLE IF NOT EXISTS lot_dispositions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT NOT NULL REFERENCES lots(lot_id),
    decision TEXT NOT NULL CHECK (decision IN ('ACCEPT','REJECT','HOLD')),
    plan_json TEXT NOT NULL,
    sample_json TEXT NOT NULL,
    n_defective INTEGER NOT NULL,
    n_critical INTEGER NOT NULL,
    decided_by TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    note TEXT,
    chain_head TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS control_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_number TEXT NOT NULL,
    characteristic TEXT NOT NULL,
    n INTEGER NOT NULL,
    center REAL NOT NULL, ucl REAL NOT NULL, lcl REAL NOT NULL,
    r_bar REAL NOT NULL, r_ucl REAL NOT NULL, r_lcl REAL NOT NULL,
    source_lot TEXT,
    frozen_by TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_cl ON control_limits(part_number, characteristic, active);
CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    station TEXT,
    operator TEXT,
    gauge_id TEXT,
    gauge_diameter_mm REAL NOT NULL,
    px_per_mm REAL NOT NULL,
    image_sha256 TEXT,
    spec_hash TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS anchors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    n_records INTEGER NOT NULL,
    head TEXT NOT NULL,
    note TEXT
);
"""

GENESIS = "0" * 64
DISPOSITIONS = ("rework", "scrap", "use_as_is", "mrb", "reinspect")
_RECORD_FIELDS = ("serial", "lot_id", "inspected_at", "verdict", "reasons", "od_mm", "pocket_mm",
                  "concentricity_mm", "worst_grade", "anomaly_score", "program_version", "spec_hash",
                  "model_hash", "station", "operator", "image_sha256", "reinspection_of")
_FLOAT_FIELDS = ("od_mm", "pocket_mm", "concentricity_mm", "anomaly_score")


def spec_hash(spec: PartSpec) -> str:
    return hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def image_sha256(img) -> str:
    import numpy as np

    a = np.ascontiguousarray(img)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode() + str(a.dtype).encode()).hexdigest()


def _num(v, as_int: bool = False):
    """Normalise a numeric field exactly as SQLite will store it (NaN/inf -> NULL, -0.0 -> 0.0)."""
    if v is None:
        return None
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
    if as_int:
        return int(v)
    v = float(v)
    if not math.isfinite(v):
        return None
    return v + 0.0  # -0.0 -> 0.0


def canonical_record(rec: dict) -> str:
    """Deterministic JSON of a record; identical whether built from Python values or re-read from SQLite."""
    clean = {}
    for k in _RECORD_FIELDS:
        v = rec.get(k)
        if k in _FLOAT_FIELDS:
            v = _num(v)
        elif k == "reinspection_of":
            v = None if v is None else int(v)
        elif v is not None and not isinstance(v, str):
            v = str(v)
        clean[k] = v
    clean["defects"] = [
        {"kind": str(d["kind"]), "grade": str(d["grade"]), "length_mm": _num(d["length_mm"]),
         "area_px": _num(d["area_px"], as_int=True), "bbox": [int(x) for x in d["bbox"]]}
        for d in rec.get("defects", [])
    ]
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(prev: str, canonical: str) -> str:
    return hashlib.sha256((prev + canonical).encode()).hexdigest()


class InspectionDB:
    def __init__(self, path: str | Path = "qinspect.db", *, create: bool = True, timeout: float = 30.0):
        self.path = str(path)
        if not create and self.path != ":memory:" and not Path(self.path).exists():
            raise FileNotFoundError(f"database not found: {self.path}")
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=timeout, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=%d" % int(timeout * 1000))
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        """Locked read: one connection is shared by several threads (API threadpool, dashboard)."""
        with self._lock:
            return self.conn.execute(sql, tuple(args)).fetchall()

    def _q1(self, sql: str, args=()):
        rows = self._q(sql, args)
        return rows[0] if rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- lots ---------------------------------------------------------------
    def ensure_lot(self, lot_id: str, spec: PartSpec) -> None:
        """Create the lot bound to `spec`; refuse a different spec revision for an existing lot."""
        if not lot_id or not isinstance(lot_id, str):
            raise ValueError("lot_id must be a non-empty string")
        with self._lock:
            row = self.conn.execute("SELECT spec_hash, status FROM lots WHERE lot_id=?", (lot_id,)).fetchone()
            sh = spec_hash(spec)
            if row is None:
                self.conn.execute(
                    "INSERT INTO lots(lot_id, part_number, spec_hash, spec_json, created_at) VALUES (?,?,?,?,?)",
                    (lot_id, spec.part_number, sh, json.dumps(spec.to_dict(), ensure_ascii=False), utc_now()),
                )
            elif row["spec_hash"] != sh:
                raise ValueError(f"lot {lot_id} is bound to spec {row['spec_hash']}, got {sh}; open a new lot for a new revision")
            elif row["status"] == "closed":
                raise ValueError(f"lot {lot_id} is closed")

    def lot_spec(self, lot_id: str) -> PartSpec | None:
        row = self._q1("SELECT spec_json FROM lots WHERE lot_id=?", (lot_id,))
        return PartSpec.from_dict(json.loads(row["spec_json"])) if row and row["spec_json"] else None

    def lots(self) -> list[dict]:
        rows = self._q(
            """SELECT l.lot_id, l.part_number, l.spec_hash, l.status, l.created_at, l.closed_at,
                      COUNT(i.id) AS n, SUM(i.verdict='PASS') AS n_pass, SUM(i.verdict='NG') AS n_ng,
                      SUM(i.verdict='ERROR') AS n_error
               FROM lots l LEFT JOIN inspections i ON i.lot_id=l.lot_id
               GROUP BY l.lot_id ORDER BY l.created_at, l.lot_id"""
        )
        return [dict(r) for r in rows]

    # -- inspections --------------------------------------------------------
    @staticmethod
    def _record(res: InspectionResult, lot_id: str, sh: str, mh: str | None, station: str | None,
                operator: str | None, img_hash: str | None, reinspection_of: int | None) -> dict:
        return {
            "serial": res.serial, "lot_id": lot_id, "inspected_at": res.inspected_at,
            "verdict": res.verdict, "reasons": ";".join(res.reasons),
            "od_mm": res.od_mm, "pocket_mm": res.pocket_mm, "concentricity_mm": res.concentricity_mm,
            "worst_grade": res.worst_grade, "anomaly_score": res.anomaly_score,
            "program_version": res.program_version, "spec_hash": sh, "model_hash": mh, "station": station,
            "operator": operator, "image_sha256": img_hash, "reinspection_of": reinspection_of,
            "defects": [
                {"kind": g.defect.kind, "grade": g.grade, "length_mm": g.defect.length_mm,
                 "area_px": g.defect.area_px, "bbox": list(g.defect.bbox)}
                for g in res.defects
            ],
        }

    def add_inspection(self, res: InspectionResult, lot_id: str, spec: PartSpec, model_hash: str | None = None,
                       station: str | None = None, *, operator: str | None = None, image_hash: str | None = None,
                       image_path: str | None = None, reinspection_of: int | None = None) -> str:
        """Append one record atomically; returns its record hash."""
        if res.verdict not in (VERDICT_PASS, VERDICT_NG, VERDICT_ERROR):
            raise ValueError(f"unknown verdict {res.verdict!r}")
        rec = self._record(res, lot_id, spec_hash(spec), model_hash, station, operator, image_hash, reinspection_of)
        canon = canonical_record(rec)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.ensure_lot(lot_id, spec)
                if reinspection_of is not None:
                    prev_row = self.conn.execute("SELECT serial FROM inspections WHERE id=?", (reinspection_of,)).fetchone()
                    if prev_row is None:
                        raise ValueError(f"reinspection_of {reinspection_of} does not exist")
                    if prev_row["serial"] != res.serial:
                        raise ValueError("reinspection must reference the same serial")
                row = self.conn.execute("SELECT record_hash FROM inspections ORDER BY id DESC LIMIT 1").fetchone()
                prev = row[0] if row else GENESIS
                rh = _hash(prev, canon)
                cur = self.conn.execute(
                    """INSERT INTO inspections(serial, lot_id, inspected_at, verdict, reasons, od_mm, pocket_mm,
                       concentricity_mm, worst_grade, anomaly_score, program_version, spec_hash, model_hash, station,
                       operator, image_sha256, image_path, reinspection_of, prev_hash, record_hash)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (rec["serial"], lot_id, rec["inspected_at"], rec["verdict"], rec["reasons"], _num(rec["od_mm"]),
                     _num(rec["pocket_mm"]), _num(rec["concentricity_mm"]), rec["worst_grade"], _num(rec["anomaly_score"]),
                     rec["program_version"], rec["spec_hash"], model_hash, station, operator, image_hash, image_path,
                     reinspection_of, prev, rh),
                )
                iid = cur.lastrowid
                for d in rec["defects"]:
                    self.conn.execute(
                        "INSERT INTO defects(inspection_id, kind, grade, length_mm, area_px, x0, y0, x1, y1) VALUES (?,?,?,?,?,?,?,?,?)",
                        (iid, d["kind"], d["grade"], _num(d["length_mm"]), int(d["area_px"]), *[int(x) for x in d["bbox"]]),
                    )
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
        return rh

    def inspections(self, lot_id: str | None = None, limit: int | None = None, *, station: str | None = None,
                    since: str | None = None, until: str | None = None, serial: str | None = None) -> list[dict]:
        """Rows in chronological order; `limit` keeps the newest N (None or <= 0 = all)."""
        where, args = [], []
        for col, val in (("lot_id", lot_id), ("station", station), ("serial", serial)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if since:
            where.append("inspected_at>=?")
            args.append(since)
        if until:
            where.append("inspected_at<?")
            args.append(until)
        q = "SELECT * FROM inspections" + (" WHERE " + " AND ".join(where) if where else "")
        if limit and limit > 0:
            q += " ORDER BY id DESC LIMIT ?"
            args.append(int(limit))
            rows = self._q(q, args)[::-1]
        else:
            rows = self._q(q + " ORDER BY id", args)
        return [dict(r) for r in rows]

    def inspection(self, inspection_id: int) -> dict | None:
        row = self._q1("SELECT * FROM inspections WHERE id=?", (inspection_id,))
        return dict(row) | {"defects": self.defects_for(inspection_id)} if row else None

    def defects_for(self, inspection_id: int) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM defects WHERE inspection_id=? ORDER BY id", (inspection_id,))]

    def serial_history(self, serial: str) -> list[dict]:
        rows = self.inspections(serial=serial)
        for r in rows:
            r["defects"] = self.defects_for(r["id"])
            r["dispositions"] = self.dispositions_for(r["id"])
        return rows

    def lot_summary(self, lot_id: str, *, since: str | None = None, until: str | None = None,
                    station: str | None = None) -> dict:
        """Yield figures exclude ERROR rows (no part was measured). First-pass yield counts the
        first inspection of each serial, final yield the latest one."""
        rows = self.inspections(lot_id, since=since, until=until, station=station)
        measured = [r for r in rows if r["verdict"] != VERDICT_ERROR]
        n_error = len(rows) - len(measured)
        first, last = {}, {}
        for r in measured:
            first.setdefault(r["serial"], r)
            last[r["serial"]] = r
        n_pass = sum(r["verdict"] == VERDICT_PASS for r in measured)
        reasons: dict[str, int] = {}
        for r in measured:
            for x in filter(None, r["reasons"].split(";")):
                reasons[x] = reasons.get(x, 0) + 1
        ng_ids = [r["id"] for r in measured if r["verdict"] == VERDICT_NG]
        open_ng = 0
        if ng_ids:
            q = f"SELECT COUNT(*) FROM inspections i WHERE i.id IN ({','.join('?' * len(ng_ids))}) AND NOT EXISTS (SELECT 1 FROM dispositions d WHERE d.inspection_id=i.id)"
            open_ng = int(self._q1(q, ng_ids)[0])
        fp = sum(r["verdict"] == VERDICT_PASS for r in first.values())
        fy = sum(r["verdict"] == VERDICT_PASS for r in last.values())
        return {
            "lot_id": lot_id, "n_inspections": len(rows), "n_error": n_error, "n": len(measured),
            "n_pass": n_pass, "n_ng": len(measured) - n_pass, "n_serials": len(first),
            "yield": n_pass / len(measured) if measured else None,
            "first_pass_yield": fp / len(first) if first else None,
            "final_yield": fy / len(last) if last else None,
            "open_ng": open_ng,
            "ng_pareto": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        }

    # -- dispositions ----------------------------------------------------------
    def add_disposition(self, inspection_id: int, decision: str, decided_by: str, note: str | None = None,
                        mrb_ref: str | None = None) -> int:
        if decision not in DISPOSITIONS:
            raise ValueError(f"decision must be one of {DISPOSITIONS}")
        if not decided_by:
            raise ValueError("decided_by is required")
        with self._lock:
            row = self.conn.execute("SELECT verdict FROM inspections WHERE id=?", (inspection_id,)).fetchone()
            if row is None:
                raise ValueError(f"inspection {inspection_id} not found")
            cur = self.conn.execute(
                "INSERT INTO dispositions(inspection_id, decision, decided_by, decided_at, note, mrb_ref) VALUES (?,?,?,?,?,?)",
                (inspection_id, decision, decided_by, utc_now(), note, mrb_ref))
            return int(cur.lastrowid)

    def dispositions_for(self, inspection_id: int) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM dispositions WHERE inspection_id=? ORDER BY id", (inspection_id,))]

    def open_ng(self, lot_id: str | None = None) -> list[dict]:
        q = """SELECT i.* FROM inspections i WHERE i.verdict='NG'
               AND NOT EXISTS (SELECT 1 FROM dispositions d WHERE d.inspection_id=i.id)"""
        args: list = []
        if lot_id:
            q += " AND i.lot_id=?"
            args.append(lot_id)
        return [dict(r) for r in self._q(q + " ORDER BY i.id", args)]

    def close_lot(self, lot_id: str, decision: str, plan: dict, sample: list[str], n_defective: int,
                  n_critical: int, decided_by: str, note: str | None = None) -> dict:
        if decision not in ("ACCEPT", "REJECT", "HOLD"):
            raise ValueError("decision must be ACCEPT, REJECT or HOLD")
        with self._lock:
            if self.conn.execute("SELECT 1 FROM lots WHERE lot_id=?", (lot_id,)).fetchone() is None:
                raise ValueError(f"lot {lot_id} not found")
            head = self.verify_chain()["head"]
            now = utc_now()
            self.conn.execute(
                """INSERT INTO lot_dispositions(lot_id, decision, plan_json, sample_json, n_defective, n_critical,
                   decided_by, decided_at, note, chain_head) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (lot_id, decision, json.dumps(plan), json.dumps(sample), n_defective, n_critical, decided_by, now, note, head))
            if decision != "HOLD":
                self.conn.execute("UPDATE lots SET status='closed', closed_at=? WHERE lot_id=?", (now, lot_id))
            self.anchor_head(f"lot {lot_id} {decision}")
        return {"lot_id": lot_id, "decision": decision, "decided_at": now, "chain_head": head}

    def lot_dispositions(self, lot_id: str) -> list[dict]:
        rows = self._q("SELECT * FROM lot_dispositions WHERE lot_id=? ORDER BY id", (lot_id,))
        return [dict(r) | {"plan_json": json.loads(r["plan_json"]), "sample_json": json.loads(r["sample_json"])} for r in rows]

    # -- control limits --------------------------------------------------------
    def freeze_limits(self, part_number: str, characteristic: str, limits: dict, frozen_by: str,
                      source_lot: str | None = None) -> int:
        with self._lock:
            self.conn.execute("UPDATE control_limits SET active=0 WHERE part_number=? AND characteristic=? AND active=1",
                              (part_number, characteristic))
            cur = self.conn.execute(
                """INSERT INTO control_limits(part_number, characteristic, n, center, ucl, lcl, r_bar, r_ucl, r_lcl,
                   source_lot, frozen_by, frozen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (part_number, characteristic, int(limits["n"]), float(limits["center"]), float(limits["ucl"]),
                 float(limits["lcl"]), float(limits["r_bar"]), float(limits["r_ucl"]), float(limits["r_lcl"]),
                 source_lot, frozen_by, utc_now()))
            return int(cur.lastrowid)

    def active_limits(self, part_number: str, characteristic: str | None = None) -> dict[str, dict]:
        q = "SELECT * FROM control_limits WHERE part_number=? AND active=1"
        args = [part_number]
        if characteristic:
            q += " AND characteristic=?"
            args.append(characteristic)
        return {r["characteristic"]: dict(r) for r in self._q(q, args)}

    # -- calibrations ----------------------------------------------------------
    def add_calibration(self, px_per_mm: float, gauge_diameter_mm: float, *, station: str | None = None,
                        operator: str | None = None, gauge_id: str | None = None, image_hash: str | None = None,
                        spec_h: str | None = None, note: str | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO calibrations(ts, station, operator, gauge_id, gauge_diameter_mm, px_per_mm, image_sha256,
                   spec_hash, note) VALUES (?,?,?,?,?,?,?,?,?)""",
                (utc_now(), station, operator, gauge_id, float(gauge_diameter_mm), float(px_per_mm), image_hash, spec_h, note))
            return int(cur.lastrowid)

    def calibrations(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM calibrations ORDER BY id DESC LIMIT ?", (limit,))][::-1]

    # -- events / audit -----------------------------------------------------
    def add_event(self, kind: str, payload: dict, lot_id: str | None = None, ts: str | None = None) -> int:
        with self._lock:
            cur = self.conn.execute("INSERT INTO events(ts, lot_id, kind, payload) VALUES (?,?,?,?)",
                                    (ts or utc_now(), lot_id, kind, json.dumps(payload, ensure_ascii=False, default=str)))
            return int(cur.lastrowid)

    def events(self, kind: str | None = None, limit: int | None = 100, *, lot_id: str | None = None,
               unacked_only: bool = False, prefix: str | None = None) -> list[dict]:
        where, args = [], []
        if kind:
            where.append("kind=?")
            args.append(kind)
        if prefix:
            where.append("kind LIKE ?")
            args.append(prefix + "%")
        if lot_id:
            where.append("lot_id=?")
            args.append(lot_id)
        if unacked_only:
            where.append("acked_at IS NULL")
        q = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC"
        if limit and limit > 0:
            q += " LIMIT ?"
            args.append(int(limit))
        rows = self._q(q, args)
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows][::-1]

    def ack_event(self, event_id: int, by: str, note: str | None = None) -> bool:
        if not by:
            raise ValueError("acknowledger is required")
        with self._lock:
            cur = self.conn.execute("UPDATE events SET acked_by=?, acked_at=?, ack_note=? WHERE id=? AND acked_at IS NULL",
                                    (by, utc_now(), note, event_id))
            return cur.rowcount == 1

    def anchor_head(self, note: str | None = None) -> dict:
        """Record the current chain head; export it somewhere outside this database too."""
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*), MAX(record_hash) FROM inspections").fetchone()
            n = int(row[0])
            head = self.conn.execute("SELECT record_hash FROM inspections ORDER BY id DESC LIMIT 1").fetchone()
            head = head[0] if head else GENESIS
            self.conn.execute("INSERT INTO anchors(ts, n_records, head, note) VALUES (?,?,?,?)", (utc_now(), n, head, note))
        return {"n_records": n, "head": head}

    def last_anchor(self) -> dict | None:
        row = self._q1("SELECT * FROM anchors ORDER BY id DESC LIMIT 1")
        return dict(row) if row else None

    def verify_chain(self) -> dict:
        """Recompute every hash in one read; report the first broken link, and check the last anchor."""
        with self._lock:
            rows = self.conn.execute("SELECT * FROM inspections ORDER BY id").fetchall()
            drows = self.conn.execute("SELECT * FROM defects ORDER BY inspection_id, id").fetchall()
            anchor = self.last_anchor()
        by_insp: dict[int, list[dict]] = {}
        for d in drows:
            by_insp.setdefault(d["inspection_id"], []).append(
                {"kind": d["kind"], "grade": d["grade"], "length_mm": d["length_mm"], "area_px": d["area_px"],
                 "bbox": [d["x0"], d["y0"], d["x1"], d["y1"]]})
        prev, n = GENESIS, 0
        heads: dict[int, str] = {}
        for r in rows:
            rec = {k: r[k] for k in _RECORD_FIELDS}
            rec["defects"] = by_insp.get(r["id"], [])
            expected = _hash(prev, canonical_record(rec))
            if r["prev_hash"] != prev or r["record_hash"] != expected:
                return {"ok": False, "checked": n, "broken_at_id": r["id"], "serial": r["serial"], "head": prev}
            prev = r["record_hash"]
            n += 1
            heads[n] = prev
        out = {"ok": True, "checked": n, "head": prev}
        if anchor:
            an, ah = anchor["n_records"], anchor["head"]
            if an > n or heads.get(an, GENESIS if an == 0 else None) != ah:
                out.update(ok=False, anchor_mismatch={"anchored_n": an, "anchored_head": ah, "current_n": n},
                           note="records after the last anchor were removed or the anchored prefix changed")
            else:
                out["anchor"] = {"n_records": an, "ts": anchor["ts"]}
        return out
