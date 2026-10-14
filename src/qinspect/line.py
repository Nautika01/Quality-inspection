"""Simulated production line: camera -> inspection -> DB -> message bus -> alarms.

The message bus is an abstraction over MQTT (what a PLC/MES/SCADA would subscribe
to). `InMemoryBus` runs everywhere and keeps the messages for tests; `MqttBus`
publishes to a real broker when `paho-mqtt` and a broker are available.
Topics follow `qinspect/<station>/<result|alarm|heartbeat>`.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .anomaly import AnomalyModel
from .db import InspectionDB, image_sha256
from .inspector import VERDICT_ERROR, VERDICT_PASS, InspectionResult, inspect_array, utc_now
from .spc import RECOMMENDED_BASELINE, ControlLimits, western_electric, xbar_r
from .spec import PartSpec
from .synth import DEFECT_TYPES, render_part

log = logging.getLogger(__name__)
CHARACTERISTICS = ("od_mm", "pocket_mm", "concentricity_mm")


class InMemoryBus:
    def __init__(self):
        self.messages: list[tuple[str, dict]] = []
        self.subscribers: list[Callable[[str, dict], None]] = []

    def publish(self, topic: str, payload: dict) -> bool:
        self.messages.append((topic, payload))
        for cb in self.subscribers:
            cb(topic, payload)
        return True

    def subscribe(self, cb: Callable[[str, dict], None]) -> None:
        self.subscribers.append(cb)

    def close(self) -> None:
        pass


class MqttBus:
    """Publishes JSON to an MQTT broker. Connection failures at start raise a clear error;
    drops during the run are retried by paho's network loop and counted in `errors`."""

    def __init__(self, host: str = "localhost", port: int = 1883, client_id: str = "qinspect", connect_timeout: float = 5.0):
        import paho.mqtt.client as mqtt

        self.errors = 0
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        try:
            self.client.connect(host, port, keepalive=30)
        except OSError as exc:
            raise ConnectionError(f"MQTT broker {host}:{port} unreachable: {exc}") from exc
        self.client.loop_start()
        t0 = time.time()
        while not self.client.is_connected() and time.time() - t0 < connect_timeout:
            time.sleep(0.05)
        if not self.client.is_connected():
            self.client.loop_stop()
            raise ConnectionError(f"MQTT broker {host}:{port} did not accept the connection within {connect_timeout}s")

    def publish(self, topic: str, payload: dict) -> bool:
        info = self.client.publish(topic, json.dumps(payload, ensure_ascii=False, default=str), qos=1)
        if info.rc != 0:
            self.errors += 1
            log.warning("MQTT publish failed rc=%s topic=%s", info.rc, topic)
            return False
        return True

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


def make_bus(url: str | None):
    """`None` / `memory` -> InMemoryBus, `mqtt://host[:port]` -> MqttBus."""
    if not url or url == "memory":
        return InMemoryBus()
    if url.startswith("mqtt://"):
        hostport = url[len("mqtt://"):]
        host, _, port = hostport.partition(":")
        return MqttBus(host or "localhost", int(port or 1883))
    raise ValueError(f"unknown bus url: {url}")


@dataclass
class SpcMonitor:
    """Rolling X-bar/R monitor for one characteristic.

    Until limits are frozen it estimates them from the first `baseline_subgroups` (Phase I)
    and then keeps them fixed; with `limits` given (from the DB) it starts in Phase II at
    once. Memory is bounded: only the last 8 subgroup statistics are kept once limits exist.
    """

    subgroup: int = 5
    baseline_subgroups: int = RECOMMENDED_BASELINE
    limits: ControlLimits | None = None
    _buf: list[float] = field(default_factory=list)
    _phase1: list[float] = field(default_factory=list)
    _recent_x: deque = field(default_factory=lambda: deque(maxlen=8))
    _recent_r: deque = field(default_factory=lambda: deque(maxlen=8))
    _n_subgroups: int = 0

    def add(self, value: float) -> list[dict]:
        if value is None or not np.isfinite(value):
            return []
        self._buf.append(float(value))
        if len(self._buf) < self.subgroup:
            return []
        sg = np.array(self._buf)
        self._buf = []
        self._n_subgroups += 1
        if self.limits is None:
            self._phase1.extend(sg.tolist())
            if self._n_subgroups < self.baseline_subgroups:
                return []
            chart = xbar_r(np.array(self._phase1), n=self.subgroup)
            self.limits = chart.limits
            self._phase1 = []
            return []  # Phase I points are not alarmed
        self._recent_x.append(float(sg.mean()))
        self._recent_r.append(float(sg.max() - sg.min()))
        rules = western_electric(np.array(self._recent_x), self.limits.center, self.limits.sigma_xbar)
        last = len(self._recent_x) - 1
        viol = list(rules.get(last, []))
        r = self._recent_r[-1]
        if r > self.limits.r_ucl or r < self.limits.r_lcl:
            viol.append("R-chart:out_of_limits")
        if not viol:
            return []
        return [{"subgroup": self._n_subgroups, "xbar": self._recent_x[-1], "range": r, "rules": viol,
                 "ucl": self.limits.ucl, "lcl": self.limits.lcl, "center": self.limits.center}]


class SimulatedCamera:
    """Yields (serial, image) like a frame grabber would; includes tool-wear drift and
    occasional empty-nest frames (`error_rate`)."""

    def __init__(self, spec: PartSpec, lot_id: str, seed: int = 0, defect_rate: float = 0.1,
                 drift_after: int = 120, drift_mm_per_part: float = 0.0007, error_rate: float = 0.01):
        self.spec, self.lot_id = spec, lot_id
        self.rng = np.random.default_rng(seed)
        self.defect_rate, self.drift_after, self.drift, self.error_rate = defect_rate, drift_after, drift_mm_per_part, error_rate
        self.i = 0

    def grab(self) -> tuple[str, np.ndarray]:
        self.i += 1
        s = self.spec
        serial = f"{self.lot_id}-{self.i:05d}"
        if self.rng.random() < self.error_rate:  # empty fixture
            return serial, np.clip(self.rng.normal(30, 3, (s.image_size, s.image_size)), 0, 255).astype(np.uint8)
        od = self.rng.normal(s.outer_diameter.nominal, 0.02) + max(0, self.i - self.drift_after) * self.drift
        pk = self.rng.normal(s.pocket_diameter.nominal, 0.012) if s.has_pocket else None
        off = tuple(self.rng.normal(0, 0.015, 2))
        defects = (str(self.rng.choice(DEFECT_TYPES)),) if self.rng.random() < self.defect_rate else ()
        img = render_part(s, self.rng, od_mm=od, pocket_mm=pk, pocket_offset_mm=off, defects=defects)
        return serial, img


class Station:
    """One inspection station. `process()` is what a real camera trigger would call."""

    def __init__(self, spec: PartSpec, db: InspectionDB, bus, *, station_id: str = "line1",
                 lot_id: str = "LOT-SIM", anomaly_model: AnomalyModel | None = None,
                 model_hash: str | None = None, subgroup: int = 5, baseline_subgroups: int = RECOMMENDED_BASELINE,
                 operator: str | None = None, image_dir: str | Path | None = None):
        self.spec, self.db, self.bus = spec, db, bus
        self.station_id, self.lot_id, self.operator = station_id, lot_id, operator
        self.model, self.model_hash = anomaly_model, model_hash
        self.image_dir = Path(image_dir) if image_dir else None
        self.db.ensure_lot(lot_id, spec)
        frozen = db.active_limits(spec.part_number)
        chars = CHARACTERISTICS if spec.has_pocket else ("od_mm",)
        self.spc = {c: SpcMonitor(subgroup, baseline_subgroups,
                                  ControlLimits.from_dict(frozen[c]) if c in frozen else None) for c in chars}
        self.recent = deque(maxlen=50)

    def _alarm(self, kind: str, detail: dict) -> None:
        payload = {"ts": utc_now(), "station": self.station_id, "lot_id": self.lot_id, "kind": kind, **detail}
        payload["event_id"] = self.db.add_event(f"alarm:{kind}", payload, self.lot_id)
        self.bus.publish(f"qinspect/{self.station_id}/alarm", payload)

    def process(self, serial: str, img: np.ndarray, reinspection_of: int | None = None) -> InspectionResult:
        res = inspect_array(img, serial, self.spec, self.model)
        img_hash = image_sha256(img)
        img_path = None
        if self.image_dir is not None and res.verdict != VERDICT_PASS:  # keep evidence for NG / ERROR frames
            from PIL import Image

            self.image_dir.mkdir(parents=True, exist_ok=True)
            img_path = str(self.image_dir / f"{serial}_{img_hash[:8]}.png")
            Image.fromarray(np.asarray(img)).save(img_path)
        self.db.add_inspection(res, self.lot_id, self.spec, self.model_hash, self.station_id, operator=self.operator,
                               image_hash=img_hash, image_path=img_path, reinspection_of=reinspection_of)
        self.bus.publish(f"qinspect/{self.station_id}/result", res.to_row() | {"lot_id": self.lot_id})

        if res.verdict == VERDICT_ERROR:  # not a part defect: re-trigger, do not feed SPC / yield
            self._alarm("vision_error", {"serial": serial, "reasons": res.reasons, "detail": res.error_detail})
            return res
        if res.worst_grade == "critical":
            self._alarm("critical_defect", {"serial": serial, "reasons": res.reasons})
        for ch, mon in self.spc.items():
            for a in mon.add(getattr(res, ch)):
                self._alarm("spc_violation", {"characteristic": ch, **a})
        self.recent.append(res.verdict == VERDICT_PASS)
        if len(self.recent) == self.recent.maxlen and np.mean(self.recent) < 0.7:
            self._alarm("yield_drop", {"window": self.recent.maxlen, "yield": float(np.mean(self.recent))})
            self.recent.clear()
        return res

    def freeze_current_limits(self, by: str) -> dict[str, dict]:
        """Persist the limits each monitor is using so the next start skips Phase I."""
        out = {}
        for ch, mon in self.spc.items():
            if mon.limits is not None:
                self.db.freeze_limits(self.spec.part_number, ch, mon.limits.to_dict(), by, self.lot_id)
                out[ch] = mon.limits.to_dict()
        return out


def run_line(n: int, db_path: str | Path, *, spec: PartSpec | None = None, bus_url: str | None = None,
             rate_hz: float = 0.0, seed: int = 0, lot_id: str = "LOT-SIM", model_path: str | None = None,
             on_result: Callable[[InspectionResult], None] | None = None, baseline_subgroups: int = RECOMMENDED_BASELINE,
             operator: str | None = None, image_dir: str | Path | None = None, freeze_by: str | None = None) -> dict:
    spec = spec or PartSpec()
    db, bus = InspectionDB(db_path), make_bus(bus_url)
    try:
        model = mh = None
        if model_path:
            from .anomaly import model_hash

            model, mh = AnomalyModel.load(model_path), model_hash(model_path)
        st = Station(spec, db, bus, lot_id=lot_id, anomaly_model=model, model_hash=mh,
                     baseline_subgroups=baseline_subgroups, operator=operator, image_dir=image_dir)
        cam = SimulatedCamera(spec, lot_id, seed=seed)
        t0 = time.time()
        for _ in range(n):
            serial, img = cam.grab()
            res = st.process(serial, img)
            if on_result:
                on_result(res)
            if rate_hz > 0:
                time.sleep(1 / rate_hz)
        if freeze_by:
            st.freeze_current_limits(freeze_by)
        bus.publish(f"qinspect/{st.station_id}/heartbeat", {"ts": utc_now(), "inspected": n})
        anchor = db.anchor_head(f"run_line {lot_id}")
        alarms = db.events(prefix="alarm:", limit=None, lot_id=lot_id)
        summary = db.lot_summary(lot_id) | {
            "alarms": alarms, "n_alarms": len(alarms), "elapsed_s": round(time.time() - t0, 1),
            "audit": db.verify_chain(), "anchor": anchor, "bus_errors": getattr(bus, "errors", 0),
            "control_limits": {c: (m.limits.to_dict() if m.limits else None) for c, m in st.spc.items()},
        }
    finally:
        bus.close()
        db.close()
    return summary
