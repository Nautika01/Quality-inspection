"""Simulated production line: camera -> inspection -> DB -> message bus -> alarms.

The message bus is an abstraction over MQTT (what a PLC/MES/SCADA would subscribe
to). `InMemoryBus` runs everywhere and keeps the messages for tests; `MqttBus`
publishes to a real broker when `paho-mqtt` and a broker are available.
Topics follow `qinspect/<station>/<result|alarm|heartbeat>`.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .anomaly import AnomalyModel
from .db import InspectionDB
from .inspector import InspectionResult, inspect_array
from .spc import xbar_r
from .spec import PartSpec
from .synth import DEFECT_TYPES, render_part


class InMemoryBus:
    def __init__(self):
        self.messages: list[tuple[str, dict]] = []
        self.subscribers: list[Callable[[str, dict], None]] = []

    def publish(self, topic: str, payload: dict) -> None:
        self.messages.append((topic, payload))
        for cb in self.subscribers:
            cb(topic, payload)

    def subscribe(self, cb: Callable[[str, dict], None]) -> None:
        self.subscribers.append(cb)

    def close(self) -> None:
        pass


class MqttBus:
    def __init__(self, host: str = "localhost", port: int = 1883, client_id: str = "qinspect"):
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()

    def publish(self, topic: str, payload: dict) -> None:
        self.client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1)

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
    """Rolling X-bar/R monitor that raises an alarm once per new violated subgroup."""

    subgroup: int = 5
    baseline_subgroups: int = 10
    values: list[float] = field(default_factory=list)
    _alarmed: set[int] = field(default_factory=set)

    def add(self, value: float) -> list[dict]:
        self.values.append(value)
        k = len(self.values) // self.subgroup
        if len(self.values) % self.subgroup or k < self.baseline_subgroups + 1:
            return []
        chart = xbar_r(np.array(self.values), n=self.subgroup, baseline=self.baseline_subgroups)
        alarms = []
        for idx, rules in chart.violations.items():
            if idx >= self.baseline_subgroups and idx not in self._alarmed:
                self._alarmed.add(idx)
                alarms.append({"subgroup": idx + 1, "xbar": float(chart.xbar[idx]), "rules": rules,
                               "ucl": chart.ucl, "lcl": chart.lcl})
        return alarms


class SimulatedCamera:
    """Yields (serial, image) like a frame grabber would; includes tool-wear drift."""

    def __init__(self, spec: PartSpec, lot_id: str, seed: int = 0, defect_rate: float = 0.1,
                 drift_after: int = 120, drift_mm_per_part: float = 0.0007):
        self.spec, self.lot_id = spec, lot_id
        self.rng = np.random.default_rng(seed)
        self.defect_rate, self.drift_after, self.drift = defect_rate, drift_after, drift_mm_per_part
        self.i = 0

    def grab(self) -> tuple[str, np.ndarray]:
        self.i += 1
        s = self.spec
        od = self.rng.normal(s.outer_diameter.nominal, 0.02) + max(0, self.i - self.drift_after) * self.drift
        pk = self.rng.normal(s.pocket_diameter.nominal, 0.012)
        off = tuple(self.rng.normal(0, 0.015, 2))
        defects = (str(self.rng.choice(DEFECT_TYPES)),) if self.rng.random() < self.defect_rate else ()
        img = render_part(s, self.rng, od_mm=od, pocket_mm=pk, pocket_offset_mm=off, defects=defects)
        return f"{self.lot_id}-{self.i:05d}", img


class Station:
    """One inspection station. `process()` is what a real camera trigger would call."""

    def __init__(self, spec: PartSpec, db: InspectionDB, bus, *, station_id: str = "line1",
                 lot_id: str = "LOT-SIM", anomaly_model: AnomalyModel | None = None,
                 model_hash: str | None = None, subgroup: int = 5, baseline_subgroups: int = 10):
        self.spec, self.db, self.bus = spec, db, bus
        self.station_id, self.lot_id = station_id, lot_id
        self.model, self.model_hash = anomaly_model, model_hash
        self.spc = SpcMonitor(subgroup, baseline_subgroups)
        self.recent = deque(maxlen=50)

    def _alarm(self, kind: str, detail: dict) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "station": self.station_id,
                   "lot_id": self.lot_id, "kind": kind, **detail}
        self.db.add_event(f"alarm:{kind}", payload, self.lot_id)
        self.bus.publish(f"qinspect/{self.station_id}/alarm", payload)

    def process(self, serial: str, img: np.ndarray) -> InspectionResult:
        res = inspect_array(img, serial, self.spec, self.model)
        self.db.add_inspection(res, self.lot_id, self.spec, self.model_hash, self.station_id)
        self.bus.publish(f"qinspect/{self.station_id}/result", res.to_row() | {"lot_id": self.lot_id})

        if res.worst_grade == "critical":
            self._alarm("critical_defect", {"serial": serial, "reasons": res.reasons})
        if res.od_mm is not None:
            for a in self.spc.add(res.od_mm):
                self._alarm("spc_violation", {"characteristic": "od_mm", **a})
        self.recent.append(res.verdict == "PASS")
        if len(self.recent) == self.recent.maxlen and np.mean(self.recent) < 0.7:
            self._alarm("yield_drop", {"window": self.recent.maxlen, "yield": float(np.mean(self.recent))})
            self.recent.clear()
        return res


def run_line(n: int, db_path: str | Path, *, spec: PartSpec | None = None, bus_url: str | None = None,
             rate_hz: float = 0.0, seed: int = 0, lot_id: str = "LOT-SIM", model_path: str | None = None,
             on_result: Callable[[InspectionResult], None] | None = None) -> dict:
    spec = spec or PartSpec()
    db, bus = InspectionDB(db_path), make_bus(bus_url)
    model = mh = None
    if model_path:
        from .anomaly import model_hash

        model, mh = AnomalyModel.load(model_path), model_hash(model_path)
    st = Station(spec, db, bus, lot_id=lot_id, anomaly_model=model, model_hash=mh)
    cam = SimulatedCamera(spec, lot_id, seed=seed)
    t0 = time.time()
    for _ in range(n):
        serial, img = cam.grab()
        res = st.process(serial, img)
        if on_result:
            on_result(res)
        if rate_hz > 0:
            time.sleep(1 / rate_hz)
    bus.publish(f"qinspect/{st.station_id}/heartbeat", {"ts": time.time(), "inspected": n})
    summary = db.lot_summary(lot_id) | {"alarms": db.events(limit=1000), "elapsed_s": round(time.time() - t0, 1),
                                        "audit": db.verify_chain()}
    summary["alarms"] = [e for e in summary["alarms"] if e["kind"].startswith("alarm:")]
    bus.close()
    db.close()
    return summary
