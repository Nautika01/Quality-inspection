"""Streamlit dashboard over the inspection database.

    streamlit run -m qinspect.dashboard -- --db qinspect.db
    (or: QINSPECT_DB=qinspect.db streamlit run src/qinspect/dashboard.py)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

from qinspect.db import InspectionDB
from qinspect.spc import capability, xbar_r
from qinspect.spec import PartSpec


def _db_path() -> str:
    if "--db" in sys.argv:
        return sys.argv[sys.argv.index("--db") + 1]
    return os.environ.get("QINSPECT_DB", "qinspect.db")


def main() -> None:
    st.set_page_config(page_title="qinspect dashboard", layout="wide")
    st.title("품질 검사 대시보드")
    db = InspectionDB(_db_path())
    spec = PartSpec()

    lots = db.lots()
    if not lots:
        st.info("검사 기록이 없습니다. `qinspect run-line` 또는 API로 데이터를 넣어 주세요.")
        return
    lot_ids = [l["lot_id"] for l in lots]
    col_a, col_b = st.columns([3, 1])
    lot_id = col_a.selectbox("Lot", lot_ids, index=len(lot_ids) - 1)
    auto = col_b.toggle("자동 새로고침 (5s)", value=False)

    rows = pd.DataFrame(db.inspections(lot_id))
    s = db.lot_summary(lot_id)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("검사 수", s["n"])
    k2.metric("수율", f"{100 * s['yield']:.1f} %")
    k3.metric("NG", s["n_ng"])
    audit = db.verify_chain()
    k4.metric("감사 체인", "정상" if audit["ok"] else "훼손", f"{audit['checked']}건 검증")

    left, right = st.columns(2)
    with left:
        st.subheader("NG 사유 파레토")
        if s["ng_pareto"]:
            st.bar_chart(pd.Series(s["ng_pareto"]))
        else:
            st.write("NG 없음")
    with right:
        st.subheader("외경 X̄ 관리도")
        od = rows.od_mm.dropna().to_numpy()
        if len(od) >= 10:
            ch = xbar_r(od, n=5, baseline=10)
            chart = pd.DataFrame({"X̄": ch.xbar, "UCL": ch.ucl, "LCL": ch.lcl, "CL": ch.center})
            st.line_chart(chart)
            cap = capability(od, spec.outer_diameter.lsl, spec.outer_diameter.usl, ch.sigma_within)
            st.caption(f"Cp {cap['cp']:.2f} · Cpk {cap['cpk']:.2f} · Ppk {cap['ppk']:.2f} · "
                       f"{'관리 이탈' if ch.out_of_control else '관리 상태'} "
                       f"(위반 subgroup: {', '.join(str(k + 1) for k in sorted(ch.violations)) or '-'})")
        else:
            st.write("데이터 부족")

    st.subheader("최근 알람")
    alarms = [e for e in db.events(limit=500) if e["kind"].startswith("alarm:") and e.get("lot_id") == lot_id]
    if alarms:
        st.dataframe(pd.DataFrame([{"ts": e["ts"], "kind": e["kind"], **{k: v for k, v in e["payload"].items()
                                    if k not in ("ts", "station", "lot_id", "kind")}} for e in alarms[-20:]]),
                     width="stretch")
    else:
        st.write("알람 없음")

    st.subheader("최근 NG")
    ng = rows[rows.verdict == "NG"].tail(20)[["serial", "inspected_at", "reasons", "od_mm", "pocket_mm",
                                             "concentricity_mm", "worst_grade", "anomaly_score"]]
    st.dataframe(ng, width="stretch")

    st.subheader("치수 분포")
    c1, c2 = st.columns(2)
    c1.caption(f"외경 (LSL {spec.outer_diameter.lsl} / USL {spec.outer_diameter.usl})")
    c1.bar_chart(np.histogram(rows.od_mm.dropna(), bins=30)[0])
    c2.caption(f"포켓 직경 (LSL {spec.pocket_diameter.lsl} / USL {spec.pocket_diameter.usl})")
    c2.bar_chart(np.histogram(rows.pocket_mm.dropna(), bins=30)[0])

    if auto:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__" or st.runtime.exists():
    main()
