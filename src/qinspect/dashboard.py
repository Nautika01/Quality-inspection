"""Streamlit dashboard over the inspection database.

    qinspect dashboard --db qinspect.db
    (equivalent: QINSPECT_DB=qinspect.db streamlit run src/qinspect/dashboard.py -- --db qinspect.db)

The spec shown per lot is the one the lot was inspected with (stored in `lots.spec_json`),
timestamps are converted to QINSPECT_TZ (default Asia/Seoul).
"""

from __future__ import annotations

import os
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from qinspect.db import InspectionDB
from qinspect.reasons import describe
from qinspect.spc import ControlLimits, capability, evaluate, xbar_r
from qinspect.spec import PartSpec


def _arg(flag: str, env: str, default: str) -> str:
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return os.environ.get(env, default)


def _local(series: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(ZoneInfo(tz)).dt.strftime("%m-%d %H:%M:%S")


def _xbar_chart(vals: np.ndarray, frozen: dict | None, subgroup: int, baseline: int):
    if frozen:
        return evaluate(vals, ControlLimits.from_dict(frozen)), "동결 한계"
    return xbar_r(vals, n=subgroup, baseline=baseline), f"Phase I {baseline} subgroup"


def main() -> None:
    st.set_page_config(page_title="qinspect dashboard", layout="wide")
    st.title("품질 검사 대시보드")
    tz = _arg("--tz", "QINSPECT_TZ", "Asia/Seoul")
    db = InspectionDB(_arg("--db", "QINSPECT_DB", "qinspect.db"))

    lots = db.lots()
    if not lots:
        st.info("검사 기록이 없습니다. `qinspect run-line` 또는 API로 데이터를 넣어 주세요.")
        return
    lot_ids = [l["lot_id"] for l in lots]
    col_a, col_b, col_c = st.columns([3, 1, 1])
    lot_id = col_a.selectbox("Lot", lot_ids, index=len(lot_ids) - 1)
    subgroup = col_b.selectbox("부분군 크기", [2, 3, 4, 5, 6, 7, 8, 9, 10], index=3)
    auto = col_c.toggle("자동 새로고침 (5s)", value=False)

    spec = db.lot_spec(lot_id) or PartSpec()
    lot_row = next(l for l in lots if l["lot_id"] == lot_id)
    st.caption(f"품번 {spec.part_number} Rev.{spec.revision} · 규격 해시 `{lot_row['spec_hash']}` · 상태 {lot_row['status']} · 표시 시간대 {tz}")

    rows = pd.DataFrame(db.inspections(lot_id))
    s = db.lot_summary(lot_id)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("검사 수", s["n"], f"ERROR {s['n_error']}건 제외" if s["n_error"] else None)
    k2.metric("수율", f"{100 * s['yield']:.1f} %" if s["yield"] is not None else "-")
    k3.metric("최종 수율(재검 반영)", f"{100 * s['final_yield']:.1f} %" if s["final_yield"] is not None else "-")
    k4.metric("NG / 미처분", f"{s['n_ng']} / {s['open_ng']}")
    audit = db.verify_chain()
    k5.metric("감사 체인", "정상" if audit["ok"] else "훼손", f"{audit['checked']}건 검증")
    unacked = [e for e in db.events(prefix="alarm:", limit=None, lot_id=lot_id, unacked_only=True)]
    k6.metric("미확인 알람", len(unacked))

    left, right = st.columns(2)
    with left:
        st.subheader("NG 사유 파레토")
        if s["ng_pareto"]:
            par = pd.Series(s["ng_pareto"])
            par.index = [f"{describe(k)} ({k})" for k in par.index]
            st.bar_chart(par)
        else:
            st.write("NG 없음")
    with right:
        st.subheader("관리도")
        frozen_all = db.active_limits(spec.part_number)
        chars = [("od_mm", "외경", spec.outer_diameter)]
        if spec.has_pocket:
            chars += [("pocket_mm", "포켓 직경", spec.pocket_diameter), ("concentricity_mm", "동심도", None)]
        ch_key = st.radio("특성", [c[0] for c in chars], format_func=lambda k: dict((c[0], c[1]) for c in chars)[k], horizontal=True)
        dim = dict((c[0], c[2]) for c in chars)[ch_key]
        vals = rows[rows.verdict != "ERROR"][ch_key].dropna().to_numpy(dtype=float) if len(rows) else np.array([])
        if len(vals) >= 2 * subgroup:
            ch, src = _xbar_chart(vals, frozen_all.get(ch_key), subgroup, 25)
            st.caption(f"한계 출처: {src}")
            st.line_chart(pd.DataFrame({"X̄": ch.xbar, "UCL": ch.ucl, "LCL": ch.lcl, "CL": ch.center}))
            st.line_chart(pd.DataFrame({"R": ch.ranges, "R UCL": ch.r_ucl, "R̄": ch.r_bar}))
            if dim is not None:
                cap = capability(vals, dim.lsl, dim.usl, ch.sigma_within)
                fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
                st.caption(f"Cp {fmt(cap.get('cp'))} · Cpk {fmt(cap.get('cpk'))} · Ppk {fmt(cap.get('ppk'))} · "
                           f"{'관리 이탈' if ch.out_of_control else '관리 상태'} "
                           f"(위반 subgroup: {', '.join(str(k + 1) for k in sorted(ch.violations)) or '-'})")
        else:
            st.write("데이터 부족")

    st.subheader("최근 알람")
    alarms = db.events(prefix="alarm:", limit=50, lot_id=lot_id)
    if alarms:
        tbl = pd.DataFrame([{"id": e["id"], "시각": e["ts"], "종류": e["kind"].replace("alarm:", ""),
                             "확인": e["acked_by"] or "", **{k: v for k, v in e["payload"].items()
                             if k not in ("ts", "station", "lot_id", "kind", "event_id")}} for e in alarms[-20:]])
        tbl["시각"] = _local(tbl["시각"], tz)
        st.dataframe(tbl, width="stretch")
        with st.form("ack"):
            c1, c2, c3 = st.columns([1, 1, 2])
            eid = c1.number_input("알람 id", min_value=1, step=1, value=int(unacked[-1]["id"]) if unacked else 1)
            who = c2.text_input("확인자")
            note = c3.text_input("메모")
            if st.form_submit_button("알람 확인") and who:
                st.success("확인됨" if db.ack_event(int(eid), who, note or None) else "이미 확인됐거나 없는 알람")
    else:
        st.write("알람 없음")

    st.subheader("최근 NG (미처분 우선)")
    if len(rows):
        ng = rows[rows.verdict == "NG"].tail(50).copy()
        open_ids = {r["id"] for r in db.open_ng(lot_id)}
        ng["처분"] = ng.id.map(lambda i: "미처분" if i in open_ids else "완료")
        ng["사유"] = ng.reasons.map(lambda r: ", ".join(describe(x) for x in r.split(";") if x))
        ng["시각"] = _local(ng.inspected_at, tz)
        st.dataframe(ng[["id", "serial", "시각", "사유", "od_mm", "pocket_mm", "concentricity_mm", "worst_grade",
                         "anomaly_score", "operator", "처분"]].sort_values(["처분", "id"], ascending=[False, False]).head(20),
                     width="stretch")

    st.subheader("치수 분포")
    c1, c2 = st.columns(2)
    if len(rows):
        od = rows.od_mm.dropna()
        c1.caption(f"외경 (LSL {spec.outer_diameter.lsl} / USL {spec.outer_diameter.usl}, n={len(od)})")
        if len(od):
            h, edges = np.histogram(od, bins=30)
            c1.bar_chart(pd.Series(h, index=np.round(edges[:-1], 3)))
        if spec.has_pocket:
            pk = rows.pocket_mm.dropna()
            c2.caption(f"포켓 직경 (LSL {spec.pocket_diameter.lsl} / USL {spec.pocket_diameter.usl}, n={len(pk)})")
            if len(pk):
                h, edges = np.histogram(pk, bins=30)
                c2.bar_chart(pd.Series(h, index=np.round(edges[:-1], 3)))

    if auto:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__" or st.runtime.exists():
    main()
