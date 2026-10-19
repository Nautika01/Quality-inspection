"""Measurement System Analysis: crossed Gage R&R (ANOVA method, AIAG MSA 4th ed.).

In an automated vision cell the "operator" factor is replaced by the *re-fixturing
condition*: the same part is re-loaded, so position and illumination change while
the true dimension does not. With synthetic parts we can do exactly that by
re-rendering one part under different lighting / placement draws.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inspector import inspect_array
from .spec import PartSpec
from .synth import render_part


@dataclass
class GageRR:
    characteristic: str
    n_parts: int
    n_operators: int
    n_trials: int
    repeatability: float  # EV, sigma
    reproducibility: float  # AV, sigma
    grr: float
    part_variation: float  # PV, sigma
    total_variation: float
    pct_grr_tv: float  # %GRR vs total variation (study variation)
    pct_grr_tol: float  # %GRR vs tolerance (P/T ratio, 6 sigma)
    ndc: int
    tolerance: float

    @property
    def verdict(self) -> str:
        # AIAG guidance: <10 % acceptable, 10-30 % marginal, >30 % unacceptable
        worst = max(self.pct_grr_tv, self.pct_grr_tol)
        return "acceptable" if worst < 10 else "marginal" if worst < 30 else "unacceptable"

    @property
    def suggested_guard_band_mm(self) -> float:
        """Guard band = 2 x GRR sigma (the measurement uncertainty at ~95 %)."""
        return 2 * self.grr

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["verdict"] = self.verdict
        d["suggested_guard_band_mm"] = self.suggested_guard_band_mm
        return d


def _validate_balanced(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Drop NaN values and require a complete, balanced part x operator x trial design."""
    need = {"part", "operator", "value"}
    if not need <= set(df.columns):
        raise ValueError(f"need columns {sorted(need)}")
    d = df.dropna(subset=["value"]).copy()
    d["value"] = d["value"].astype(float)
    p, o = d.part.nunique(), d.operator.nunique()
    counts = d.groupby(["part", "operator"]).value.count()
    full = p * o
    if len(counts) != full:
        missing = full - len(counts)
        raise ValueError(f"unbalanced study: {missing} part x operator cell(s) have no measurement")
    if counts.nunique() != 1:
        bad = counts[counts != counts.max()]
        raise ValueError(f"unbalanced study: unequal trials per cell, e.g. {bad.head(3).to_dict()}")
    r = int(counts.iloc[0])
    if p < 2 or o < 2 or r < 2:
        raise ValueError("need >= 2 parts, >= 2 operators and >= 2 trials")
    return d, p, o, r


def gage_rr_anova(df: pd.DataFrame, tolerance: float, characteristic: str = "value") -> GageRR:
    """Crossed ANOVA Gage R&R. `df` needs columns part, operator, trial, value (balanced design)."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    df, p, o, r = _validate_balanced(df)

    grand = df.value.mean()
    part_means = df.groupby("part").value.mean()
    op_means = df.groupby("operator").value.mean()
    cell_means = df.groupby(["part", "operator"]).value.mean()

    ss_part = o * r * ((part_means - grand) ** 2).sum()
    ss_op = p * r * ((op_means - grand) ** 2).sum()
    ss_cell = r * ((cell_means - grand) ** 2).sum()
    ss_inter = max(ss_cell - ss_part - ss_op, 0.0)
    ss_total = ((df.value - grand) ** 2).sum()
    ss_rep = max(ss_total - ss_cell, 0.0)

    ms_part = ss_part / (p - 1)
    ms_op = ss_op / (o - 1)
    ms_inter = ss_inter / ((p - 1) * (o - 1))
    ms_rep = ss_rep / (p * o * (r - 1))

    # F-test on the interaction (alpha = 0.25 per AIAG); pool it if not significant
    from scipy import stats

    if ms_rep > 0:
        p_inter = float(stats.f.sf(ms_inter / ms_rep, (p - 1) * (o - 1), p * o * (r - 1)))
    else:  # zero repeatability (resolution-limited gauge): F is +inf whenever the interaction MS is non-zero
        p_inter = 0.0 if ms_inter > 0 else 1.0
        warnings.warn("gage_rr_anova: repeat readings are identical; gauge resolution may be too coarse", stacklevel=2)
    if p_inter < 0.25:
        var_rep = ms_rep
        var_inter = max((ms_inter - ms_rep) / r, 0.0)
        var_op = max((ms_op - ms_inter) / (p * r), 0.0)
        var_part = max((ms_part - ms_inter) / (o * r), 0.0)
    else:
        ms_pooled = (ss_rep + ss_inter) / (p * o * (r - 1) + (p - 1) * (o - 1))
        var_rep = ms_pooled
        var_inter = 0.0
        var_op = max((ms_op - ms_pooled) / (p * r), 0.0)
        var_part = max((ms_part - ms_pooled) / (o * r), 0.0)

    ev = float(np.sqrt(var_rep))
    av = float(np.sqrt(var_op + var_inter))
    grr = float(np.sqrt(ev**2 + av**2))
    pv = float(np.sqrt(var_part))
    tv = float(np.sqrt(grr**2 + pv**2))
    if tv == 0:
        raise ValueError("study has no variation: every reading is identical (gauge resolution too coarse for this study)")
    if grr == 0:
        ndc = 99 if pv > 0 else 0
    else:
        ndc = max(int(np.floor(1.41 * pv / grr)), 0)
    return GageRR(
        characteristic=characteristic,
        n_parts=p,
        n_operators=o,
        n_trials=r,
        repeatability=ev,
        reproducibility=av,
        grr=grr,
        part_variation=pv,
        total_variation=tv,
        pct_grr_tv=100 * grr / tv,
        pct_grr_tol=100 * 6 * grr / tolerance,
        ndc=ndc,
        tolerance=tolerance,
    )


def simulate_gage_study(
    spec: PartSpec,
    *,
    n_parts: int = 10,
    n_operators: int = 3,
    n_trials: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, GageRR]]:
    """Render `n_parts` parts, re-fixture each under `n_operators` conditions x `n_trials`.

    A render that cannot be measured is re-drawn (up to 5 times) so the design stays
    balanced; the number of such retries is reported as a warning.
    """
    rng = np.random.default_rng(seed)
    od, pk = spec.outer_diameter, spec.pocket_diameter
    parts = [
        (rng.uniform(od.lsl, od.usl), rng.uniform(pk.lsl, pk.usl) if pk else None, tuple(rng.normal(0, 0.02, 2)))
        for _ in range(n_parts)
    ]
    rows, retries = [], 0
    for op in range(n_operators):
        op_rng = np.random.default_rng(1000 + op)  # condition-specific lighting/placement bias
        noise = op_rng.uniform(2.0, 4.0)
        gradient = tuple(op_rng.uniform(-25, 25, 2))
        for pi, (od_mm, pk_mm, off) in enumerate(parts):
            for t in range(n_trials):
                for _attempt in range(5):
                    img = render_part(spec, rng, od_mm=od_mm, pocket_mm=pk_mm, pocket_offset_mm=off,
                                      noise_sigma=noise, gradient=gradient)
                    res = inspect_array(img, f"P{pi}", spec)
                    if res.od_mm is not None:
                        break
                    retries += 1
                else:
                    raise RuntimeError(f"part {pi} could not be measured under condition {op}")
                rows.append({"part": pi, "operator": f"COND{op}", "trial": t, "od_mm": res.od_mm,
                             "pocket_mm": res.pocket_mm, "true_od_mm": od_mm, "true_pocket_mm": pk_mm})
    if retries:
        warnings.warn(f"simulate_gage_study: {retries} render(s) re-drawn after a failed measurement", stacklevel=2)
    df = pd.DataFrame(rows)
    studies = {"od_mm": gage_rr_anova(df.rename(columns={"od_mm": "value"}), od.usl - od.lsl, "outer diameter")}
    if pk is not None:
        studies["pocket_mm"] = gage_rr_anova(df.rename(columns={"pocket_mm": "value"}), pk.usl - pk.lsl, "pocket diameter")
    return df, studies
