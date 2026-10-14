"""Statistical process control: X-bar/R charts, Western Electric rules, Cp/Cpk."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Shewhart constants for subgroup size n = 2..10
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
_D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}


@dataclass
class XbarRChart:
    n: int
    xbar: np.ndarray
    ranges: np.ndarray
    center: float
    ucl: float
    lcl: float
    r_bar: float
    r_ucl: float
    r_lcl: float
    sigma_within: float
    violations: dict[int, list[str]] = field(default_factory=dict)  # subgroup index -> rules

    @property
    def out_of_control(self) -> bool:
        return bool(self.violations)


def subgroups(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    k = len(values) // n
    return values[: k * n].reshape(k, n)


def western_electric(points: np.ndarray, center: float, sigma: float) -> dict[int, list[str]]:
    """Rules 1-4 on a sequence of plotted points; returns index -> violated rule names."""
    z = (np.asarray(points, dtype=float) - center) / sigma
    out: dict[int, list[str]] = {}

    def flag(i: int, rule: str) -> None:
        out.setdefault(i, [])
        if rule not in out[i]:
            out[i].append(rule)

    for i in range(len(z)):
        if abs(z[i]) > 3:
            flag(i, "R1:beyond_3sigma")
        for side in (1, -1):
            w = z[max(0, i - 2) : i + 1] * side
            if len(w) == 3 and (w > 2).sum() >= 2 and z[i] * side > 2:
                flag(i, "R2:2of3_beyond_2sigma")
            w = z[max(0, i - 4) : i + 1] * side
            if len(w) == 5 and (w > 1).sum() >= 4 and z[i] * side > 1:
                flag(i, "R3:4of5_beyond_1sigma")
            w = z[max(0, i - 7) : i + 1] * side
            if len(w) == 8 and (w > 0).all():
                flag(i, "R4:8_same_side")
    return out


def xbar_r(values: np.ndarray, n: int = 5, baseline: int | None = None) -> XbarRChart:
    """Build an X-bar/R chart.

    `baseline` = number of initial subgroups used to set the limits (Phase I).
    The remaining subgroups are monitored against them (Phase II). None uses all.
    """
    if n not in _A2:
        raise ValueError("subgroup size must be 2..10")
    g = subgroups(values, n)
    if len(g) < 2:
        raise ValueError("need at least 2 subgroups")
    xbar = g.mean(axis=1)
    ranges = g.max(axis=1) - g.min(axis=1)
    b = len(g) if baseline is None else max(2, min(baseline, len(g)))
    center = float(xbar[:b].mean())
    r_bar = float(ranges[:b].mean())
    ucl, lcl = center + _A2[n] * r_bar, center - _A2[n] * r_bar
    sigma_xbar = _A2[n] * r_bar / 3
    chart = XbarRChart(
        n=n,
        xbar=xbar,
        ranges=ranges,
        center=center,
        ucl=ucl,
        lcl=lcl,
        r_bar=r_bar,
        r_ucl=_D4[n] * r_bar,
        r_lcl=_D3[n] * r_bar,
        sigma_within=r_bar / _D2[n],
    )
    if sigma_xbar > 0:
        chart.violations = western_electric(xbar, center, sigma_xbar)
    for i, r in enumerate(ranges):
        if r > chart.r_ucl or r < chart.r_lcl:
            chart.violations.setdefault(i, []).append("R-chart:out_of_limits")
    return chart


def capability(values: np.ndarray, lsl: float, usl: float, sigma_within: float | None = None) -> dict:
    """Cp/Cpk (within-subgroup sigma, if given) and Pp/Ppk (overall sigma)."""
    x = np.asarray(values, dtype=float)
    mu, s = float(x.mean()), float(x.std(ddof=1))
    res = {"mean": mu, "std_overall": s, "pp": (usl - lsl) / (6 * s), "ppk": min(usl - mu, mu - lsl) / (3 * s)}
    if sigma_within:
        res["std_within"] = sigma_within
        res["cp"] = (usl - lsl) / (6 * sigma_within)
        res["cpk"] = min(usl - mu, mu - lsl) / (3 * sigma_within)
    return res
