"""Statistical process control: X-bar/R charts, Western Electric rules, Cp/Cpk, EWMA.

Phase I (`baseline` subgroups) estimates the limits; Phase II monitors against them.
`ControlLimits` lets a quality engineer freeze limits (persisted in the DB) so a
station restart or a new lot does not silently re-baseline on a drifted process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Shewhart constants for subgroup size n = 2..10
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
_D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
SUBGROUP_SIZES = tuple(_A2)
RECOMMENDED_BASELINE = 25  # AIAG SPC manual: >= 25 subgroups for Phase I


@dataclass(frozen=True)
class ControlLimits:
    """Frozen X-bar/R limits for one characteristic."""

    n: int
    center: float
    ucl: float
    lcl: float
    r_bar: float
    r_ucl: float
    r_lcl: float

    @property
    def sigma_xbar(self) -> float:
        return (self.ucl - self.center) / 3

    @property
    def sigma_within(self) -> float:
        return self.r_bar / _D2[self.n]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ControlLimits":
        return cls(**{k: d[k] for k in ("n", "center", "ucl", "lcl", "r_bar", "r_ucl", "r_lcl")})


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
    baseline: int | None = None

    @property
    def out_of_control(self) -> bool:
        return bool(self.violations)

    @property
    def limits(self) -> ControlLimits:
        return ControlLimits(self.n, self.center, self.ucl, self.lcl, self.r_bar, self.r_ucl, self.r_lcl)

    def first_violation_after_baseline(self) -> int | None:
        """1-based subgroup number of the first X-bar rule violation in Phase II (None if none)."""
        b = self.baseline or 0
        cands = [i for i, rules in self.violations.items() if i >= b and any(r.startswith("R") and not r.startswith("R-chart") for r in rules)]
        return min(cands) + 1 if cands else None


def subgroups(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    k = len(values) // n
    return values[: k * n].reshape(k, n)


def western_electric(points: np.ndarray, center: float, sigma: float) -> dict[int, list[str]]:
    """Rules 1-4 on a sequence of plotted points; returns index -> violated rule names.

    A pattern is flagged at the point that completes it, also at the start of the series
    (e.g. the first two points both beyond 2 sigma trigger rule 2 at index 1).
    """
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
            if (w > 2).sum() >= 2 and w[-1] > 2:
                flag(i, "R2:2of3_beyond_2sigma")
            w = z[max(0, i - 4) : i + 1] * side
            if (w > 1).sum() >= 4 and w[-1] > 1:
                flag(i, "R3:4of5_beyond_1sigma")
            w = z[max(0, i - 7) : i + 1] * side
            if len(w) == 8 and (w > 0).all():
                flag(i, "R4:8_same_side")
    return out


def evaluate(values: np.ndarray, limits: ControlLimits) -> XbarRChart:
    """Phase II only: plot `values` against frozen `limits`."""
    g = subgroups(values, limits.n)
    if len(g) < 1:
        raise ValueError("need at least one full subgroup")
    xbar = g.mean(axis=1)
    ranges = g.max(axis=1) - g.min(axis=1)
    chart = XbarRChart(limits.n, xbar, ranges, limits.center, limits.ucl, limits.lcl, limits.r_bar, limits.r_ucl,
                       limits.r_lcl, limits.sigma_within, baseline=0)
    if limits.sigma_xbar > 0:
        chart.violations = western_electric(xbar, limits.center, limits.sigma_xbar)
    for i, r in enumerate(ranges):
        if r > limits.r_ucl or r < limits.r_lcl:
            chart.violations.setdefault(i, []).append("R-chart:out_of_limits")
    return chart


def xbar_r(values: np.ndarray, n: int = 5, baseline: int | None = None) -> XbarRChart:
    """Build an X-bar/R chart.

    `baseline` = number of initial subgroups used to set the limits (Phase I).
    The remaining subgroups are monitored against them (Phase II). None uses all.
    """
    if n not in _A2:
        raise ValueError(f"subgroup size must be one of {SUBGROUP_SIZES}")
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
        baseline=b if baseline is not None else None,
    )
    if sigma_xbar > 0:
        chart.violations = western_electric(xbar, center, sigma_xbar)
    for i, r in enumerate(ranges):
        if r > chart.r_ucl or r < chart.r_lcl:
            chart.violations.setdefault(i, []).append("R-chart:out_of_limits")
    return chart


def ewma(values: np.ndarray, center: float, sigma: float, lam: float = 0.2, L: float = 2.7) -> dict:
    """EWMA chart for small sustained shifts (complements Shewhart rules).

    Returns the EWMA series, time-varying limits and the first out-of-limit index.
    """
    x = np.asarray(values, dtype=float)
    z = np.empty(len(x))
    prev = center
    for i, v in enumerate(x):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    t = np.arange(1, len(x) + 1)
    half = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    ucl, lcl = center + half, center - half
    bad = np.nonzero((z > ucl) | (z < lcl))[0]
    return {"ewma": z, "ucl": ucl, "lcl": lcl, "first_signal": int(bad[0]) if len(bad) else None}


def capability(values: np.ndarray, lsl: float, usl: float, sigma_within: float | None = None) -> dict:
    """Cp/Cpk (within-subgroup sigma, if given) and Pp/Ppk (overall sigma).

    Indices are None when they are undefined (fewer than 2 values or zero spread).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if usl <= lsl:
        raise ValueError("usl must exceed lsl")
    res: dict = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "std_overall": None,
                 "pp": None, "ppk": None}
    if len(x) < 2:
        return res
    mu, s = float(x.mean()), float(x.std(ddof=1))
    res["std_overall"] = s
    if s > 0:
        res["pp"] = (usl - lsl) / (6 * s)
        res["ppk"] = min(usl - mu, mu - lsl) / (3 * s)
    if sigma_within is not None:
        res["std_within"] = sigma_within
        if sigma_within > 0:
            res["cp"] = (usl - lsl) / (6 * sigma_within)
            res["cpk"] = min(usl - mu, mu - lsl) / (3 * sigma_within)
        else:
            res["cp"] = res["cpk"] = None
    return res
