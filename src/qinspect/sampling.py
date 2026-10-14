"""Acceptance sampling for lot disposition.

* KS Q ISO 2859-1 / ANSI-ASQ Z1.4 single sampling, normal inspection (Table I + II-A).
* Zero-acceptance (c = 0) plans in the spirit of MIL-STD-1916 / Squeglia, sized from a
  required confidence against a limiting quality.

The ISO 2859-1 numbers below were transcribed from the public master tables for
general inspection level II. Verify against the official standard before using
this module for a real contractual acceptance decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# lot size upper bound -> code letter, general inspection levels I / II / III
_LOT_RANGES = [8, 15, 25, 50, 90, 150, 280, 500, 1200, 3200, 10000, 35000, 150000, 500000, math.inf]
_CODE_LETTERS = {
    "I": list("AABCCDEFGHJKLMN"),
    "II": list("ABCDEFGHJKLMNPQ"),
    "III": list("BCDEFGHJKLMNPQR"),
}
SAMPLE_SIZE = {"A": 2, "B": 3, "C": 5, "D": 8, "E": 13, "F": 20, "G": 32, "H": 50, "J": 80, "K": 125,
               "L": 200, "M": 315, "N": 500, "P": 800, "Q": 1250, "R": 2000}
_LETTER_ORDER = "ABCDEFGHJKLMNPQR"

# Table II-A (single sampling, normal inspection): AQL -> {code letter: Ac}.
# A missing letter means "follow the arrow": use the first plan below (larger sample).
_AC_TABLE: dict[float, dict[str, int]] = {
    0.065: {"L": 0, "N": 1, "P": 2, "Q": 3, "R": 5},
    0.10: {"K": 0, "M": 1, "N": 2, "P": 3, "Q": 5, "R": 7},
    0.15: {"J": 0, "L": 1, "M": 2, "N": 3, "P": 5, "Q": 7, "R": 10},
    0.25: {"J": 0, "L": 1, "M": 2, "N": 3, "P": 5, "Q": 7, "R": 10},
    0.40: {"H": 0, "K": 1, "L": 2, "M": 3, "N": 5, "P": 7, "Q": 10, "R": 14},
    0.65: {"G": 0, "J": 1, "K": 2, "L": 3, "M": 5, "N": 7, "P": 10, "Q": 14, "R": 21},
    1.0: {"F": 0, "H": 1, "J": 2, "K": 3, "L": 5, "M": 7, "N": 10, "P": 14, "Q": 21, "R": 21},
    1.5: {"E": 0, "G": 1, "H": 2, "J": 3, "K": 5, "L": 7, "M": 10, "N": 14, "P": 21, "Q": 21, "R": 21},
    2.5: {"D": 0, "F": 1, "G": 2, "H": 3, "J": 5, "K": 7, "L": 10, "M": 14, "N": 21, "P": 21, "Q": 21, "R": 21},
    4.0: {"C": 0, "E": 1, "F": 2, "G": 3, "H": 5, "J": 7, "K": 10, "L": 14, "M": 21, "N": 21, "P": 21, "Q": 21, "R": 21},
    6.5: {"B": 0, "D": 1, "E": 2, "F": 3, "G": 5, "H": 7, "J": 10, "K": 14, "L": 21, "M": 21, "N": 21, "P": 21, "Q": 21, "R": 21},
}
# For the 0.25 % column the standard differs from 0.15 % only at letter K (arrow vs. 0); keep as transcribed above.


@dataclass(frozen=True)
class SamplingPlan:
    standard: str
    lot_size: int
    code_letter: str | None
    sample_size: int
    accept: int
    reject: int
    aql: float | None = None
    note: str = ""

    def decide(self, n_defective: int) -> str:
        return "ACCEPT" if n_defective <= self.accept else "REJECT"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def code_letter(lot_size: int, level: str = "II") -> str:
    if lot_size < 2:
        raise ValueError("lot size must be >= 2")
    idx = next(i for i, ub in enumerate(_LOT_RANGES) if lot_size <= ub)
    return _CODE_LETTERS[level][idx]


def iso2859_single_normal(lot_size: int, aql: float, level: str = "II") -> SamplingPlan:
    """Single sampling plan, normal inspection, per KS Q ISO 2859-1 Table II-A."""
    if aql not in _AC_TABLE:
        raise ValueError(f"AQL must be one of {sorted(_AC_TABLE)}")
    letter = code_letter(lot_size, level)
    col = _AC_TABLE[aql]
    i = _LETTER_ORDER.index(letter)
    while _LETTER_ORDER[i] not in col:  # follow the arrow down
        i += 1
    use = _LETTER_ORDER[i]
    n = min(SAMPLE_SIZE[use], lot_size)  # 100 % inspection when the sample exceeds the lot
    ac = col[use]
    note = f"arrow: letter {letter} -> {use}" if use != letter else ""
    if n == lot_size:
        note = (note + "; " if note else "") + "sample >= lot: 100 % inspection"
    return SamplingPlan("KS Q ISO 2859-1 single/normal", lot_size, letter, n, ac, ac + 1, aql, note)


def zero_acceptance_plan(lot_size: int, limiting_quality: float, confidence: float = 0.90) -> SamplingPlan:
    """c = 0 plan: smallest n such that a lot at `limiting_quality` (fraction defective)
    is rejected with probability >= `confidence`. Hypergeometric for finite lots.

    This is the accept-on-zero philosophy of MIL-STD-1916 (the exact 1916 tables are
    indexed by verification level instead of LQ; use them when the contract cites 1916).
    """
    if not 0 < limiting_quality < 1 or not 0 < confidence < 1:
        raise ValueError("limiting_quality and confidence must be in (0, 1)")
    D = max(1, math.ceil(limiting_quality * lot_size))
    for n in range(1, lot_size + 1):
        # P(0 defective in sample | D defective in lot)
        p0 = math.comb(lot_size - D, n) / math.comb(lot_size, n) if n <= lot_size - D else 0.0
        if 1 - p0 >= confidence:
            return SamplingPlan("accept-on-zero (c=0)", lot_size, None, n, 0, 1,
                                note=f"LQ={limiting_quality:.3%}, confidence={confidence:.0%}")
    return SamplingPlan("accept-on-zero (c=0)", lot_size, None, lot_size, 0, 1, note="100 % inspection")


def oc_curve(plan: SamplingPlan, p: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Operating characteristic: P(accept) vs fraction defective (binomial approximation)."""
    from scipy import stats

    p = np.linspace(0, 0.2, 201) if p is None else np.asarray(p)
    pa = stats.binom.cdf(plan.accept, plan.sample_size, p)
    return p, pa


def draw_sample(serials: list[str], plan: SamplingPlan, seed: int | None = None) -> list[str]:
    rng = np.random.default_rng(seed)
    n = min(plan.sample_size, len(serials))
    return [serials[i] for i in sorted(rng.choice(len(serials), n, replace=False))]
