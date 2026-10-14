"""Acceptance sampling for lot disposition.

* KS Q ISO 2859-1 / ANSI-ASQ Z1.4 single sampling, normal inspection (Table I + II-A).
* Zero-acceptance (c = 0) plans in the spirit of MIL-STD-1916 / Squeglia, sized from a
  required confidence against a limiting quality (exact hypergeometric for finite lots).

Table II-A is encoded by its structure: in every AQL column the accept numbers run
0, (arrow up), 1, 2, 3, 5, 7, 10, 14, 21 down the code letters, starting at the letter
whose sample size gives n x AQL ~ 0.2, and all cells beyond Ac = 21 are up-arrows.
Letters above the Ac = 0 cell are down-arrows. This reproduces the published table;
verify against the official standard before using it for a contractual decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

# lot size upper bound -> code letter, general inspection levels I / II / III (Table I)
_LOT_RANGES = [8, 15, 25, 50, 90, 150, 280, 500, 1200, 3200, 10000, 35000, 150000, 500000, math.inf]
_CODE_LETTERS = {
    "I": list("AABCCDEFGHJKLMN"),
    "II": list("ABCDEFGHJKLMNPQ"),
    "III": list("BCDEFGHJKLMNPQR"),
}
INSPECTION_LEVELS = tuple(_CODE_LETTERS)
_LETTER_ORDER = "ABCDEFGHJKLMNPQR"
SAMPLE_SIZE = dict(zip(_LETTER_ORDER, [2, 3, 5, 8, 13, 20, 32, 50, 80, 125, 200, 315, 500, 800, 1250, 2000]))

# letter of the Ac = 0 cell for each AQL column of Table II-A (single sampling, normal)
_AC0_LETTER = {0.010: "R", 0.015: "Q", 0.025: "P", 0.040: "N", 0.065: "M", 0.10: "L", 0.15: "K", 0.25: "J",
               0.40: "H", 0.65: "G", 1.0: "F", 1.5: "E", 2.5: "D", 4.0: "C", 6.5: "B", 10.0: "A"}
# accept numbers down the column starting at the Ac = 0 letter; None = up-arrow cell
_AC_SEQUENCE = [0, None, 1, 2, 3, 5, 7, 10, 14, 21]
AQLS = tuple(sorted(_AC0_LETTER))


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
    if level not in _CODE_LETTERS:
        raise ValueError(f"inspection level must be one of {INSPECTION_LEVELS}")
    if lot_size < 2:
        raise ValueError("lot size must be >= 2")
    idx = next(i for i, ub in enumerate(_LOT_RANGES) if lot_size <= ub)
    return _CODE_LETTERS[level][idx]


def table_cell(letter: str, aql: float) -> tuple[str, int]:
    """Resolve one Table II-A cell (following arrows) -> (letter actually used, Ac)."""
    if aql not in _AC0_LETTER:
        raise ValueError(f"AQL must be one of {AQLS}")
    i0 = _LETTER_ORDER.index(_AC0_LETTER[aql])
    i = _LETTER_ORDER.index(letter)
    pos = i - i0
    if pos < 0:  # above the Ac = 0 cell: arrow down to it
        return _LETTER_ORDER[i0], 0
    if pos >= len(_AC_SEQUENCE) or pos > len(_LETTER_ORDER) - 1 - i0:  # beyond Ac = 21: arrow up
        last = min(len(_AC_SEQUENCE) - 1, len(_LETTER_ORDER) - 1 - i0)
        return _LETTER_ORDER[i0 + last], _AC_SEQUENCE[last]
    ac = _AC_SEQUENCE[pos]
    if ac is None:  # the up-arrow cell right after Ac = 0
        return _LETTER_ORDER[i0], 0
    return letter, ac


def iso2859_single_normal(lot_size: int, aql: float, level: str = "II") -> SamplingPlan:
    """Single sampling plan, normal inspection, per KS Q ISO 2859-1 Table II-A."""
    letter = code_letter(lot_size, level)
    use, ac = table_cell(letter, aql)
    n = SAMPLE_SIZE[use]
    note = ""
    if use != letter:
        note = f"arrow: letter {letter} -> {use} ({'down' if _LETTER_ORDER.index(use) > _LETTER_ORDER.index(letter) else 'up'})"
    if n >= lot_size:
        n = lot_size
        note = (note + "; " if note else "") + "sample >= lot: 100 % inspection"
    return SamplingPlan("KS Q ISO 2859-1 single/normal", lot_size, letter, n, ac, ac + 1, aql, note)


def zero_acceptance_plan(lot_size: int, limiting_quality: float, confidence: float = 0.90) -> SamplingPlan:
    """c = 0 plan: smallest n such that a lot at `limiting_quality` (fraction defective)
    is rejected with probability >= `confidence`. Exact hypergeometric for finite lots.

    This is the accept-on-zero philosophy of MIL-STD-1916 (the exact 1916 tables are
    indexed by verification level instead of LQ; use them when the contract cites 1916).
    """
    if not 0 < limiting_quality < 1 or not 0 < confidence < 1:
        raise ValueError("limiting_quality and confidence must be in (0, 1)")
    if lot_size < 1:
        raise ValueError("lot size must be >= 1")
    D = max(1, math.ceil(Fraction(str(limiting_quality)) * lot_size))  # exact, no float round-up
    for n in range(1, lot_size + 1):
        # P(0 defective in sample | D defective in lot)
        p0 = math.comb(lot_size - D, n) / math.comb(lot_size, n) if n <= lot_size - D else 0.0
        if 1 - p0 >= confidence:
            return SamplingPlan("accept-on-zero (c=0)", lot_size, None, n, 0, 1,
                                note=f"LQ={limiting_quality:.3%}, confidence={confidence:.0%}, D={D}")
    return SamplingPlan("accept-on-zero (c=0)", lot_size, None, lot_size, 0, 1, note="100 % inspection")


def oc_curve(plan: SamplingPlan, p: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Operating characteristic: P(accept) vs fraction defective.

    Exact hypergeometric for the finite lot the plan was built for; binomial would
    overstate the risk for small lots.
    """
    from scipy import stats

    p = np.linspace(0, 0.2, 201) if p is None else np.asarray(p, dtype=float)
    N, n, c = plan.lot_size, plan.sample_size, plan.accept
    D = np.round(p * N).astype(int)
    pa = stats.hypergeom.cdf(c, N, D, n)
    return p, np.asarray(pa, dtype=float)


def draw_sample(serials: list[str], plan: SamplingPlan, seed: int | None = None) -> list[str]:
    rng = np.random.default_rng(seed)
    n = min(plan.sample_size, len(serials))
    return [serials[i] for i in sorted(rng.choice(len(serials), n, replace=False))]


def lot_disposition(plan: SamplingPlan, n_defective: int, n_critical: int = 0) -> dict:
    """Lot decision = sampling plan + the critical-defect rule used in the inspection standard:
    any critical defect in the sample rejects the lot regardless of Ac (c = 0 for criticals)."""
    by_plan = plan.decide(n_defective)
    decision = "REJECT" if n_critical > 0 else by_plan
    return {"decision": decision, "by_plan": by_plan, "n_defective": n_defective, "n_critical": n_critical,
            "rule": "critical->reject; else Ac/Re", "plan": plan.to_dict()}
