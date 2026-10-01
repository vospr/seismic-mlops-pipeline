"""Promotion gate: beat the majority-class baseline AND the current champion."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class GateResult:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def score(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def majority_baseline(y_train, y_test) -> dict:
    """Always predict the most frequent *training* class."""
    vals, counts = np.unique(y_train, return_counts=True)
    return score(y_test, np.full(len(y_test), vals[counts.argmax()]))


def decide(challenger: dict, baseline: dict, champion: dict | None) -> GateResult:
    reasons = []
    for k in ("accuracy", "macro_f1"):
        if challenger[k] <= baseline[k]:
            reasons.append(f"{k} {challenger[k]:.3f} does not beat majority baseline {baseline[k]:.3f}")
    if champion is not None and challenger["macro_f1"] <= champion["macro_f1"]:
        reasons.append(f"macro_f1 {challenger['macro_f1']:.3f} does not beat champion {champion['macro_f1']:.3f}")
    return GateResult(promote=not reasons, reasons=reasons or ["beats baseline" + (" and champion" if champion else "")])
