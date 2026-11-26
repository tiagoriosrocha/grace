"""Transparent candidate and condition rules for LOCAL-HEURISTIC."""
from __future__ import annotations

import math


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def candidate_probabilities(candidate: dict, use_exact: bool = True) -> dict[str, float]:
    ss, sr, so = (float(candidate[k]) for k in ("subject_similarity", "relation_similarity", "object_similarity"))
    exact = [float(candidate[k]) if use_exact else 0.0 for k in ("exact_subject", "exact_relation", "exact_object")]
    # Support requires all three endpoints to align; a high object similarity
    # cannot compensate for an incompatible subject.
    support_raw = min(ss, sr, so) + 0.20 * sum(exact) / 3.0
    anchor = min(ss, sr)
    contradict_raw = anchor * (1.0 - max(-1.0, min(1.0, so)))
    if use_exact and exact[0] and exact[1] and not exact[2]:
        contradict_raw += 0.25
    irrelevant_raw = (1.0 - max(ss, sr)) + 0.25 * (1.0 - exact[0])
    logits = [5.0 * (support_raw - 0.45), 5.0 * (contradict_raw - 0.35), 3.0 * (irrelevant_raw - 0.45)]
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return {name: value / total for name, value in zip(("SUPPORT", "CONTRADICT", "IRRELEVANT"), exps)}


def weak_label(candidate: dict, use_exact: bool = True) -> str:
    """Generate explicitly marked weak labels without reading the claim label."""
    probs = candidate_probabilities(candidate, use_exact)
    if probs["CONTRADICT"] >= probs["SUPPORT"] and probs["CONTRADICT"] >= 0.45:
        return "CONTRADICT"
    if probs["SUPPORT"] >= 0.55:
        return "SUPPORT"
    return "IRRELEVANT"
