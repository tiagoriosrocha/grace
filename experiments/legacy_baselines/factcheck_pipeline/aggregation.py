"""Explicit condition and claim aggregation rules."""
from __future__ import annotations


def decide_condition(candidates: list[dict], support_threshold: float = .5,
                     contradict_threshold: float = .5) -> dict:
    best_support = max(candidates, key=lambda x: x["p_support"], default=None)
    best_contradict = max(candidates, key=lambda x: x["p_contradict"], default=None)
    support_score = float(best_support["p_support"]) if best_support else 0.0
    contradict_score = float(best_contradict["p_contradict"]) if best_contradict else 0.0
    if contradict_score >= contradict_threshold:
        decision = "CONTRADICTED"
    elif support_score >= support_threshold:
        decision = "SUPPORTED"
    else:
        decision = "INSUFFICIENT"
    return {"decision": decision, "support_score": support_score,
            "contradict_score": contradict_score,
            "best_support_evidence": best_support,
            "best_contradict_evidence": best_contradict}


def aggregate_conditions(decisions: list[str], mode: str = "binary") -> str:
    if any(value == "CONTRADICTED" for value in decisions):
        return "REFUTES"
    if decisions and all(value == "SUPPORTED" for value in decisions):
        return "SUPPORTS"
    if mode == "three_way":
        return "NOT_ENOUGH_INFORMATION"
    return "REFUTES"
