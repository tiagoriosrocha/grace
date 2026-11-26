"""Diagnostics for repeated-entity alignment; never changes the verdict."""
from __future__ import annotations


def entity_consistency(conditions, condition_audits: list[dict]) -> dict:
    mappings = {}
    disagreements = []
    for condition, audit in zip(conditions, condition_audits):
        candidate = (audit.get("best_contradict_evidence")
                     if audit.get("decision") == "CONTRADICTED"
                     else audit.get("best_support_evidence"))
        if not candidate:
            continue
        evidence = candidate["evidence_triple"]
        for claim_id, evidence_id, endpoint in (
            (condition.source_id, evidence["source_id"], "source"),
            (condition.target_id, evidence["target_id"], "target"),
        ):
            previous = mappings.get(claim_id)
            if previous is not None and previous != evidence_id:
                disagreements.append({"claim_entity_id": claim_id,
                                      "previous_evidence_entity_id": previous,
                                      "current_evidence_entity_id": evidence_id,
                                      "endpoint": endpoint,
                                      "condition_id": condition.condition_id})
            mappings[claim_id] = evidence_id
    return {"entity_consistency": not disagreements, "mappings": mappings,
            "disagreements": disagreements}
