"""Explicit extraction of factual claim conditions.

The first protocol treats every non-EntityType factual triple as one
condition.  Keeping this as a separate data structure makes a later logical
representation possible without changing retrieval or aggregation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from hover_exp.source_contract import normalize_node_type, triples


@dataclass(frozen=True)
class Condition:
    condition_id: str
    source_id: str
    source_text: str
    relation: str
    relation_key: str
    target_id: str
    target_text: str
    source_type: str
    target_type: str
    source_edge_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_conditions(claim_graph: Any) -> list[Condition]:
    """Return factual conditions in stable source-edge order.

    ``claim_graph`` can be a raw graph object/JSON string or an already
    sanitized list of triples.  EntityType edges remain available as graph
    context elsewhere, but are intentionally excluded from obligations.
    """
    parsed = claim_graph if isinstance(claim_graph, list) else triples(claim_graph)
    result: list[Condition] = []
    for edge_index, triple in enumerate(parsed):
        if (normalize_node_type(triple.get("source_type", "")) == "entitytype" or
                normalize_node_type(triple.get("target_type", "")) == "entitytype"):
            continue
        result.append(Condition(
            condition_id=f"condition_{len(result):04d}",
            source_id=str(triple["source_id"]),
            source_text=str(triple["source_text"]),
            relation=str(triple["relation"]),
            relation_key=str(triple.get("relation_key", triple["relation"]).casefold()),
            target_id=str(triple["target_id"]),
            target_text=str(triple["target_text"]),
            source_type=str(triple.get("source_type", "")),
            target_type=str(triple.get("target_type", "")),
            source_edge_index=edge_index,
        ))
    return result


def conditions_from_record(record: dict[str, Any]) -> list[Condition]:
    return extract_conditions(record["claim_triples"])
