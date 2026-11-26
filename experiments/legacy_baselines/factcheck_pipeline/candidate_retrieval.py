"""Deterministic top-k retrieval over factual evidence triples."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


def _norm(value: Any) -> str:
    return " ".join(str(value).casefold().replace("_", " ").split())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    return float(torch.dot(a, b).item() / denom) if denom else 0.0


@dataclass(frozen=True)
class Candidate:
    candidate_rank: int
    evidence_edge_index: int
    evidence_triple: dict[str, Any]
    subject_similarity: float
    relation_similarity: float
    object_similarity: float
    mean_similarity: float
    exact_subject: bool
    exact_relation: bool
    exact_object: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def retrieve_candidates(condition, evidence_triples: list[dict[str, Any]],
                        embeddings: Mapping[str, torch.Tensor], top_k: int = 4) -> list[Candidate]:
    """Retrieve candidates using only frozen text embeddings and triple text.

    Ties are resolved by the original evidence edge index.  Labels, splits and
    claim-level predictions are never read here.
    """
    source = embeddings.get(condition.source_text)
    relation = embeddings.get(condition.relation)
    target = embeddings.get(condition.target_text)
    if source is None or relation is None or target is None:
        raise KeyError("Missing embedding for condition text")
    scored = []
    for index, evidence in enumerate(evidence_triples):
        es = embeddings.get(evidence["source_text"])
        er = embeddings.get(evidence["relation"])
        eo = embeddings.get(evidence["target_text"])
        if es is None or er is None or eo is None:
            raise KeyError("Missing embedding for evidence text")
        ss, sr, so = _cosine(source, es), _cosine(relation, er), _cosine(target, eo)
        scored.append((-(ss + sr + so) / 3.0, index, ss, sr, so, evidence))
    scored.sort(key=lambda row: (row[0], row[1]))
    result = []
    for rank, (_, index, ss, sr, so, evidence) in enumerate(scored[:max(0, int(top_k))], 1):
        result.append(Candidate(
            candidate_rank=rank, evidence_edge_index=index,
            evidence_triple=dict(evidence), subject_similarity=ss,
            relation_similarity=sr, object_similarity=so,
            mean_similarity=(ss + sr + so) / 3.0,
            exact_subject=_norm(condition.source_text) == _norm(evidence["source_text"]),
            exact_relation=_norm(condition.relation) == _norm(evidence["relation"]),
            exact_object=_norm(condition.target_text) == _norm(evidence["target_text"]),
        ))
    return result


def candidates_from_bundle(record: dict[str, Any], pair: dict[str, Any],
                           conditions, text_index: list[str], embeddings: torch.Tensor,
                           top_k: int) -> list[list[Candidate]]:
    """Materialize auditable candidates from the shared baseline feature cache.

    The cache is only used when it was built with the same deterministic
    retrieval contract.  Recomputing is cheap and makes this function safe for
    direct runs without trusting opaque cached candidate objects.
    """
    table = {text: embeddings[i].cpu() for i, text in enumerate(text_index)}
    factual_indices = [int(index) for index in pair["evidence"]["factual"].tolist()]
    factual_triples = [record["evidence_triples"][index] for index in factual_indices]
    result = []
    for condition in conditions:
        candidates = retrieve_candidates(condition, factual_triples, table, top_k)
        # Keep the original graph-edge index in the audit trail rather than a
        # compressed factual-only position.
        result.append([Candidate(
            candidate_rank=candidate.candidate_rank,
            evidence_edge_index=factual_indices[candidate.evidence_edge_index],
            evidence_triple=candidate.evidence_triple,
            subject_similarity=candidate.subject_similarity,
            relation_similarity=candidate.relation_similarity,
            object_similarity=candidate.object_similarity,
            mean_similarity=candidate.mean_similarity,
            exact_subject=candidate.exact_subject,
            exact_relation=candidate.exact_relation,
            exact_object=candidate.exact_object,
        ) for candidate in candidates])
    return result
