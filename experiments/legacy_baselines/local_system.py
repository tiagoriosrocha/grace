"""Seventh Alternative-5 system: explicit local verification and aggregation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from factcheck_pipeline.audit import write_html, write_jsonl
from factcheck_pipeline.aggregation import aggregate_conditions, decide_condition
from factcheck_pipeline.candidate_retrieval import candidates_from_bundle
from factcheck_pipeline.conditions import conditions_from_record
from factcheck_pipeline.consistency import entity_consistency
from factcheck_pipeline.graph_encoder import ContextualGraphEncoder
from factcheck_pipeline.heuristic_verifier import candidate_probabilities
from factcheck_pipeline.local_verifier import (LOCAL_LABELS, make_features,
                                                train_local_mlp, weak_training_label)
from factcheck_pipeline.metrics import retrieval_metrics
from hover_exp.experiment import seed_all


def _signals(candidate: dict) -> torch.Tensor:
    return torch.tensor([
        candidate["subject_similarity"], candidate["relation_similarity"], candidate["object_similarity"],
        float(candidate["exact_subject"]), float(candidate["exact_relation"]), float(candidate["exact_object"]),
    ], dtype=torch.float32)


def _load_legacy_context(encoder, out: Path, seed: int) -> str:
    """Reuse only the selected RGAT representation; discard its global head."""
    checkpoint = out / "models" / "relational_gnn" / f"relational_gnn_{seed}.pt"
    if not checkpoint.exists():
        return "fresh_rgat_initialization"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adapted = {}
    for key, value in state.items():
        if key.startswith("triple."):
            adapted["triple_project." + key[len("triple."):]] = value
        elif (key.startswith("node_project.") or key.startswith("rel_category.") or
              key.startswith("rel_project.") or key.startswith("convs.") or
              key.startswith("norms.")):
            adapted[key] = value
    encoder.load_state_dict(adapted, strict=False)
    return "relational_gnn_encoder_selected_on_validation"


def _candidate_probs(candidate: dict, row_probs: dict | None, key: tuple, use_exact=True):
    if row_probs is not None and key in row_probs:
        values = row_probs[key]
        return {"SUPPORT": float(values[0]), "CONTRADICT": float(values[1]), "IRRELEVANT": float(values[2])}
    return candidate_probabilities(candidate, use_exact)


def run_local_system(records: list[dict], bundle: dict, cfg: dict, train, validation, test,
                     seed: int, device: torch.device, out: Path) -> tuple[np.ndarray, dict]:
    """Train/evaluate the seventh system and return test support scores."""
    seed_all(seed)
    text_index = json.loads((out / "text_index.json").read_text(encoding="utf-8"))
    pair_by_id = {pair["id"]: pair for pair in bundle["pairs"]}
    encoder = ContextualGraphEncoder(bundle["embeddings"], len(bundle["relation_vocab"]), cfg)
    context_source = _load_legacy_context(encoder, out, seed)
    encoder = encoder.to(device).eval()
    local_cfg = dict(cfg, local_hidden_dim=cfg.get("hidden_dim", 128),
                     local_epochs=cfg.get("local_epochs", 12), local_patience=cfg.get("local_patience", 3),
                     local_batch_size=cfg.get("local_batch_size", 256),
                     local_learning_rate=cfg.get("local_learning_rate", .001),
                     local_weight_decay=cfg.get("local_weight_decay", 1e-4))

    feature_rows = []
    candidate_cache = {}
    for record in sorted(records, key=lambda item: str(item["id"])):
        pair = pair_by_id[record["id"]]
        conditions = conditions_from_record(record)
        candidates_by_condition = candidates_from_bundle(
            record, pair, conditions, text_index, bundle["embeddings"], int(cfg.get("top_k", 4)))
        candidate_cache[record["id"]] = (conditions, candidates_by_condition)
        with torch.no_grad():
            claim_vectors, evidence_vectors = encoder.pair_vectors({
                "claim": {key: value.to(device) if isinstance(value, torch.Tensor) else value
                          for key, value in pair["claim"].items()},
                "evidence": {key: value.to(device) if isinstance(value, torch.Tensor) else value
                             for key, value in pair["evidence"].items()},
            })
        evidence_positions = {int(edge): pos for pos, edge in enumerate(pair["evidence"]["factual"].tolist())}
        for condition_index, (condition, candidates) in enumerate(zip(conditions, candidates_by_condition)):
            for candidate in candidates:
                position = evidence_positions[candidate.evidence_edge_index]
                feature = make_features(claim_vectors[condition_index], evidence_vectors[position],
                                        _signals(candidate.to_dict()).to(device))
                candidate_dict = candidate.to_dict()
                feature_rows.append({
                    "id": record["id"], "role": record["evaluation_role"],
                    "condition_id": condition.condition_id,
                    "evidence_edge_index": candidate.evidence_edge_index,
                    "feature": feature.cpu().numpy().astype("float32"),
                    "weak_label": weak_training_label(candidate_dict, True),
                })

    x = np.stack([row["feature"] for row in feature_rows])
    # weak_training_label already returns the encoded integer label.
    y = np.asarray([row["weak_label"] for row in feature_rows], dtype=np.int64)
    roles = np.asarray([row["role"] for row in feature_rows], dtype=object)
    model, training_info = train_local_mlp(x, y, roles, local_cfg, device, seed)
    row_probs = None
    if model is not None:
        row_probs = {}
        model.eval()
        with torch.no_grad():
            values = model(torch.tensor(x, dtype=torch.float32, device=device)).softmax(-1).cpu().numpy()
        for row, value in zip(feature_rows, values):
            row_probs[(row["id"], row["condition_id"], row["evidence_edge_index"])] = value
    else:
        training_info["fallback"] = "deterministic heuristic verifier"

    audits = []
    scores_by_id = {}
    for record in sorted(records, key=lambda item: str(item["id"])):
        conditions, candidates_by_condition = candidate_cache[record["id"]]
        condition_audits, decisions, margins = [], [], []
        for condition, candidates in zip(conditions, candidates_by_condition):
            candidate_audits = []
            for candidate in candidates:
                key = (record["id"], condition.condition_id, candidate.evidence_edge_index)
                probs = _candidate_probs(candidate.to_dict(), row_probs, key)
                value = candidate.to_dict()
                value.update(p_support=probs["SUPPORT"], p_contradict=probs["CONTRADICT"],
                             p_irrelevant=probs["IRRELEVANT"], decision=max(probs, key=probs.get))
                candidate_audits.append(value)
            decision = decide_condition(candidate_audits, cfg.get("support_threshold", .5),
                                        cfg.get("contradict_threshold", .5))
            condition_audits.append(dict(condition_id=condition.condition_id,
                                         claim_triple=condition.to_dict(), candidates=candidate_audits,
                                         **decision))
            decisions.append(decision["decision"])
            margins.append(decision["support_score"] - decision["contradict_score"])
        verdict = aggregate_conditions(decisions, cfg.get("aggregation_mode", "binary"))
        # This is an aggregation score, not a calibrated factual probability.
        score = float(1.0 / (1.0 + np.exp(-5.0 * (min(margins) if margins else -1.0))))
        scores_by_id[record["id"]] = score
        audits.append({"id": record["id"], "claim": record["claim"], "gold": record["label_name"],
                       "prediction": verdict, "conditions": condition_audits,
                       "entity_consistency": entity_consistency(conditions, condition_audits),
                       "method": "local_factcheck_pipeline", "seed": int(seed),
                       "weak_supervision": model is not None})

    audit_path = out / f"local_decisions_{seed}.jsonl"
    write_jsonl(audit_path, audits)
    if seed == int(cfg["seeds"][0]):
        write_html(out / "AUDIT_LOCAL.html",
                   [row for row in audits if row["id"] in {records[int(index)]["id"] for index in test}])
        required = [row for row in audits if (
            "discoverer of (66063)" in row["claim"].casefold() or
            "buzz aldrin was an american" in row["claim"].casefold())]
        (out / "AUDIT_REQUIRED_EXAMPLES.json").write_text(
            json.dumps({"available": bool(required), "claims": required},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"local_training_{seed}.json").write_text(
        json.dumps({"context_source": context_source, "training": training_info,
                    "retrieval": "candidate coverage not available without local annotations"},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    save = out / f"retrieval_metrics_local_{seed}.json"
    save.write_text(json.dumps(retrieval_metrics([], []), indent=2), encoding="utf-8")
    return np.asarray([scores_by_id[records[int(index)]["id"]] for index in test], dtype=float), {
        "audit": audits, "training": training_info, "context_source": context_source,
    }
