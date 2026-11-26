"""Separate retrieval, local verification and global claim metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def global_metrics(y_true, y_pred, probabilities=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    result = {"accuracy": float(accuracy_score(y_true, y_pred)),
              "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
              "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    for index, name in enumerate(("negative", "positive")):
        result[f"{name}_precision"], result[f"{name}_recall"], result[f"{name}_f1"] = map(float, (p[index], r[index], f[index]))
    if probabilities is not None and len(set(y_true.tolist())) == 2:
        result["auc"] = float(roc_auc_score(y_true, probabilities))
        result["brier"] = float(np.mean((y_true - probabilities) ** 2))
    else:
        result["auc"], result["brier"] = None, None
    return result


def local_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    result = {"macro_f1": float(f.mean())}
    for index, name in enumerate(("support", "contradict", "irrelevant")):
        result[f"{name}_precision"], result[f"{name}_recall"], result[f"{name}_f1"] = map(float, (p[index], r[index], f[index]))
    return result


def retrieval_metrics(ranked_edge_ids: list[list[int]], gold_edge_ids: list[set[int]]) -> dict:
    if not gold_edge_ids or not any(gold_edge_ids):
        return {"available": False, "note": "candidate coverage not available without local annotations"}
    values = {}
    for k in (1, 2, 4):
        hits = [bool(set(rank[:k]) & gold) for rank, gold in zip(ranked_edge_ids, gold_edge_ids) if gold]
        values[f"recall_at_{k}"] = float(np.mean(hits)) if hits else 0.0
    reciprocal = []
    for rank, gold in zip(ranked_edge_ids, gold_edge_ids):
        positions = [i + 1 for i, edge in enumerate(rank) if edge in gold]
        reciprocal.append(1.0 / min(positions) if positions else 0.0)
    values["mrr"] = float(np.mean(reciprocal)) if reciprocal else 0.0
    values["available"] = True
    return values
