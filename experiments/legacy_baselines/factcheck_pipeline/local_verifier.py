"""Independent candidate-wise MLP verifier with optional weak supervision."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .heuristic_verifier import weak_label

LOCAL_LABELS = {"SUPPORT": 0, "CONTRADICT": 1, "IRRELEVANT": 2}


class LocalMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 6, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 3)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def make_features(claim_vector: torch.Tensor, evidence_vector: torch.Tensor,
                  signals: list[float] | torch.Tensor) -> torch.Tensor:
    if not isinstance(signals, torch.Tensor):
        signals = torch.tensor(signals, dtype=claim_vector.dtype, device=claim_vector.device)
    return torch.cat([claim_vector, evidence_vector,
                      (claim_vector - evidence_vector).abs(),
                      claim_vector * evidence_vector, signals], dim=-1)


def weak_training_label(candidate: dict[str, Any], use_exact: bool) -> int:
    return LOCAL_LABELS[weak_label(candidate, use_exact)]


def load_annotated_labels(path: str | Path | None) -> dict[tuple[str, str, int], int]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    labels: dict[tuple[str, str, int], int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json
        row = json.loads(line)
        label = str(row["label"]).upper()
        if label not in LOCAL_LABELS:
            raise ValueError(f"Local label inválido: {label}")
        labels[(str(row["claim_id"]), str(row["condition_id"]), int(row["evidence_edge_id"]))] = LOCAL_LABELS[label]
    return labels


def train_local_mlp(features: np.ndarray, labels: np.ndarray, roles: np.ndarray,
                    cfg: dict, device: torch.device, seed: int) -> tuple[LocalMLP | None, dict]:
    """Train only on local supervision and select epochs on local validation labels."""
    if len(features) == 0 or len(set(labels.tolist())) < 2:
        return None, {"status": "fallback", "reason": "insufficient local label diversity"}
    torch.manual_seed(seed)
    model = LocalMLP(int(cfg.get("local_hidden_dim", 96)), float(cfg.get("dropout", 0.1))).to(device)
    train_mask, val_mask = roles == "train", roles == "validation"
    if not train_mask.any() or not val_mask.any():
        return None, {"status": "fallback", "reason": "missing train/validation local pairs"}
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("local_learning_rate", .001)),
                                  weight_decay=float(cfg.get("local_weight_decay", 1e-4)))
    best_state, best_score, stale = None, -1.0, 0
    history = []
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    batch_size = max(1, int(cfg.get("local_batch_size", 256)))
    for epoch in range(1, int(cfg.get("local_epochs", 12)) + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch * 1009)
        order = rng.permutation(train_indices)
        for start in range(0, len(order), batch_size):
            ix = torch.tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[ix]), y[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x[torch.tensor(val_indices, device=device)]).argmax(-1).cpu().numpy()
        score = float(f1_score(labels[val_indices], pred, average="macro", zero_division=0))
        history.append({"epoch": epoch, "validation_macro_f1": score})
        if score > best_score + 1e-8:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(cfg.get("local_patience", 3)):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"status": "trained", "validation_macro_f1": best_score, "history": history,
                   "train_pairs": int(train_mask.sum()), "validation_pairs": int(val_mask.sum()),
                   "label_distribution": Counter(labels[train_mask].tolist())}
