"""Literature-anchored adaptations used by the Alt5 comparison protocol.

The implementations preserve the central mechanisms of KGAT and Graph
Matching Networks while using the common converted graph export.  They are
reported as adaptations, not as byte-for-byte reproductions of the original
papers' FEVER pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hover_exp.experiment import metrics, seed_all


def _triple_features(graph, embeddings):
    """One vector per factual edge/triple, preserving direction."""
    factual = graph["factual"].to(embeddings.device)
    if len(factual) == 0:
        return embeddings.new_zeros((0, embeddings.shape[-1] * 3))
    edge_index = graph["edge_index"].to(embeddings.device)
    node_text = graph["node_text"].to(embeddings.device)
    edge_text = graph["edge_text"].to(embeddings.device)
    src, dst = edge_index[:, factual]
    return torch.cat((embeddings[node_text[src]],
                      embeddings[edge_text[factual]],
                      embeddings[node_text[dst]]), dim=-1)


def _node_features(graph, embeddings):
    """Node vectors plus the mean relation embedding incident to each node."""
    node_text = graph["node_text"].to(embeddings.device)
    nodes = embeddings[node_text]
    if len(nodes) == 0:
        return nodes.new_zeros((0, embeddings.shape[-1] * 2))
    src, dst = graph["edge_index"].to(embeddings.device)
    rel = embeddings[graph["edge_text"].to(embeddings.device)]
    relation_sum = nodes.new_zeros(nodes.shape)
    degree = nodes.new_zeros((len(nodes), 1))
    relation_sum.index_add_(0, src, rel)
    relation_sum.index_add_(0, dst, rel)
    degree.index_add_(0, src, nodes.new_ones((len(src), 1)))
    degree.index_add_(0, dst, nodes.new_ones((len(dst), 1)))
    return torch.cat((nodes, relation_sum / degree.clamp_min(1.0)), dim=-1)


class KGATStyle(nn.Module):
    """Evidence-graph attention with claim-conditioned kernel attention.

    This follows the KGAT idea of node/edge-aware evidence propagation and
    multi-view aggregation, adapted from sentence evidence to evidence triples.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, layers: int = 2, dropout: float = .2):
        super().__init__()
        self.input = nn.Linear(embedding_dim * 3, hidden_dim)
        self.claim = nn.Linear(embedding_dim * 3, hidden_dim)
        self.q = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.k = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.v = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, pair, embeddings):
        evidence = self.input(_triple_features(pair["evidence"], embeddings))
        claim_triples = _triple_features(pair["claim"], embeddings)
        claim = self.claim(claim_triples).mean(0) if len(claim_triples) else evidence.new_zeros(self.claim.out_features)
        if len(evidence):
            for q, k, v, norm in zip(self.q, self.k, self.v, self.norm):
                scores = (q(evidence) @ k(evidence).T) / evidence.shape[-1] ** .5
                weights = scores.softmax(-1)
                evidence = norm(evidence + self.dropout(weights @ v(evidence)))
            query = claim @ evidence.T / evidence.shape[-1] ** .5
            claim_weights = query.softmax(-1)
            attended = claim_weights @ evidence
            mean, maximum = evidence.mean(0), evidence.max(0).values
        else:
            attended = mean = maximum = claim.new_zeros(claim.shape)
        pooled = torch.cat((claim, attended, mean, maximum, (claim - attended).abs()))
        return self.head(pooled).squeeze(-1)


class GMNStyle(nn.Module):
    """Cross-graph matching network adapted to binary verification."""

    def __init__(self, embedding_dim: int, hidden_dim: int, layers: int = 2, dropout: float = .2):
        super().__init__()
        self.project = nn.Linear(embedding_dim * 2, hidden_dim)
        self.updates = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 8, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _encode(self, graph, embeddings):
        return self.project(_node_features(graph, embeddings))

    def forward(self, pair, embeddings):
        claim = self._encode(pair["claim"], embeddings)
        evidence = self._encode(pair["evidence"], embeddings)
        if not len(claim):
            claim = evidence.new_zeros((1, evidence.shape[-1]))
        if not len(evidence):
            evidence = claim.new_zeros((1, claim.shape[-1]))
        for update, norm in zip(self.updates, self.norms):
            similarity = claim @ evidence.T / claim.shape[-1] ** .5
            claim_message = similarity.softmax(-1) @ evidence
            evidence_message = similarity.T.softmax(-1) @ claim
            claim = norm(claim + self.dropout(update(torch.cat((claim, claim_message), -1))))
            evidence = norm(evidence + self.dropout(update(torch.cat((evidence, evidence_message), -1))))
        c_mean, c_max = claim.mean(0), claim.max(0).values
        e_mean, e_max = evidence.mean(0), evidence.max(0).values
        pooled = torch.cat((c_mean, c_max, e_mean, e_max,
                            (c_mean - e_mean).abs(), c_mean * e_mean,
                            (c_max - e_max).abs(), c_max * e_max))
        return self.head(pooled).squeeze(-1)


def _predict(model, pairs, embeddings, device):
    model.eval()
    with torch.no_grad():
        values = [torch.sigmoid(model(pairs[int(i)], embeddings)) for i in range(len(pairs))]
    return torch.stack(values).detach().cpu().numpy() if values else np.empty(0)


def train_literature_baseline(name, bundle, cfg, train, val, test, seed, device, out):
    """Train one literature adaptation and return test probabilities."""
    seed_all(seed)
    embeddings = bundle["embeddings"].to(device)
    pairs = bundle["pairs"]
    hidden = int(cfg.get("literature_hidden_dim", cfg["hidden_dim"]))
    layers = int(cfg.get("literature_layers", cfg["layers"]))
    dropout = float(cfg.get("literature_dropout", cfg["dropout"]))
    embedding_dim = int(embeddings.shape[-1])
    cls = KGATStyle if name == "kgat_style" else GMNStyle
    model = cls(embedding_dim, hidden, layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    labels = np.asarray([p["label"] for p in pairs], dtype=np.float32)
    best_score, best_state, stale = -1.0, None, 0
    epochs = int(cfg.get("literature_epochs", cfg["epochs"]))
    patience = int(cfg.get("literature_patience", cfg["patience"]))
    for epoch in range(epochs):
        model.train()
        order = np.asarray(train)[np.random.permutation(len(train))]
        for start in range(0, len(order), int(cfg["batch_size"])):
            batch = order[start:start + int(cfg["batch_size"])]
            logits = torch.stack([model(pairs[int(i)], embeddings) for i in batch])
            y = torch.as_tensor(labels[batch], device=device)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        probabilities = _predict(model, pairs, embeddings, device)
        score = metrics(labels[val], probabilities[val])["macro_f1"]
        if score > best_score:
            best_score, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"Nenhum checkpoint de validação foi salvo para {name}")
    model.load_state_dict(best_state)
    model_dir = Path(out) / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / f"{name}_{seed}.pt")
    (model_dir / f"{name}_{seed}.json").write_text(json.dumps({
        "variant": name, "seed": int(seed), "validation_macro_f1": float(best_score),
        "protocol": "common converted graph export; literature mechanism adaptation",
    }, indent=2), encoding="utf-8")
    return _predict(model, [pairs[int(i)] for i in test], embeddings, device), float(best_score)
