"""Context-only graph encoders; no global verdict head lives here."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGATConv


class ContextualGraphEncoder(nn.Module):
    """The old directed RGAT representation, stopped before classification."""

    def __init__(self, embeddings: torch.Tensor, relation_count: int, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("local_hidden_dim", cfg.get("hidden_dim", 96)))
        self.hidden_dim = hidden
        self.dropout = float(cfg.get("dropout", 0.1))
        self.register_buffer("text_embeddings", embeddings, persistent=False)
        self.node_project = nn.Linear(embeddings.shape[1] + 2, hidden)
        self.rel_category = nn.Embedding(max(1, relation_count), 16)
        self.rel_project = nn.Linear(embeddings.shape[1] + 16, hidden)
        heads = int(cfg.get("heads", 2))
        self.convs = nn.ModuleList([
            RGATConv(hidden, hidden // heads, heads=heads,
                     num_relations=max(1, relation_count),
                     num_bases=min(int(cfg.get("relation_bases", 4)), max(1, relation_count)),
                     edge_dim=hidden, dropout=self.dropout)
            for _ in range(int(cfg.get("layers", 4)))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.convs])
        self.triple_project = nn.Linear(hidden * 3, hidden)

    def encode_graph(self, graph: dict[str, torch.Tensor]):
        nodes = self.node_project(torch.cat([
            self.text_embeddings[graph["node_text"]], graph["structure"]
        ], dim=-1))
        relations = self.rel_project(torch.cat([
            self.text_embeddings[graph["edge_text"]], self.rel_category(graph["edge_type"])
        ], dim=-1))
        edges = graph["edge_index"]
        for conv, norm in zip(self.convs, self.norms):
            if edges.shape[1]:
                message = conv(nodes, edges, edge_type=graph["edge_type"], edge_attr=relations)
            else:
                message = torch.zeros_like(nodes)
            nodes = norm(nodes + F.dropout(F.gelu(message), p=self.dropout, training=self.training))
        return nodes, relations

    def triple_vectors(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        nodes, relations = self.encode_graph(graph)
        src, dst = graph["edge_index"]
        all_triples = self.triple_project(torch.cat([nodes[src], relations, nodes[dst]], dim=-1))
        return all_triples[graph["factual"]]

    def pair_vectors(self, pair: dict[str, dict[str, torch.Tensor]]):
        return self.triple_vectors(pair["claim"]), self.triple_vectors(pair["evidence"])


class RawTripleEncoder(nn.Module):
    """A0: frozen projection of the three BERT vectors without RGAT context."""

    def __init__(self, embeddings: torch.Tensor, cfg: dict):
        super().__init__()
        hidden = int(cfg.get("local_hidden_dim", cfg.get("hidden_dim", 96)))
        self.hidden_dim = hidden
        self.register_buffer("text_embeddings", embeddings, persistent=False)
        self.project = nn.Linear(embeddings.shape[1] * 3, hidden)

    def triple_vectors(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        src, dst = graph["edge_index"]
        values = torch.cat([
            self.text_embeddings[graph["node_text"]][src],
            self.text_embeddings[graph["edge_text"]],
            self.text_embeddings[graph["node_text"]][dst],
        ], dim=-1)
        return self.project(values)[graph["factual"]]

    def pair_vectors(self, pair):
        return self.triple_vectors(pair["claim"]), self.triple_vectors(pair["evidence"])
