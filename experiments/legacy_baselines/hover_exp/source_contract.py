"""Funções de contrato/divisão copiadas do artigo-1; ver PROVENANCE.md."""

import json, hashlib, re

import numpy as np

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

EXCLUDED_NODE_TYPES = frozenset({"textsummary", "textdocument", "documentchunk"})

def decode(value):
    return json.loads(value) if isinstance(value, str) else value

def normalize_node_type(value):
    return re.sub(r"[\s_-]+", "", str(value).strip().casefold())

def normalize_relation(value):
    """Chave determinística do vocabulário, sem alterar o texto da relação."""
    return re.sub(r"\s+", " ", str(value).strip().casefold())

def triples(value, sanitization=None, side=None):
    g = decode(value)
    if not isinstance(g, dict):
        raise ValueError("grafo deve ser objeto com nodes e edges")
    raw_nodes = g.get("nodes", [])
    raw_edges = g.get("edges", [])
    nodes = {}
    all_node_ids = set()
    for n in raw_nodes:
        key = str(n.get("id", n.get("node_id", "")))
        if not key or key in all_node_ids:
            raise ValueError("ID de nó ausente/duplicado")
        all_node_ids.add(key)
        raw_type = n.get("type", n.get("node_type", ""))
        node_type = normalize_node_type(raw_type)
        if node_type in EXCLUDED_NODE_TYPES:
            continue
        name = n.get("text", n.get("name", n.get("label", key)))
        nodes[key] = {"text": str(name), "type": str(raw_type).strip()}
    result = []
    edges_removed = 0
    for e in raw_edges:
        s, o = str(e["source"]), str(e["target"])
        if s not in all_node_ids or o not in all_node_ids:
            raise ValueError("aresta referencia nó inexistente")
        if s not in nodes or o not in nodes:
            edges_removed += 1
            continue
        rel = e.get("type", e.get("relation", e.get("edge_type")))
        if rel is None:
            raise ValueError("relação sem tipo")
        result.append({
            "source_id": s,
            "source_text": nodes[s]["text"],
            "source_type": nodes[s]["type"],
            "relation": str(rel).strip(),
            "relation_key": normalize_relation(rel),
            "target_id": o,
            "target_text": nodes[o]["text"],
            "target_type": nodes[o]["type"],
        })
    if sanitization is not None:
        if side not in ("claim", "evidence"):
            raise ValueError("lado inválido para estatística de sanitização")
        stats = sanitization[side]
        stats["graphs"] += 1
        stats["nodes_before"] += len(raw_nodes)
        stats["nodes_after"] += len(nodes)
        stats["nodes_removed"] += len(raw_nodes) - len(nodes)
        stats["edges_before"] += len(raw_edges)
        stats["edges_after"] += len(result)
        stats["edges_removed"] += edges_removed
        stats["empty_after"] += int(not result)
    return result

def group_claims(df):
    """Return deterministic leakage groups from optional source groups and claims.

    The converter is deliberately allowed to omit ``group_id``.  A normalized
    claim is therefore always a grouping key; a source group, when present,
    is added as a second key so that transitive relationships are preserved.
    """
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    component_keys = {}
    for i, row in df.reset_index(drop=True).iterrows():
        claim = " ".join(str(row.get("claim", "")).casefold().split())
        source_group = row.get("source_group_id", row.get("group_id", ""))
        if pd.isna(source_group):
            source_group = ""
        keys = [("claim", claim)]
        if str(source_group).strip():
            keys.append(("group", str(source_group).strip()))
        component_keys[i] = keys
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = {}
    for i in range(len(df)):
        members.setdefault(find(i), []).extend(component_keys[i])
    stable = {}
    for root, keys in members.items():
        payload = "|".join(sorted(f"{kind}:{value}" for kind, value in keys))
        stable[root] = "claim_group_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return [stable[find(i)] for i in range(len(df))]

def assign_roles(df, cfg):
    df = df.copy()
    df["group_id"] = group_claims(df)
    override = cfg.get("evaluation_role_override", {})
    if override:
        df["evaluation_role"] = df.apply(lambda r: override.get(str(r.source_split), r.evaluation_role), axis=1)
    explicit = df["evaluation_role"].fillna("").astype(str).str.strip()
    if (explicit != "").any() and (explicit == "").any():
        raise ValueError("evaluation_role deve estar preenchido em todas as linhas ou em nenhuma")
    if not (explicit != "").any():
        mapping = cfg.get("source_split_roles")
        if mapping:
            df["evaluation_role"] = df.source_split.map(mapping)
            if df.evaluation_role.isna().any():
                raise ValueError("source_split_roles não cobre todos os splits de origem")
        else:
            # Raw converted exports intentionally contain no experimental role.
            # Treat the complete export as a pool and create reproducible
            # train/validation/test groups below.
            df["evaluation_role"] = "pool"
    if not set(df.evaluation_role) <= {"train", "validation", "test", "pool"}:
        raise ValueError("Papéis aceitos: train, validation, test, pool")
    if df.groupby("group_id").evaluation_role.nunique().max() > 1:
        raise ValueError("Vazamento: mesma claim/grupo em papéis experimentais distintos")
    pool = df.index[df.evaluation_role == "pool"].to_numpy()
    if len(pool):
        if "train" in set(df.evaluation_role) or "validation" in set(df.evaluation_role):
            raise ValueError("pool não pode coexistir com train/validation explícitos")
        def split(indices, fraction, seed):
            a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(indices, groups=df.loc[indices, "group_id"]))
            return indices[a], indices[b]
        if "test" not in set(df.evaluation_role):
            pool, test = split(pool, cfg["test_fraction"], cfg["split_seed"])
            df.loc[test, "evaluation_role"] = "test"
        train, val = split(pool, cfg["validation_fraction"], cfg["split_seed"] + 1)
        df.loc[train, "evaluation_role"] = "train"
        df.loc[val, "evaluation_role"] = "validation"
    classes = set(df.label)
    for role in ("train", "validation", "test"):
        if set(df.loc[df.evaluation_role == role, "label"]) != classes:
            raise ValueError(f"{role} vazio ou sem todas as classes; defina divisão válida antes de treinar")
    if classes != {0, 1}:
        raise ValueError('O experimento requer as duas classes binárias.')
    # Fixed seed, no search guided by model scores or test performance.
    targets = {'test': cfg['test_fraction'],
               'validation': (1-cfg['test_fraction'])*cfg['validation_fraction']}
    targets['train'] = 1-sum(targets.values())
    for role, target in targets.items():
        subset = df[df.evaluation_role == role]
        if abs(len(subset)/len(df)-target) > cfg.get('split_max_size_deviation', .05):
            raise ValueError(f'Split inviável: {role} tem {len(subset)}/{len(df)} exemplos; alvo {target:.1%}. Revise os grupos.')
        if abs(subset.label.mean()-df.label.mean()) > cfg.get('split_max_class_deviation', .10):
            raise ValueError(f'Desequilíbrio excessivo entre conjuntos em {role}; revise os grupos antes de treinar.')
    return df
