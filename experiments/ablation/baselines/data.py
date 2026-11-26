from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

EXCLUDED = {"textsummary", "textdocument", "documentchunk"}


def norm(x):
    return re.sub(r"\s+", " ", str(x).strip().casefold())


def decode(x):
    return json.loads(x) if isinstance(x, str) else x


def graph_triples(value):
    g = decode(value)
    nodes = {}
    all_ids = set()
    for node in g.get("nodes", []):
        key = str(node.get("id", node.get("node_id", "")))
        if not key or key in all_ids:
            raise ValueError("ID de nó ausente ou duplicado")
        all_ids.add(key)
        typ = str(node.get("type", node.get("node_type", ""))).strip()
        if norm(typ).replace("_", "") in {x.replace("_", "") for x in EXCLUDED}:
            continue
        nodes[key] = (str(node.get("text", node.get("name", node.get("label", key)))), typ)
    triples, removed = [], 0
    for edge in g.get("edges", []):
        s, o = str(edge["source"]), str(edge["target"])
        if s not in all_ids or o not in all_ids:
            raise ValueError("aresta referencia nó inexistente")
        if s not in nodes or o not in nodes:
            removed += 1
            continue
        rel = str(edge.get("type", edge.get("relation", edge.get("edge_type", "")))).strip()
        triples.append({"source": nodes[s][0], "source_type": nodes[s][1], "relation": rel,
                        "target": nodes[o][0], "target_type": nodes[o][1]})
    return triples, {"nodes_before": len(g.get("nodes", [])), "nodes_after": len(nodes),
                     "edges_before": len(g.get("edges", [])), "edges_after": len(triples),
                     "edges_removed": removed, "empty_after": int(not triples)}


def triple_key(t):
    return (norm(t["source"]), norm(t["relation"]), norm(t["target"]))


def linearize(triples):
    return " ; ".join(f"{t['source']} | {t['relation']} | {t['target']}" for t in triples)


def read_records(root, size, smoke=False):
    if smoke:
        files = [Path(root) / "data" / "factkg_fake" / "examples.jsonl"]
    else:
        files = sorted((Path(root) / "data" / "factkg").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError("Dados do FactKG não encontrados em data/factkg")
    total = eligible = excluded = 0
    records, audit = [], {"claim": {"graphs": 0, "nodes_before": 0, "nodes_after": 0, "edges_before": 0, "edges_after": 0, "edges_removed": 0, "empty_after": 0},
                          "evidence": {"graphs": 0, "nodes_before": 0, "nodes_after": 0, "edges_before": 0, "edges_after": 0, "edges_removed": 0, "empty_after": 0}}
    threshold = int(max(0.0, min(1.0, size)) * (2**64 - 1))
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                total += 1
                raw = json.loads(line)
                digest = int(hashlib.sha256(str(raw.get("id", total)).encode()).hexdigest()[:16], 16)
                if not smoke and size < 1.0 and digest > threshold:
                    continue
                if raw.get("label") not in {"SUPPORTS", "REFUTES"}:
                    excluded += 1; continue
                try:
                    claim, ca = graph_triples(raw["grafo_claim"]); evidence, ea = graph_triples(raw["grafo_evidencia"])
                except Exception:
                    excluded += 1; continue
                for side, stats in (("claim", ca), ("evidence", ea)):
                    audit[side]["graphs"] += 1
                    for key in ("nodes_before", "nodes_after", "edges_before", "edges_after", "edges_removed", "empty_after"):
                        audit[side][key] += stats[key]
                if not claim or not evidence:
                    excluded += 1; continue
                eligible += 1
                group = " ".join(str(raw.get("claim", "")).casefold().split())
                records.append({"id": str(raw["id"]), "claim": str(raw.get("claim", "")), "evidence": str(raw.get("evidencia", "")),
                                "label": int(raw["label"] == "SUPPORTS"), "label_name": raw["label"], "group_id": group,
                                "types": ",".join(raw.get("types", [])), "claim_triples": claim, "evidence_triples": evidence,
                                "graph_claim": linearize(claim), "graph_evidence": linearize(evidence)})
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Nenhum exemplo elegível")
    groups = df.group_id.to_numpy()
    all_ix = np.arange(len(df))
    train_ix, test_ix = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=31415).split(all_ix, groups=groups))
    train_ix, val_ix = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=31416).split(train_ix, groups=groups[train_ix]))
    roles = np.array(["test"] * len(df), dtype=object)
    roles[test_ix] = "test"
    roles[np.asarray(test_ix)[0:0]] = "test"
    train_pool = np.asarray(test_ix) if False else np.asarray(next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=31415).split(all_ix, groups=groups))[0])
    _, val_local = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=31416).split(train_pool, groups=groups[train_pool]))
    val_ix = train_pool[val_local]; train_ix = np.setdiff1d(train_pool, val_ix)
    roles[train_ix] = "train"; roles[val_ix] = "validation"
    df["role"] = roles
    audit.update({"input": total, "eligible": eligible, "excluded": total - eligible, "sample_fraction": size,
                  "coverage": eligible / total if total else 0.0, "test_scope": "Divisão interna agrupada por claim; não é o teste oficial.",
                  "labels": df.label_name.value_counts().to_dict()})
    return df, audit


def feature_matrix(df):
    rows = []
    for r in df.itertuples():
        c, e = r.claim_triples, r.evidence_triples
        ck, ek = {triple_key(x) for x in c}, {triple_key(x) for x in e}
        cr, er = {norm(x["relation"]) for x in c}, {norm(x["relation"]) for x in e}
        cn, en = {norm(x["source"]) for x in c} | {norm(x["target"]) for x in c}, {norm(x["source"]) for x in e} | {norm(x["target"]) for x in e}
        overlap = ck & ek
        rows.append([len(c), len(e), len(ck), len(ek), len(cr), len(er), len(cn), len(en), len(overlap),
                     len(overlap) / max(len(ck), 1), len(cr & er) / max(len(cr), 1), len(cn & en) / max(len(cn), 1),
                     int(len(overlap) == len(ck)), int(len(overlap) > 0), len(e) / max(len(c), 1),
                     int("negation" in r.types.casefold()), int("multi-hop" in r.types.casefold() or "multi_hop" in r.types.casefold())])
    return np.asarray(rows, dtype=np.float32)
