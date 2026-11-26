"""Unified, leakage-audited experiment runner for Alternative 5.

The public model names in this module are deliberately descriptive.  The
older implementation modules are used only as internal building blocks.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hover_exp.aggregation import extract
from hover_exp.data import file_hash, prepare, features, save_json
from hover_exp.experiment import metrics, predict, seed_all, train_one
from hover_exp.model import Verifier
from hover_exp.x23_data import prepare_x23
from hover_exp.x23_train import make_model, probabilities, train as train_attention
from baselines.runner import transformer
from reporting import write_reports as _write_reports
from run_contract import start_run
from factcheck_pipeline.local_alt6 import run as run_local_alt6
from literature_baselines import train_literature_baseline


MODEL_ORDER = [
    "majority_baseline",
    "text_tfidf",
    "graph_pair_transformer",
    "matching_nli",
    "kgat_style",
    "gmn_style",
    "relational_gnn",
    "claim_evidence_attention",
    "local_factcheck_pipeline",
]


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def records_frame(records: list[dict]) -> pd.DataFrame:
    """Create the common tabular view consumed by all baselines."""
    rows = []
    for r in records:
        evidence_text = " ".join(
            f"{t['source_text']} {t['relation']} {t['target_text']}"
            for t in r["evidence_triples"])
        rows.append({
            "id": r["id"], "group_id": r["group_id"], "role": r["evaluation_role"],
            "label": int(r["label"]), "num_hops": r["num_hops"],
            "claim": r["claim"], "evidence": evidence_text,
            "graph_claim": " ; ".join(
                f"{t['source_text']} | {t['relation']} | {t['target_text']}"
                for t in r["claim_triples"]),
            "graph_evidence": " ; ".join(
                f"{t['source_text']} | {t['relation']} | {t['target_text']}"
                for t in r["evidence_triples"]),
            "claim_triples": r["claim_triples"],
            "evidence_triples": r["evidence_triples"],
        })
    return pd.DataFrame(rows)


def _indices(frame: pd.DataFrame):
    return tuple(np.flatnonzero(frame.role.to_numpy() == role)
                 for role in ("train", "validation", "test"))


def _add_predictions(rows, model, seed, ids, groups, labels, probabilities_, split="test"):
    for item_id, group_id, label, probability in zip(ids, groups, labels, probabilities_):
        rows.append(dict(id=item_id, group_id=group_id, label=int(label),
                         model=model, seed=int(seed), split=split,
                         probability=float(probability),
                         prediction=int(float(probability) >= .5)))


def _metric_row(model, seed, y, probabilities_, validation_macro_f1=None):
    result = dict(model=model, seed=int(seed), split="test", **metrics(y, probabilities_))
    if validation_macro_f1 is not None:
        result["validation_macro_f1"] = float(validation_macro_f1)
    return result




def _fit_text_tfidf(frame, train, test, seed, out):
    texts = (frame.claim + " [EVIDENCE] " + frame.evidence).to_numpy()
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit(texts[train], frame.label.to_numpy()[train])
    import joblib
    joblib.dump(model, out / f'text_tfidf_{seed}.joblib')
    return model.predict_proba(texts[test])[:, 1]


def _fit_matching_nli(records, bundle, cfg, train, test, seed, out):
    values = []
    for pair in bundle["pairs"]:
        value, _ = extract(pair, bundle["embeddings"], "matching_nli", cfg)
        values.append(value)
    names = sorted(values[0])
    matrix = np.asarray([[row[name] for name in names] for row in values], dtype=np.float32)
    pd.DataFrame(matrix, columns=names).assign(id=[r["id"] for r in records]).to_csv(
        out / "matching_nli_features.csv", index=False)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    labels = np.asarray([r["label"] for r in records], dtype=int)
    model.fit(matrix[train], labels[train])
    import joblib
    joblib.dump(dict(model=model, feature_names=names), out / f'matching_nli_{seed}.joblib')
    return model.predict_proba(matrix[test])[:, 1]


def _fit_relational_gnn(bundle, cfg, train, test, seed, device, out):
    model_dir = out / "models" / "relational_gnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_one(bundle, cfg, "relational_gnn", seed, device, model_dir)
    model = Verifier(bundle["embeddings"], len(bundle["relation_vocab"]), cfg, "relational_gnn").to(device)
    model.load_state_dict(torch.load(model_dir / f"relational_gnn_{seed}.pt", map_location=device, weights_only=True))
    return predict(model, [bundle["pairs"][int(i)] for i in test], device)


def _fit_literature_baseline(name, bundle, cfg, train, val, test, seed, device, out):
    return train_literature_baseline(name, bundle, cfg, train, val, test, seed, device, out)


def _fit_attention(records, bundle, cfg, train, test, seed, device, out):
    # The reference tensor is intentionally zeroed: the selected attention
    # model does not use NLI features, and this keeps the interface explicit.
    reference = {"ids": [r["id"] for r in records],
                 "features": torch.zeros((len(records), 31), dtype=torch.float32)}
    store, data = prepare_x23(records, bundle, cfg, device, out, reference)
    model_dir = out / "models" / "claim_evidence_attention"
    model_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = make_model(data, cfg, "claim_evidence_attention").to(device)
    train_attention(model, data["examples"], store, cfg, seed, device, model_dir)
    model.load_state_dict(torch.load(model_dir / f"claim_evidence_attention_{seed}.pt", map_location=device, weights_only=True))
    examples = [data["examples"][int(i)] for i in test]
    return probabilities(model, examples, store, device)


def _exact_match_diagnostic(records, test):
    values = []
    for i in test:
        claim = {(t["source_text"].casefold(), t["relation"].casefold(), t["target_text"].casefold())
                 for t in records[int(i)]["claim_triples"]}
        evidence = {(t["source_text"].casefold(), t["relation"].casefold(), t["target_text"].casefold())
                    for t in records[int(i)]["evidence_triples"]}
        values.append(float(claim <= evidence))
    return np.asarray(values, dtype=float)




def run_experiment(cfg: dict, input_path: Path, out: Path, device: torch.device, audit_only=False):
    started = time.time()
    cfg = dict(cfg)
    cfg["device"] = str(device)
    cfg["smoke"] = bool(cfg.get("smoke", False))
    if cfg.get('use_exact_features', False):
        raise ValueError('A comparação principal exige use_exact_features=false.')
    manifest = start_run(out, cfg, input_path)

    records, audit = prepare(input_path, out, cfg)
    manifest['split_sha256'] = file_hash(out/'splits.csv')
    if audit_only:
        manifest.update(status='audit_complete', seconds=time.time()-started)
        save_json(out/'manifest.json',manifest)
        return out
    selected = tuple(cfg.get('systems', MODEL_ORDER))
    needs_bundle = bool(set(selected) & {
        'graph_pair_transformer', 'matching_nli', 'relational_gnn',
        'claim_evidence_attention', 'local_factcheck_pipeline',
        'kgat_style', 'gmn_style'})
    bundle = features(records, cfg, device, out) if needs_bundle else None
    frame = records_frame(records)
    train, val, test = _indices(frame)
    ids = frame.iloc[test].id.tolist()
    groups = frame.iloc[test].group_id.tolist()
    labels = frame.iloc[test].label.to_numpy()
    metric_rows, prediction_rows = [], []

    majority = float(frame.iloc[train].label.mean() >= .5)
    for seed in cfg["seeds"]:
        if 'majority_baseline' in selected:
            p = np.full(len(test), majority, dtype=float)
            metric_rows.append(_metric_row("majority_baseline", seed, labels, p))
            _add_predictions(prediction_rows, "majority_baseline", seed, ids, groups, labels, p)

        if 'text_tfidf' in selected:
            p = _fit_text_tfidf(frame, train, test, seed, out)
            metric_rows.append(_metric_row("text_tfidf", seed, labels, p))
            _add_predictions(prediction_rows, "text_tfidf", seed, ids, groups, labels, p)

        if 'graph_pair_transformer' in selected:
            seed_all(seed)
            p, best = transformer(train, val, test, frame, cfg, seed, "graph_pair", device,
                                  output_dir=out/'models'/f'graph_pair_transformer_{seed}')
            metric_rows.append(_metric_row("graph_pair_transformer", seed, labels, p, best))
            _add_predictions(prediction_rows, "graph_pair_transformer", seed, ids, groups, labels, p)

        if 'matching_nli' in selected:
            p = _fit_matching_nli(records, bundle, cfg, train, test, seed, out)
            metric_rows.append(_metric_row("matching_nli", seed, labels, p))
            _add_predictions(prediction_rows, "matching_nli", seed, ids, groups, labels, p)

        for literature_name in ("kgat_style", "gmn_style"):
            if literature_name in selected:
                p, best = _fit_literature_baseline(
                    literature_name, bundle, cfg, train, val, test, seed, device, out)
                metric_rows.append(_metric_row(literature_name, seed, labels, p, best))
                _add_predictions(prediction_rows, literature_name, seed, ids, groups, labels, p)

        if 'relational_gnn' in selected:
            p = _fit_relational_gnn(bundle, cfg, train, test, seed, device, out)
            metric_rows.append(_metric_row("relational_gnn", seed, labels, p))
            _add_predictions(prediction_rows, "relational_gnn", seed, ids, groups, labels, p)

        if 'claim_evidence_attention' in selected:
            p = _fit_attention(records, bundle, cfg, train, test, seed, device, out)
            metric_rows.append(_metric_row("claim_evidence_attention", seed, labels, p))
            _add_predictions(prediction_rows, "claim_evidence_attention", seed, ids, groups, labels, p)

        # The local system is run once below.  It is the deterministic alt6
        # protocol, without the legacy weakly supervised MLP/GNN context.
        pd.DataFrame(metric_rows).to_csv(out/'metrics_partial.csv',index=False)
        pd.DataFrame(prediction_rows).to_csv(out/'predictions_partial.csv',index=False)

    if 'local_factcheck_pipeline' in selected:
        p, local_selection, local_nli_info, local_predictions = run_local_alt6(
            records, bundle, cfg, train, val, test, device, out)
        # Alt6's local protocol is deterministic after validation selection and
        # has no optimization-seed ensemble.  Repeat the fixed result for the
        # comparison table so paired reports remain well-defined.
        for seed in cfg['seeds']:
            metric_rows.append(_metric_row('local_factcheck_pipeline', seed, labels, p))
            _add_predictions(prediction_rows, 'local_factcheck_pipeline', seed,
                             ids, groups, labels, p)

    if selected == tuple(MODEL_ORDER):
        exact = _exact_match_diagnostic(records, test)
        exact_metrics = _metric_row("exact_match_diagnostic", 0, labels, exact)
        exact_metrics["diagnostic_only"] = True
        metric_rows.append(exact_metrics)
        _add_predictions(prediction_rows, "exact_match_diagnostic", 0, ids, groups, labels, exact)

    manifest.update(device=str(device), seeds=cfg['seeds'], seconds=time.time()-started,
                    selected_systems=list(selected),
                    embedding_info=bundle['info'] if bundle is not None else None,
                    nli_info=bundle['nli_info'] if bundle is not None else None,
                    local_system='local_factcheck_pipeline: alt6 frozen NLI + train-only claim aggregator')
    _write_reports(out, cfg, frame, metric_rows, prediction_rows, audit, manifest)
    return out
