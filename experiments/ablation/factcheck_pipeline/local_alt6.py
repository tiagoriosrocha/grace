"""Alt6-compatible local FactKG pipeline used by Alternative 5.

This module deliberately keeps the local candidate and NLI trace separate from
the legacy alt5 local MLP.  The claim classifier is fitted only on train labels
and its threshold is selected only on validation labels.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .local_alt6_alignment import factual, norm, retrieve
from .local_alt6_nli import FrozenNLI

SIGNALS = ('nli_support', 'nli_contradiction', 'nli_neutral', 'aligned_support',
           'anchored_contradiction', 'exact_triple', 'subject_relation',
           'subject_object', 'semantic', 'path_support', 'candidate_count')
FEATURE_NAMES = [f'{s}_{stat}' for s in SIGNALS for stat in ('min', 'mean', 'max')]
FEATURE_NAMES += ['condition_count', 'missing_fraction', 'low_support_fraction']


def _save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def claim_features(condition_candidates):
    values = []
    for candidates in condition_candidates:
        rows = []
        for c in candidates:
            contradiction, support, neutral = c['nli']
            s, r, o = c['exact']
            direct = not c['path']
            rows.append([support, contradiction, neutral,
                         support * s * r if direct else 0.,
                         contradiction * s * r if direct else 0.,
                         s * r * o if direct else 0.,
                         s * r if direct else 0., s * o,
                         float(np.mean(c['similarity'])),
                         support if c['path'] else 0., len(candidates)])
        values.append(np.max(rows, axis=0) if rows else np.zeros(len(SIGNALS)))
    a = np.asarray(values) if values else np.zeros((1, len(SIGNALS)))
    vector = np.stack([a.min(0), a.mean(0), a.max(0)], axis=1).ravel().tolist()
    vector += [len(condition_candidates), float(np.mean(a[:, -1] == 0)),
               float(np.mean(a[:, 0] < .5))]
    return np.asarray(vector, dtype=float)


def _select(x_train, y_train, x_val, y_val, cfg):
    trials, best = [], None
    thresholds = sorted(np.linspace(.1, .9, 17), key=lambda t: (abs(t - .5), t))
    for c in cfg.get('aggregation_C', [.01, .1, 1., 10.]):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=float(c), class_weight='balanced', max_iter=2000,
            solver='lbfgs', random_state=int(cfg.get('split_seed', 0))))
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_val)[:, 1]
        for threshold in thresholds:
            score = float(f1_score(y_val, probabilities >= threshold,
                                   labels=[0, 1], average='macro', zero_division=0))
            trial = {'C': float(c), 'threshold': float(threshold),
                     'validation_macro_f1': score}
            trials.append(trial)
            if best is None or score > best[0] + 1e-12:
                best = score, model, trial
    if best is None:
        raise ValueError('Treino e validação precisam conter as duas classes.')
    selection = dict(best[2], selection_role='validation', fit_role='train', trials=trials)
    return best[1], selection


def _metrics(y, probability, threshold):
    prediction = np.asarray(probability) >= threshold
    return {'examples': len(y),
            'macro_f1': float(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'auc': float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
            'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            'predicted_support_fraction': float(prediction.mean())}


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(records, bundle, cfg, train, validation, test, device, out: Path):
    """Run the alt6 local method on alt5 records and return test probabilities."""
    import joblib
    import torch

    text_index = json.loads((out / 'text_index.json').read_text(encoding='utf-8'))
    vectors = {}
    for i, text in enumerate(text_index):
        vector = bundle['embeddings'][i].detach().cpu().numpy()
        vectors[text] = vector
        vectors.setdefault(norm(text), vector)
    local_cfg = dict(cfg, nli_max_length=256)
    nli = FrozenNLI(local_cfg, device)
    _save_json(out / 'local_protocol.json', {
        'protocol': 'alt6-local-supervised-v2',
        'embedding_model': cfg['embedding_model'],
        'embedding_revision': cfg['embedding_revision'],
        'nli_model': cfg['nli_model'],
        'nli_revision': cfg['nli_revision'],
        'nli_max_length': 256,
        'top_k': int(cfg.get('top_k', 4)),
        'path_candidates': int(cfg.get('path_candidates', 4)),
        'fit_role': 'train', 'selection_role': 'validation',
        'label_independent_retrieval': True,
    })
    features, rows = [], []
    for record in records:
        conditions = [t for t in record['claim_triples'] if factual(t)]
        candidates = [retrieve(t, record['evidence_triples'], vectors,
                              int(cfg.get('top_k', 4)), int(cfg.get('path_candidates', 4)))
                      for t in conditions]
        flat = [candidate for group in candidates for candidate in group]
        for candidate, score in zip(flat, nli.score(flat)):
            candidate['nli'] = score.tolist()
        features.append(claim_features(candidates))
        rows.append({'id': record['id'], 'evaluation_role': record['evaluation_role'],
                     'conditions': [{'condition_id': f'condition_{i:04d}',
                                     'claim_triple': condition,
                                     'candidates': group}
                                    for i, (condition, group) in enumerate(zip(conditions, candidates))]})
    nli_info = nli.metadata.copy()
    x = np.stack(features)
    roles = np.asarray([r['evaluation_role'] for r in records])
    labels = np.asarray([r['label'] for r in records])
    model, selection = _select(x[roles == 'train'], labels[roles == 'train'],
                               x[roles == 'validation'], labels[roles == 'validation'], cfg)
    _save_json(out / 'selection.json', selection)
    joblib.dump({'model': model, 'feature_names': FEATURE_NAMES,
                 'threshold': selection['threshold']}, out / 'aggregator.joblib')
    _save_json(out / 'coefficients.json',
               dict(zip(FEATURE_NAMES, map(float, model[-1].coef_[0]))))
    _write_csv(out / 'claim_features.csv', [
        {'id': record['id'], 'evaluation_role': record['evaluation_role'],
         **dict(zip(FEATURE_NAMES, map(float, vector)))}
        for record, vector in zip(records, x)])
    probabilities = model.predict_proba(x)[:, 1]
    threshold = float(selection['threshold'])
    predictions = [{'id': record['id'], 'group_id': record['group_id'],
                    'label': int(record['label']), 'split': record['evaluation_role'],
                    'model': 'local_factcheck_pipeline', 'score': float(probabilities[i]),
                    'threshold': threshold, 'prediction': int(probabilities[i] >= threshold)}
                   for i, record in enumerate(records)]
    _write_csv(out / 'local_predictions.csv', predictions)
    _save_json(out / 'local_metrics.json', [
        {'model': 'local_factcheck_pipeline', 'split': role,
         **_metrics(labels[roles == role], probabilities[roles == role], threshold)}
        for role in ('validation', 'test') if np.any(roles == role)])
    with (out / 'local_candidates.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    by_id = {p['id']: p for p in predictions}
    with (out / 'local_decisions.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            pred = by_id[row['id']]
            row = dict(row, claim=next(r['claim'] for r in records if r['id'] == row['id']),
                       gold=next(r['label_name'] for r in records if r['id'] == row['id']),
                       split=pred['split'], label=pred['label'],
                       prediction='SUPPORTS' if pred['prediction'] else 'REFUTES',
                       global_verdict='SUPPORTS' if pred['prediction'] else 'REFUTES',
                       score=pred['score'], threshold=threshold)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    _save_json(out / 'nli_info_local.json', nli_info)
    return probabilities[np.asarray(test)], selection, nli_info, predictions
