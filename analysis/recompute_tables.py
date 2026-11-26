#!/usr/bin/env python3
"""Recompute published point metrics from saved predictions; no model execution.

Default: standard library only. Optional --bootstrap N requires NumPy.
Original run files are never modified; outputs go to analysis/generated/.
"""
from pathlib import Path
from collections import defaultdict
from itertools import groupby
from statistics import mean, stdev
import argparse
import csv
import json
import math
import platform

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 21, 42]
MODELS = ['text_tfidf', 'graph_pair_transformer', 'kgat_style', 'gmn_style', 'relational_gnn']
DATASETS = ['factkg', 'hover', 'fever', 'feverous']
VARIANTS = ['no-null', 'no-relational-context', 'no-nli', 'no-condition-aggregation']
METRICS = ['macro_f1', 'accuracy', 'balanced_accuracy', 'auc', 'brier']
PAPER_F1 = {
    'factkg': [.830, .975, .924, .929, .959],
    'hover': [.482, .368, .422, .540, .596],
    'fever': [.535, .659, .671, .691, .800],
    'feverous': [.637, .796, .675, .776, .793],
}
PAPER_SD = {
    'factkg': [.000, .003, .005, .009, .002],
    'hover': [.000, .000, .087, .022, .002],
    'fever': [.000, .285, .007, .008, .008],
    'feverous': [.000, .010, .012, .006, .006],
}
PAPER_DELTAS_PP = {
    'factkg': [-1.413, -.217, .036, -.252],
    'hover': [-1.807, .454, -2.950, .196],
    'fever': [-.505, -.978, -9.958, -.927],
    'feverous': [.008, .002, -3.655, .109],
}


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(rows):
    counts = [0, 0, 0, 0]
    brier = []
    ranked = []
    for row in rows:
        y, p = int(row['label']), float(row['probability'])
        require(y in (0, 1) and math.isfinite(p) and 0 <= p <= 1, 'Invalid label/probability')
        prediction = int(p >= .5)
        require(int(row['prediction']) == prediction, 'Saved prediction differs from the 0.5 rule')
        counts[2 * y + prediction] += 1
        brier.append((p - y) ** 2)
        ranked.append((p, y))
    tn, fp, fn, tp = counts
    require(tp + fn > 0 and tn + fp > 0, 'Both test classes are required')
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    negative_below = 0
    wins = 0.
    for _, items in groupby(sorted(ranked), key=lambda item: item[0]):
        items = list(items)
        positive = sum(y for _, y in items)
        negative = len(items) - positive
        wins += positive * (negative_below + .5 * negative)
        negative_below += negative
    return dict(macro_f1=(f0 + f1) / 2, accuracy=(tp + tn) / len(rows),
                balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                auc=wins / ((tp + fn) * (tn + fp)), brier=math.fsum(brier) / len(rows))


def signature(rows):
    return [(r['id'], r['group_id'], r['label']) for r in rows]


def bootstrap(predictions, models, n, random_state):
    import numpy as np
    base = predictions[models[0], 7]
    groups = sorted({r['group_id'] for r in base})
    lookup = {g: i for i, g in enumerate(groups)}
    counts = np.zeros((len(models), len(SEEDS), len(groups), 4), dtype=np.int64)
    for mi, model in enumerate(models):
        for si, seed in enumerate(SEEDS):
            for row in predictions[model, seed]:
                cell = 2 * int(row['label']) + int(float(row['probability']) >= .5)
                counts[mi, si, lookup[row['group_id']], cell] += 1
    def f1(c):
        tn, fp, fn, tp = np.moveaxis(c.astype(float), -1, 0)
        d0, d1 = 2 * tn + fp + fn, 2 * tp + fp + fn
        return .5 * (np.divide(2 * tn, d0, out=np.zeros_like(tn), where=d0 != 0)
                     + np.divide(2 * tp, d1, out=np.zeros_like(tp), where=d1 != 0))
    draws = np.empty((n, len(models)))
    rng = np.random.default_rng(random_state)
    for i in range(n):
        sampled = rng.integers(len(groups), size=len(groups))
        draws[i] = f1(counts[:, :, sampled, :].sum(axis=2)).mean(axis=1)
    observed = f1(counts.sum(axis=2)).mean(axis=1)
    return observed, draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=0, metavar='N', help='Optional paired group bootstrap; requires NumPy')
    args = parser.parse_args()
    require(args.bootstrap >= 0, 'Bootstrap count must be nonnegative')
    if args.bootstrap:
        try:
            import numpy as np
        except ImportError:
            raise SystemExit('Bootstrap requires NumPy. Install numpy==1.26.4 in Python 3.10–3.12.')
    records = json.loads((ROOT / 'provenance/runs.json').read_text())
    per_seed, summary_rows, ablation_rows = [], [], []
    paired = {ds: {} for ds in DATASETS}
    ablated = {ds: {} for ds in DATASETS}
    reference_signatures, reference_splits, reference_inputs = {}, {}, {}
    checks = 0
    for record in records:
        directory = ROOT / record['path']
        manifest = json.loads((directory / 'manifest.json').read_text())
        ds = record['dataset']
        require(manifest['status'] == 'complete' and manifest['scientific_result'], 'Incomplete run')
        require(manifest['seeds'] == SEEDS, 'Unexpected seeds')
        require(reference_splits.setdefault(ds, manifest['split_sha256']) == manifest['split_sha256'], 'Different splits')
        require(reference_inputs.setdefault(ds, manifest['input_sha256']) == manifest['input_sha256'], 'Different inputs')
        grouped = defaultdict(list)
        for row in read_csv(directory / 'predictions.csv'):
            if row['model'] in record['models']:
                require(row['split'] == 'test', 'Non-test prediction')
                grouped[row['model'], int(row['seed'])].append(row)
        archived_metrics = {(r['model'], int(r['seed'])): r for r in read_csv(directory / 'metrics.csv') if r['split'] == 'test'}
        archived_summary = {r['model']: r for r in read_csv(directory / 'summary.csv')}
        split_test = {r['id']: (r['group_id'], r['label']) for r in read_csv(directory / 'splits.csv') if r['evaluation_role'] == 'test'}
        for model in record['models']:
            measured = []
            for seed in SEEDS:
                rows = sorted(grouped[model, seed], key=lambda r: r['id'])
                require(bool(rows), f'Missing predictions: {record["run"]}/{model}/{seed}')
                require(len({r['id'] for r in rows}) == len(rows), 'Duplicate prediction ID')
                sig = signature(rows)
                require(reference_signatures.setdefault(ds, sig) == sig, f'Unpaired test IDs/labels/groups: {ds}')
                require({r['id']: (r['group_id'], r['label']) for r in rows} == split_test, 'Predictions differ from saved test split')
                result = score(rows)
                for metric in METRICS:
                    require(abs(result[metric] - float(archived_metrics[model, seed][metric])) < 1e-10,
                            f'Archived metric differs: {ds}/{record["variant"]}/{model}/{seed}/{metric}')
                    checks += 1
                measured.append(result)
                per_seed.append(dict(dataset=ds, role=record['role'], variant=record['variant'], model=model,
                                     seed=seed, n=len(rows), run=record['run'], **result))
                if record['role'] == 'ablation':
                    ablated[ds][record['variant'], seed] = rows
                else:
                    require((model, seed) not in paired[ds], 'Duplicate selected main result')
                    paired[ds][model, seed] = rows
            aggregate = {}
            for metric in METRICS:
                aggregate[metric + '_mean'] = mean(r[metric] for r in measured)
                aggregate[metric + '_std'] = stdev(r[metric] for r in measured)
                for suffix in ['_mean', '_std']:
                    require(abs(aggregate[metric + suffix] - float(archived_summary[model][metric + suffix])) < 1e-10,
                            f'Archived summary differs: {record["run"]}/{model}/{metric}{suffix}')
                    checks += 1
            summary_rows.append(dict(dataset=ds, role=record['role'], variant=record['variant'], model=model,
                                     run=record['run'], **aggregate))
    main_rows = [r for r in summary_rows if r['role'] != 'ablation']
    lookup = {(r['dataset'], r['model']): r for r in main_rows}
    for ds in DATASETS:
        for i, model in enumerate(MODELS):
            require(round(lookup[ds, model]['macro_f1_mean'], 3) == PAPER_F1[ds][i], f'Manuscript F1 mismatch: {ds}/{model}')
            require(round(lookup[ds, model]['macro_f1_std'], 3) == PAPER_SD[ds][i], f'Manuscript SD mismatch: {ds}/{model}')
        for i, variant in enumerate(VARIANTS):
            row = next(r for r in summary_rows if r['dataset'] == ds and r['role'] == 'ablation' and r['variant'] == variant)
            full = lookup[ds, 'relational_gnn']['macro_f1_mean']
            difference = 100 * (row['macro_f1_mean'] - full)
            require(round(difference, 3) == PAPER_DELTAS_PP[ds][i], f'Manuscript ablation mismatch: {ds}/{variant}')
            ablation_rows.append(dict(dataset=ds, variant=variant, full_macro_f1=full,
                                      ablated_macro_f1=row['macro_f1_mean'], difference_pp=difference,
                                      auc_difference_pp=100 * (row['auc_mean'] - lookup[ds, 'relational_gnn']['auc_mean']),
                                      brier_difference_pp=100 * (row['brier_mean'] - lookup[ds, 'relational_gnn']['brier_mean']),
                                      run=row['run']))
    out = ROOT / 'analysis/generated'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'per_seed.csv', per_seed)
    write_csv(out / 'main_results.csv', main_rows)
    write_csv(out / 'ablation_summary.csv', ablation_rows)
    if args.bootstrap:
        comparisons = []
        for ds in DATASETS:
            pred = dict(paired[ds])
            models = list(MODELS)
            for variant in VARIANTS:
                models.append(variant)
                for seed in SEEDS:
                    pred[variant, seed] = ablated[ds][variant, seed]
            cfg = json.loads((ROOT / next(r['path'] for r in records if r['dataset'] == ds and r['role'] == 'comparison') / 'config.json').read_text())
            observed, draws = bootstrap(pred, models, args.bootstrap, cfg['split_seed'] + 9001)
            reference_index = models.index('relational_gnn')
            for i, comparator in enumerate(models):
                if i == reference_index:
                    continue
                values = 100 * (draws[:, reference_index] - draws[:, i])
                low, high = np.quantile(values, [.025, .975])
                comparisons.append(dict(dataset=ds, model_a='relational_gnn', model_b=comparator,
                                        difference_pp=100 * (observed[reference_index] - observed[i]),
                                        ci95_low_pp=float(low), ci95_high_pp=float(high), samples=args.bootstrap))
            print(f'Bootstrap complete: {ds}', flush=True)
        write_csv(out / 'bootstrap_comparisons.csv', comparisons)
    report = dict(status='passed', python=platform.python_version(), selected_runs=len(records),
                  model_seed_combinations=len(per_seed), archived_metric_and_summary_checks=checks,
                  manuscript_mean_and_sd_cells=40, manuscript_ablation_differences=16,
                  test_pairing_verified=True, bootstrap_replicates=args.bootstrap, training_rerun=False)
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(per_seed)} model/seed combinations, {checks} metric/summary checks, 40 manuscript mean/SD cells, 16 ablation differences.')


if __name__ == '__main__':
    main()
