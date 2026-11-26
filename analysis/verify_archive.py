#!/usr/bin/env python3
"""Verify the publication archive with Python's standard library only."""
from pathlib import Path
import hashlib
import json
import platform

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    checked = 0
    for line in (ROOT / 'provenance/checksums.sha256').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise SystemExit(f'Missing or changed archive file: {rel}')
        checked += 1
    inputs = json.loads((ROOT / 'provenance/datasets.json').read_text())
    datasets = {}
    for record in inputs:
        directory = ROOT / 'data' / record['dataset']
        expected_names = {f['file'] for f in record['files']}
        if {p.name for p in directory.glob('*.jsonl')} != expected_names:
            raise SystemExit(f'Changed input file set: {record["dataset"]}')
        digest = hashlib.sha256()
        for item in sorted(record['files'], key=lambda x: x['file']):
            path = directory / item['file']
            if sha(path) != item['sha256']:
                raise SystemExit(f'Changed input: {path.name}')
            digest.update(path.name.encode('utf-8'))
            digest.update(item['sha256'].encode('ascii'))
        if digest.hexdigest() != record['input_sha256']:
            raise SystemExit(f'Changed input directory hash: {record["dataset"]}')
        datasets[record['dataset']] = digest.hexdigest()
    runs = json.loads((ROOT / 'provenance/runs.json').read_text())
    split_by_dataset = {}
    source_checks = 0
    for run in runs:
        directory = ROOT / run['path']
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['status'] != 'complete' or not manifest['scientific_result']:
            raise SystemExit(f'Incomplete/non-scientific run: {run["path"]}')
        if manifest['seeds'] != [7, 21, 42]:
            raise SystemExit(f'Unexpected seeds: {run["path"]}')
        if manifest['input_sha256'] != datasets[run['dataset']]:
            raise SystemExit(f'Input mismatch: {run["path"]}')
        split = sha(directory / 'splits.csv')
        if split != manifest['split_sha256']:
            raise SystemExit(f'Split mismatch: {run["path"]}')
        if split_by_dataset.setdefault(run['dataset'], split) != split:
            raise SystemExit(f'Runs use different splits: {run["dataset"]}')
        for rel, expected in manifest['source_sha256'].items():
            if sha(ROOT / 'experiments' / run['snapshot'] / rel) != expected:
                raise SystemExit(f'Source mismatch: {run["snapshot"]}/{rel}')
            source_checks += 1
    for snapshot in ['comparison', 'ablation', 'legacy_baselines']:
        if (ROOT / 'experiments' / snapshot / 'data').resolve() != (ROOT / 'data').resolve():
            raise SystemExit(f'Invalid shared data link: {snapshot}')
    report = dict(status='passed', python=platform.python_version(), checksummed_files=checked,
                  datasets=len(datasets), selected_runs=len(runs), source_hash_checks=source_checks,
                  split_hashes=split_by_dataset, training_rerun=False)
    output = ROOT / 'analysis/generated'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'archive_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {checked} files, {len(datasets)} datasets, {len(runs)} runs, {source_checks} source checks.')


if __name__ == '__main__':
    main()
