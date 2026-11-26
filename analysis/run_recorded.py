#!/usr/bin/env python3
"""Retrain a selected run with its saved configuration and original system order.

Requires the ML dependencies in experiments/<snapshot>/requirements.txt.
This utility does not resume checkpoints or overwrite the published results.
"""
from pathlib import Path
from datetime import datetime
import argparse
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Exact run name from provenance/runs.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--audit-only', action='store_true', help='Check preparation/split without training')
    args = parser.parse_args()
    runs = json.loads((ROOT / 'provenance/runs.json').read_text())
    selected = [r for r in runs if r['run'] == args.run]
    if len(selected) != 1:
        parser.error('Choose one of the exact run names in provenance/runs.json.')
    record = selected[0]
    snapshot = ROOT / 'experiments' / record['snapshot']
    cfg = json.loads((ROOT / record['path'] / 'config.json').read_text())
    cfg['device'] = args.device
    sys.path.insert(0, str(snapshot))
    os.chdir(snapshot)
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; prepare the original environment or select --device cpu.')
    run_experiment = importlib.import_module('final_pipeline').run_experiment
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = 'audit' if args.audit_only else 'reproduction'
    output = snapshot / 'results' / f'{prefix}_{record["dataset"]}_{record["variant"]}_{stamp}'
    inputs = snapshot / cfg['data_paths'][record['dataset']]
    print(f'Recorded configuration: {record["run"]}', flush=True)
    print(f'Systems in original order: {cfg["systems"]}', flush=True)
    print(f'New output: {output}', flush=True)
    run_experiment(cfg, inputs, output, torch.device(args.device), audit_only=args.audit_only)


if __name__ == '__main__':
    main()
