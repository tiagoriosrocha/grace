#!/usr/bin/env python3
"""Run the single, article-facing comparison for Alternative 5."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

import final_pipeline
from final_pipeline import run_experiment
from hover_exp.data import DATASETS
from hover_exp.fixtures import write_fixture


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="factkg",
                        help="dataset a usar; os arquivos são lidos de data/<dataset>/")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="small offline protocol check; not scientific results")
    parser.add_argument('--audit-only', action='store_true', help='prepare and validate real split without downloading or training models')
    parser.add_argument("--output", type=Path, help="explicit output directory")
    parser.add_argument("--systems", default="all",
                        help="comma-separated systems, or all (for example local_factcheck_pipeline)")
    parser.add_argument("--system", action="append", dest="system_args",
                        help="system to run; may be repeated")
    parser.add_argument("--fraction", "--dataset-fraction", dest="fraction", type=float, default=1.0,
                        help="deterministic fraction of the export to process, in (0,1]")
    args = parser.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cfg["dataset"] = args.dataset
    cfg["smoke"] = args.smoke
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction deve estar no intervalo (0, 1].")
    if args.smoke and args.fraction != 1.0:
        parser.error("--fraction não se aplica ao smoke sintético.")
    cfg["sample_fraction"] = args.fraction
    available = tuple(final_pipeline.MODEL_ORDER)
    requested = list(args.system_args or [])
    if args.systems != 'all':
        requested.extend(x.strip() for x in args.systems.split(',') if x.strip())
    selected = list(available) if not requested else requested
    invalid = sorted(set(selected) - set(available))
    if invalid:
        parser.error(f"sistema(s) inválido(s): {', '.join(invalid)}. Opções: {', '.join(available)}")
    cfg['systems'] = selected
    if args.smoke:
        cfg.update(dataset='hover', seeds=[7], epochs=1, x23_epochs=1, transformer_epochs=1,
                   bootstrap_samples=100, hidden_dim=16, x23_dim=16,
                   local_hidden_dim=16, local_epochs=2, local_patience=2,
                   relation_bases=2, layers=1, x23_claim_max_length=32, transformer_max_length=64)
        import tempfile
        fixture = Path(tempfile.mkdtemp(prefix='alt5-smoke-')) / 'fixture.jsonl'
        write_fixture(fixture, n=64)
        input_path = fixture
    else:
        input_path = ROOT / cfg["data_paths"][args.dataset]
    if not input_path.exists():
        raise SystemExit(f"Dados não encontrados: {input_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda foi solicitado, mas CUDA não está disponível.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = 'audit' if args.audit_only else ('smoke' if args.smoke else 'article')
    output = args.output or ROOT / 'results' / f'{prefix}_{stamp}'
    print(f"Dataset: {cfg['dataset']}; dispositivo: {device}; saída: {output}", flush=True)
    run_experiment(cfg, input_path, output, device, audit_only=args.audit_only)
    print(f"Concluído: {output}", flush=True)


if __name__ == "__main__":
    main()
