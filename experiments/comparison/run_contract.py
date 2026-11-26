"""Immutable output directories prevent reuse across incompatible experiments."""
import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

from hover_exp.data import file_hash, save_json


def start_run(out, cfg, input_path):
    out = Path(out)
    # Refuse even an existing empty directory: concurrent writers cannot share it.
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in ('*.py','hover_exp/*.py','baselines/*.py','factcheck_pipeline/*.py',
                        '*.json','*.sh','requirements*.txt')
               for p in sorted(root.glob(pattern)) if p.is_file()}
    packages = {}
    for name in ('torch','torch-geometric','numpy','pandas','scikit-learn','transformers','tokenizers','safetensors'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    manifest = dict(status='running', scientific_result=False, config=cfg,
                    dataset=cfg['dataset'], input=str(input_path), input_sha256=file_hash(input_path),
                    source_sha256=sources, packages=packages, python=sys.version,
                    platform=platform.platform(), protocol=cfg['protocol_version'])
    save_json(out/'config.json',cfg)
    save_json(out/'manifest.json',manifest)
    return manifest
