#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -f .venv/READY_5080 ] || ! .venv/bin/python -c 'import torch, transformers, sklearn; assert torch.cuda.is_available()' 2>/dev/null; then
    if [ -n "${PYTHON_BIN:-}" ]; then
        "$PYTHON_BIN" setup_5080.py
    else
        selected_python=''
        for candidate in python3.12 python3.11 python3.10 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' 2>/dev/null; then
                selected_python="$candidate"
                break
            fi
        done
        if [ -z "$selected_python" ]; then
            echo 'Instale Python 3.10–3.12 ou defina PYTHON_BIN=/caminho/python3.12.' >&2
            exit 1
        fi
        "$selected_python" setup_5080.py
    fi
fi
exec .venv/bin/python run.py "$@"
