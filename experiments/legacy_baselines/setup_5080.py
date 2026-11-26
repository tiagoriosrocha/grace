#!/usr/bin/env python3
"""Create the isolated CUDA environment used by the article run."""
from pathlib import Path
import platform
import subprocess
import sys
import venv

root = Path(__file__).resolve().parent
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("Use Python 3.10, 3.11 ou 3.12.")
if platform.system() != "Linux" or platform.machine().lower() not in ("amd64", "x86_64"):
    raise SystemExit("O instalador da RTX 5080 requer Linux x86_64.")
env = root / ".venv"
if not env.exists():
    venv.EnvBuilder(with_pip=True).create(env)
python = env / "bin" / "python"


def run(*args):
    subprocess.run([str(python), *args], check=True, cwd=root)


run("-m", "pip", "install", "--upgrade", "pip")
run("-m", "pip", "install", "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128")
run("-m", "pip", "install", "-r", "requirements.txt")
run("-m", "pip", "check")
run("-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())")
run('-c', 'import torch; x=torch.randn(32,32,device="cuda"); print((x@x).sum().item()); assert torch.cuda.get_device_capability()[0]<12 or "sm_120" in torch.cuda.get_arch_list()')
run("-m", "pytest", "-q")
run("run.py", "--smoke", "--device", "cuda")
(env / "READY_5080").write_text("CUDA environment and smoke test completed.\n", encoding="utf-8")
print("Ambiente pronto. O treino FactKG ainda não foi iniciado.")
