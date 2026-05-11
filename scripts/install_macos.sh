#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS. Use install_linux_cuda.sh on Linux."
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 is missing. Install Python 3 first."
  exit 2
fi

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if ! command -v clang++ >/dev/null 2>&1; then
  echo "clang++ is missing. Run: xcode-select --install"
  exit 2
fi

clang++ -std=c++17 -O2 -fobjc-arc hash_gpu_metal.mm \
  -framework Foundation \
  -framework Metal \
  -o hash_gpu_metal

chmod +x hash_gpu_metal hash_miner.py hash_pool_miner.py hash_cuda_pool_miner.py
python hash_miner.py --doctor
