#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is for Linux. Use install_macos.sh on macOS."
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

if [[ ! -x ./hash_gpu_cuda ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    ./scripts/build_cuda_linux.sh
  else
    echo "hash_gpu_cuda is missing and nvcc was not found."
    echo "Install NVIDIA CUDA Toolkit or download a matching binary release."
  fi
fi

chmod +x hash_miner.py hash_pool_miner.py hash_cuda_pool_miner.py
python hash_miner.py --doctor --backend cuda || true
