#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
CUDA_ARCH="${CUDA_ARCH:-auto}"
LINUX_CUDA_BIN="./hash_gpu_cuda"
WINDOWS_CUDA_BIN="./hash_gpu_cuda.exe"

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

if [[ -f "$LINUX_CUDA_BIN" && ! -x "$LINUX_CUDA_BIN" ]]; then
  chmod +x "$LINUX_CUDA_BIN"
fi

if [[ ! -x "$LINUX_CUDA_BIN" ]]; then
  if [[ -f "$WINDOWS_CUDA_BIN" ]]; then
    echo "Found $WINDOWS_CUDA_BIN, but Linux cannot run the Windows CUDA .exe."
    echo "Linux miners need the native no-extension binary named hash_gpu_cuda."
  fi
  if command -v nvcc >/dev/null 2>&1; then
    ./scripts/build_cuda_linux.sh
  else
    echo "missing native Linux CUDA binary: $LINUX_CUDA_BIN"
    echo "Install NVIDIA CUDA Toolkit and rerun this script, or download hash-miner-linux-cuda-x64.tar.gz."
    exit 2
  fi
fi

if [[ ! -x "$LINUX_CUDA_BIN" ]]; then
  echo "Linux CUDA setup did not produce an executable $LINUX_CUDA_BIN."
  echo "Rebuild with ./scripts/build_cuda_linux.sh or download the Linux CUDA release bundle."
  exit 2
fi

chmod +x hash_miner.py hash_pool_miner.py hash_cuda_pool_miner.py
python hash_miner.py --doctor --backend cuda
