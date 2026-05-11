#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_ARCH="${CUDA_ARCH:-sm_86}"
OUT="${OUT:-hash_gpu_cuda}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This builder is for Linux CUDA hosts."
  exit 2
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install NVIDIA CUDA Toolkit or use a release binary."
  exit 2
fi

nvcc -O3 -std=c++17 -arch="$CUDA_ARCH" hash_gpu_cuda.cu -o "$OUT"
chmod +x "$OUT"
echo "built $OUT for $CUDA_ARCH"
