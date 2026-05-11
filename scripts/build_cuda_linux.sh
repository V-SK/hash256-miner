#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_ARCH="${CUDA_ARCH:-auto}"
OUT="${OUT:-hash_gpu_cuda}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This builder is for Linux CUDA hosts."
  exit 2
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install NVIDIA CUDA Toolkit or use a release binary."
  exit 2
fi

if [[ "$CUDA_ARCH" == "auto" ]]; then
  CUDA_ARCH="$(./scripts/detect_cuda_arch.sh)"
fi

echo "building $OUT for $CUDA_ARCH"

cmd=(
  nvcc
  -O3
  -std=c++17
  -arch="$CUDA_ARCH"
  -Xptxas
  -O3,-dlcm=ca
  --extra-device-vectorization
  -Xcompiler
  -O3
  hash_gpu_cuda.cu
  -o
  "$OUT"
)

if [[ -n "${EXTRA_NVCC_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_flags=($EXTRA_NVCC_FLAGS)
  cmd+=("${extra_flags[@]}")
fi

if ! "${cmd[@]}"; then
  echo "optimized nvcc flags failed; retrying with conservative -O3 build"
  nvcc -O3 -std=c++17 -arch="$CUDA_ARCH" hash_gpu_cuda.cu -o "$OUT"
fi

chmod +x "$OUT"
echo "built $OUT for $CUDA_ARCH"
