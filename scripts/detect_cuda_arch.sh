#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CUDA_ARCH:-}" && "${CUDA_ARCH}" != "auto" ]]; then
  echo "$CUDA_ARCH"
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:].' || true)"
  if [[ "$cc" =~ ^[0-9]+$ && ${#cc} -ge 2 ]]; then
    echo "sm_${cc}"
    exit 0
  fi

  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  case "$name" in
    *H100*|*H200*|*GH200*) echo "sm_90"; exit 0 ;;
    *RTX\ 50*|*5090*|*5080*|*5070*) echo "sm_120"; exit 0 ;;
    *RTX\ 40*|*4090*|*4080*|*4070*|*4060*|*L4*|*L40*) echo "sm_89"; exit 0 ;;
    *A100*|*A800*|*A30*) echo "sm_80"; exit 0 ;;
    *RTX\ 30*|*3090*|*3080*|*3070*|*3060*|*A10*|*A16*|*A40*) echo "sm_86"; exit 0 ;;
    *RTX\ 20*|*2080*|*2070*|*2060*|*T4*) echo "sm_75"; exit 0 ;;
    *V100*) echo "sm_70"; exit 0 ;;
    *P100*) echo "sm_60"; exit 0 ;;
  esac
fi

echo "sm_86"
