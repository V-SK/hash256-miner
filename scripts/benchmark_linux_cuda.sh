#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-./hash_gpu_cuda}"
DEVICE="${DEVICE:-0}"
BENCH_SECONDS="${BENCH_SECONDS:-15}"
CHALLENGE="${CHALLENGE:-0x0000000000000000000000000000000000000000000000000000000000000000}"
TARGET="${TARGET:-0x0000000000000000000000000000000000000000000000000000000000000001}"
OUT="${OUT:-linux_cuda_benchmark_$(date -u +%Y%m%dT%H%M%SZ).csv}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This benchmark is for Linux CUDA hosts."
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "missing CUDA binary: $BIN"
  echo "run: ./scripts/build_cuda_linux.sh"
  exit 2
fi

configs=(
  "16777216 4 128 2"
  "16777216 8 256 2"
  "33554432 8 256 2"
  "67108864 8 256 2"
  "16777216 16 256 2"
  "33554432 16 256 2"
  "67108864 16 256 2"
  "33554432 8 256 3"
  "67108864 8 256 3"
)

echo "batch,iters,group,streams,rate_hps,checked,elapsed_s,exit_code" > "$OUT"
echo "writing $OUT"

for cfg in "${configs[@]}"; do
  read -r batch iters group streams <<< "$cfg"
  echo "bench batch=$batch iters=$iters group=$group streams=$streams seconds=$BENCH_SECONDS"
  set +e
  output="$("$BIN" \
    --device "$DEVICE" \
    --challenge "$CHALLENGE" \
    --target "$TARGET" \
    --seconds "$BENCH_SECONDS" \
    --batch "$batch" \
    --iters "$iters" \
    --group "$group" \
    --streams "$streams" 2>&1)"
  code=$?
  set -e
  rate="$(awk '/^rate:/ {print $2}' <<< "$output" | tail -n 1)"
  checked="$(awk '/^checked:/ {print $2}' <<< "$output" | tail -n 1)"
  elapsed="$(awk '/^elapsed:/ {gsub(/s$/, "", $2); print $2}' <<< "$output" | tail -n 1)"
  echo "$batch,$iters,$group,$streams,${rate:-0},${checked:-0},${elapsed:-0},$code" >> "$OUT"
done

echo
echo "top results:"
tail -n +2 "$OUT" | sort -t, -k5,5nr | head -n 5
