#!/usr/bin/env bash
set -euo pipefail

APPLY=0
POWER_LIMIT=""
LOCK_GRAPHICS_CLOCK=""
UNLOCK_CLOCKS=0

usage() {
  cat <<'EOF'
usage: ./scripts/tune_linux_nvidia.sh [--apply] [--power-limit W] [--lock-graphics-clock MHz] [--unlock-clocks]

Default mode is dry-run. Nothing is changed unless --apply is present.
Safe defaults enable persistence mode and set CPU governors to performance when available.
Power limit and clock locking are opt-in because supported ranges vary by GPU.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --power-limit) POWER_LIMIT="${2:-}"; shift 2 ;;
    --lock-graphics-clock) LOCK_GRAPHICS_CLOCK="${2:-}"; shift 2 ;;
    --unlock-clocks) UNLOCK_CLOCKS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This tuner is for Linux NVIDIA hosts."
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA driver first."
  exit 2
fi

run_root() {
  if [[ "$APPLY" -eq 1 ]]; then
    sudo "$@"
  else
    printf '+ sudo'
    printf ' %q' "$@"
    printf '\n'
  fi
}

echo "Detected GPUs:"
nvidia-smi --query-gpu=index,name,power.limit,power.draw,clocks.gr,clocks.mem,temperature.gpu --format=csv,noheader || true
echo

run_root nvidia-smi -pm 1

if [[ "$UNLOCK_CLOCKS" -eq 1 ]]; then
  run_root nvidia-smi -rgc
fi

if [[ -n "$POWER_LIMIT" ]]; then
  run_root nvidia-smi -pl "$POWER_LIMIT"
fi

if [[ -n "$LOCK_GRAPHICS_CLOCK" ]]; then
  run_root nvidia-smi -lgc "${LOCK_GRAPHICS_CLOCK},${LOCK_GRAPHICS_CLOCK}"
fi

if compgen -G "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor" >/dev/null; then
  for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [[ "$APPLY" -eq 1 ]]; then
      echo performance | sudo tee "$governor" >/dev/null
    else
      echo "+ echo performance | sudo tee $governor >/dev/null"
    fi
  done
else
  echo "CPU governor controls not found; skipping."
fi

if [[ "$APPLY" -eq 0 ]]; then
  echo
  echo "dry-run only; rerun with --apply to change settings."
fi
