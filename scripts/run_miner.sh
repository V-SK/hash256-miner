#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: ./scripts/run_miner.sh 0xYourPayoutAddress [extra hash_miner.py args]"
  exit 2
fi

ADDRESS="$1"
shift

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON:-python3}"
"$PYTHON_BIN" hash_miner.py --address "$ADDRESS" "$@"
