#!/usr/bin/env python3
"""CUDA stdout wrapper for the local HASH256 pool-server.

The wrapper fetches pool jobs, runs the local CUDA scanner, and submits
shares/candidates back to the pool-server. It never reads private keys, signs
transactions, or broadcasts transactions.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from hash_pool_miner import (
    close_worker,
    complete_lease,
    job_url,
    parse_checked,
    parse_rate,
    parse_record,
    read_json,
    register_worker,
    sanitized_response,
    lease_nonce_count,
    submit_record,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CUDA_BIN = ROOT / ("hash_gpu_cuda.exe" if os.name == "nt" else "hash_gpu_cuda")
DEFAULT_POOL_URL = os.environ.get("SYNTH_MINER_POOL_URL", "https://synth-miner.vercel.app/api/pool")
DEFAULT_CUDA_BATCH = 33_554_432
DEFAULT_CUDA_ITERS = 8
DEFAULT_CUDA_GROUP = 256
DEFAULT_CUDA_STREAMS = 2


def run_cuda(args: argparse.Namespace, job: dict[str, object], start: int) -> tuple[int, str]:
    cmd = [
        str(args.cuda_bin),
        "--device",
        str(args.device),
        "--challenge",
        str(job["challenge"]),
        "--target",
        str(job["network_target"]),
        "--share-target",
        str(job["share_target"]),
        "--start",
        str(start),
        "--stride",
        str(job["nonce_stride"]),
        "--seconds",
        str(args.slice_seconds),
        "--batch",
        str(args.batch),
        "--iters",
        str(args.iters),
        "--group",
        str(args.group),
        "--streams",
        str(args.streams),
        "--kernel",
        args.kernel,
    ]
    total = lease_nonce_count(job)
    if total is not None:
        cmd.extend(["--total", str(total)])
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hash_gpu_cuda against a local HASH256 pool-server.")
    parser.add_argument("--pool-url", default=DEFAULT_POOL_URL)
    parser.add_argument("--worker-id", type=int, help="Deprecated/local hint only; pool-server assigns the active worker id.")
    parser.add_argument("--worker-name", default=socket.gethostname())
    parser.add_argument("--payout-address", help="Optional EVM payout address stored for dashboard/search only.")
    parser.add_argument("--miner-token", default=os.environ.get("HASH_POOL_MINER_TOKEN"), help="Miner API token for remote pool-server access.")
    parser.add_argument("--miner-version", default="hash-cuda-pool-miner/0.1")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--device-name", default="NVIDIA CUDA")
    parser.add_argument("--device-id", default="cuda0")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--slice-seconds", type=float, default=30.0)
    parser.add_argument("--cuda-bin", type=Path, default=DEFAULT_CUDA_BIN)
    parser.add_argument("--kernel", default="cuda-v2")
    parser.add_argument("--batch", type=int, default=DEFAULT_CUDA_BATCH)
    parser.add_argument("--iters", type=int, default=DEFAULT_CUDA_ITERS)
    parser.add_argument("--group", type=int, default=DEFAULT_CUDA_GROUP)
    parser.add_argument("--streams", type=int, default=DEFAULT_CUDA_STREAMS)
    parser.add_argument("--rounds", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--debug-output", action="store_true", help="Print raw miner/API output including nonce/digest/calldata for local debugging only.")
    args = parser.parse_args()

    if args.worker_id is not None and args.worker_id < 0:
        raise SystemExit("--worker-id must be >= 0")
    if args.device < 0:
        raise SystemExit("--device must be >= 0")
    if args.slice_seconds <= 0:
        raise SystemExit("--slice-seconds must be > 0")
    if args.batch <= 0 or args.iters <= 0 or args.group <= 0 or args.streams <= 0:
        raise SystemExit("--batch, --iters, --group, and --streams must be positive")
    if not args.cuda_bin.exists():
        raise SystemExit(f"missing --cuda-bin {args.cuda_bin}")

    last_job_id: int | None = None
    next_start: int | None = None
    rounds_done = 0
    accepted_shares = 0
    accepted_candidates = 0

    register_worker(args)
    args.submit_sequence = 0

    print("HASH CUDA pool miner wrapper")
    print("  safety: no private keys, no transaction signing, no broadcasting")
    print(f"  pool:   {args.pool_url}")
    print(f"  worker: id={args.worker_id} name={args.worker_name} session=issued")
    print(f"  device: {args.device_id} {args.device_name}")
    sys.stdout.flush()

    try:
        while args.rounds == 0 or rounds_done < args.rounds:
            job = read_json(job_url(args), args.miner_token, args.worker_session)
            if job.get("status") != "ok":
                raise RuntimeError(f"pool rejected job request: {job}")
            job_id = int(job["job_id"])
            if job_id != last_job_id or next_start is None:
                lease = job.get("lease")
                next_start = int(lease["start_nonce"]) if isinstance(lease, dict) and "start_nonce" in lease else int(job["nonce_start"])
                last_job_id = job_id
            elif isinstance(job.get("lease"), dict) and "start_nonce" in job["lease"]:
                next_start = int(job["lease"]["start_nonce"])

            rc, output = run_cuda(args, job, next_start)
            checked = parse_checked(output)
            rate = parse_rate(output)
            record = parse_record(output)
            if args.debug_output:
                print(output, end="")
            elif record is not None:
                print(f"miner_record: {record['kind']} checked={checked} rate_hps={rate}")
            else:
                print(f"miner_record: none checked={checked} rate_hps={rate}")

            if checked is not None:
                next_start += checked * int(job["nonce_stride"])
            elif record is not None:
                next_start = max(next_start + int(job["nonce_stride"]), int(record["nonce"]) + int(job["nonce_stride"]))
            else:
                next_start += args.batch * args.streams * int(job["nonce_stride"])

            if rc not in (0, 1):
                raise RuntimeError("CUDA runner returned an error")
            if record is not None:
                response = submit_record(args, job, record, checked, rate)
                if args.debug_output:
                    print(f"pool_{record['kind'].lower()}_response: {json.dumps(response, sort_keys=True)}")
                else:
                    print(f"pool_{record['kind'].lower()}_response: {json.dumps(sanitized_response(record['kind'], response), sort_keys=True)}")
                if response.get("status") == "accepted" and record["kind"] == "SHARE":
                    accepted_shares += 1
                if response.get("status") == "accepted" and record["kind"] == "FOUND":
                    accepted_candidates += 1
            complete_lease(args, job, checked, rate)

            rounds_done += 1
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        close_worker(args)

    print(f"summary: rounds={rounds_done} accepted_shares={accepted_shares} accepted_full_proofs={accepted_candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
