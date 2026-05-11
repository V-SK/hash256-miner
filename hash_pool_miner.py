#!/usr/bin/env python3
"""Metal stdout wrapper for the local HASH256 pool-server.

The wrapper only fetches pool jobs, runs the local Metal scanner, and submits
shares/candidates back to the pool-server. It never reads private keys, signs
transactions, or broadcasts transactions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_METAL_BIN = ROOT / "hash_gpu_metal"
DEFAULT_POOL_URL = os.environ.get("SYNTH_MINER_POOL_URL", "https://synth-miner.vercel.app/api/pool")
CHECKED_RE = re.compile(r"checked:\s+([0-9,]+)")
RATE_RE = re.compile(r"rate:\s+([0-9,]+)\s+H/s")
RECORD_RE = re.compile(r"^(FOUND|SHARE)$", re.MULTILINE)
NONCE_RE = re.compile(r"nonce:\s+(\d+)")
DIGEST_RE = re.compile(r"digest:\s+(0x[0-9a-fA-F]{64})")
CALLDATA_RE = re.compile(r"calldata:\s+(0x[0-9a-fA-F]+)")


def auth_headers(token: str | None = None, worker_session: str | None = None) -> dict[str, str]:
    headers = {"user-agent": "hash-pool-miner/0.1"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    if worker_session:
        headers["x-worker-session"] = worker_session
    return headers


def read_json(url: str, token: str | None = None, worker_session: str | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=auth_headers(token, worker_session))
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            data = json.loads(res.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"pool HTTP {exc.code}: {body}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("pool returned non-object JSON")
    return data


def post_json(
    url: str,
    payload: dict[str, Any],
    token: str | None = None,
    worker_session: str | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload, sort_keys=True).encode()
    headers = auth_headers(token, worker_session)
    headers["content-type"] = "application/json"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            data = json.loads(res.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"pool HTTP {exc.code}: {body}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("pool returned non-object JSON")
    return data


def parse_checked(output: str) -> int | None:
    match = CHECKED_RE.search(output)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_rate(output: str) -> int | None:
    match = RATE_RE.search(output)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_record(output: str) -> dict[str, Any] | None:
    record_match = RECORD_RE.search(output)
    if not record_match:
        return None
    nonce_match = NONCE_RE.search(output)
    digest_match = DIGEST_RE.search(output)
    calldata_match = CALLDATA_RE.search(output)
    if not nonce_match or not digest_match or not calldata_match:
        raise RuntimeError("miner emitted a record without nonce/digest/calldata")
    return {
        "kind": record_match.group(1),
        "nonce": int(nonce_match.group(1)),
        "digest": digest_match.group(1).lower(),
        "calldata": calldata_match.group(1).lower(),
    }


def job_url(args: argparse.Namespace) -> str:
    params = {
        "worker_id": str(args.worker_id),
        "worker_name": args.worker_name,
    }
    for key in ["payout_address", "miner_version", "backend", "device_name", "device_id"]:
        value = getattr(args, key)
        if value:
            params[key] = value
    return args.pool_url.rstrip("/") + "/job?" + urllib.parse.urlencode(params)


def worker_metadata(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"worker_name": args.worker_name}
    for key in ["payout_address", "miner_version", "backend", "device_name", "device_id"]:
        value = getattr(args, key)
        if value:
            payload[key] = value
    return payload


def register_worker(args: argparse.Namespace) -> dict[str, Any]:
    response = post_json(args.pool_url.rstrip("/") + "/worker/register", worker_metadata(args), args.miner_token)
    if response.get("status") != "ok":
        raise RuntimeError(f"pool rejected worker registration: {response}")
    if "worker_id" not in response or "worker_session" not in response:
        raise RuntimeError(f"pool registration missing worker credentials: {response}")
    args.worker_id = int(response["worker_id"])
    args.worker_session = str(response["worker_session"])
    return response


def close_worker(args: argparse.Namespace) -> None:
    worker_session = getattr(args, "worker_session", None)
    if not worker_session:
        return
    try:
        post_json(args.pool_url.rstrip("/") + "/worker/close", {}, args.miner_token, worker_session)
    except Exception as exc:
        print(f"worker_session_close_warning: {exc}", file=sys.stderr)


def complete_lease(args: argparse.Namespace, job: dict[str, Any], checked: int | None, rate: int | None) -> None:
    lease = job.get("lease")
    if not isinstance(lease, dict) or "lease_id" not in lease:
        return
    payload: dict[str, Any] = {
        "lease_id": lease["lease_id"],
        "checked_count": checked or 0,
    }
    if rate is not None:
        payload["rate_hps"] = rate
    try:
        post_json(args.pool_url.rstrip("/") + "/lease/complete", payload, args.miner_token, args.worker_session)
    except Exception as exc:
        print(f"lease_complete_warning: {exc}", file=sys.stderr)


def lease_nonce_count(job: dict[str, Any]) -> int | None:
    lease = job.get("lease")
    if not isinstance(lease, dict) or "nonce_count" not in lease:
        return None
    try:
        count = int(lease["nonce_count"])
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def run_metal(args: argparse.Namespace, job: dict[str, Any], start: int) -> tuple[int, str]:
    cmd = [
        str(args.metal_bin),
        "--kernel",
        args.kernel,
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
        "--inflight",
        str(args.inflight),
    ]
    total = lease_nonce_count(job)
    if total is not None:
        cmd.extend(["--total", str(total)])
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return proc.returncode, proc.stdout + proc.stderr


def submit_record(args: argparse.Namespace, job: dict[str, Any], record: dict[str, Any], checked: int | None, rate: int | None) -> dict[str, Any]:
    submit_sequence = int(getattr(args, "submit_sequence", 0)) + 1
    args.submit_sequence = submit_sequence
    payload: dict[str, Any] = {
        "worker_id": args.worker_id,
        "worker_name": args.worker_name,
        "job_id": job["job_id"],
        "nonce": str(record["nonce"]),
        "digest": record["digest"],
        "submit_sequence": submit_sequence,
        "kernel": args.kernel,
    }
    lease = job.get("lease")
    if isinstance(lease, dict) and "lease_id" in lease:
        payload["lease_id"] = lease["lease_id"]
    for key in ["payout_address", "miner_version", "backend", "device_name", "device_id"]:
        value = getattr(args, key)
        if value:
            payload[key] = value
    if checked is not None:
        payload["checked_count"] = checked
    if rate is not None:
        payload["rate_hps"] = rate

    if record["kind"] == "FOUND":
        payload["calldata"] = record["calldata"]
        return post_json(args.pool_url.rstrip("/") + "/candidate", payload, args.miner_token, args.worker_session)
    if record["kind"] == "SHARE":
        return post_json(args.pool_url.rstrip("/") + "/share", payload, args.miner_token, args.worker_session)
    raise RuntimeError(f"unknown miner record kind: {record['kind']}")


def sanitized_response(kind: str, response: dict[str, Any]) -> dict[str, Any]:
    out = {
        "status": response.get("status"),
        "counted": response.get("counted"),
        "job_id": response.get("job_id"),
        "worker_id": response.get("worker_id"),
        "submit_sequence": response.get("submit_sequence"),
    }
    if kind == "SHARE":
        out["share_id"] = response.get("share_id")
    if kind == "FOUND":
        share = response.get("share")
        if isinstance(share, dict):
            out["share"] = {
                "status": share.get("status"),
                "counted": share.get("counted"),
                "share_id": share.get("share_id"),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hash_gpu_metal against a local HASH256 pool-server.")
    parser.add_argument("--pool-url", default=DEFAULT_POOL_URL)
    parser.add_argument("--worker-id", type=int, help="Deprecated/local hint only; pool-server assigns the active worker id.")
    parser.add_argument("--worker-name", default=socket.gethostname())
    parser.add_argument("--payout-address", help="Optional EVM payout address stored for dashboard/search only.")
    parser.add_argument("--miner-token", default=os.environ.get("HASH_POOL_MINER_TOKEN"), help="Miner API token for remote pool-server access.")
    parser.add_argument("--miner-version", default="hash-pool-miner/0.2")
    parser.add_argument("--backend", default="metal")
    parser.add_argument("--device-name", default="Apple Metal")
    parser.add_argument("--device-id", default="metal0")
    parser.add_argument("--slice-seconds", type=float, default=120.0)
    parser.add_argument("--metal-bin", type=Path, default=DEFAULT_METAL_BIN)
    parser.add_argument("--kernel", choices=["compact", "scalar", "u64", "u32"], default="compact")
    parser.add_argument("--batch", type=int, default=1 << 25)
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--group", type=int, default=256)
    parser.add_argument("--inflight", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--debug-output", action="store_true", help="Print raw miner/API output including nonce/digest/calldata for local debugging only.")
    args = parser.parse_args()

    if args.worker_id is not None and args.worker_id < 0:
        raise SystemExit("--worker-id must be >= 0")
    if args.slice_seconds <= 0:
        raise SystemExit("--slice-seconds must be > 0")
    if args.batch <= 0 or args.iters <= 0 or args.group <= 0 or args.inflight <= 0:
        raise SystemExit("--batch, --iters, --group, and --inflight must be positive")
    if not args.metal_bin.exists():
        raise SystemExit(f"missing --metal-bin {args.metal_bin}")

    last_job_id: int | None = None
    next_start: int | None = None
    rounds_done = 0
    accepted_shares = 0
    accepted_candidates = 0

    register_worker(args)
    args.submit_sequence = 0

    print("HASH pool miner wrapper")
    print("  safety: no private keys, no transaction signing, no broadcasting")
    print(f"  pool:   {args.pool_url}")
    print(f"  worker: id={args.worker_id} name={args.worker_name} session=issued")
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

            rc, output = run_metal(args, job, next_start)
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
                next_start += args.batch * int(job["nonce_stride"])

            if rc == 2:
                raise RuntimeError("Metal runner returned an error")
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
