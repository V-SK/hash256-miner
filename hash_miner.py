#!/usr/bin/env python3
"""Professional launcher for HASH256 pool mining.

This entrypoint is for miners, not submitters. It validates a payout address,
detects the available GPU backend, and launches the existing pool miner wrapper.
It never reads private keys, signs transactions, broadcasts transactions, or
deploys contracts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from eth_utils import is_address, to_checksum_address


ROOT = Path(__file__).resolve().parent
DEFAULT_METAL_BIN = ROOT / "hash_gpu_metal"
CUDA_BIN_NAMES = ["hash_gpu_cuda", "hash_gpu_cuda.exe"]
MINER_VERSION = "hash-miner/0.1"
POOL_FEE_PERCENT = 2
DEFAULT_POOL_URL = os.environ.get("SYNTH_MINER_POOL_URL", "https://synth-miner.vercel.app/api/pool")
DEFAULT_METAL_BATCH = 1 << 23
DEFAULT_METAL_ITERS = 8
DEFAULT_METAL_GROUP = 64
DEFAULT_CUDA_BATCH = 33_554_432
DEFAULT_CUDA_ITERS = 8
DEFAULT_CUDA_GROUP = 256
DEFAULT_CUDA_STREAMS = 2


@dataclass(frozen=True)
class CudaGpu:
    index: int
    name: str
    memory_mb: int | None = None
    driver_version: str | None = None


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    hint: str | None = None


@dataclass(frozen=True)
class DoctorResult:
    status: str
    selected_backend: str | None
    checks: list[DoctorCheck]


def find_cuda_bin(override: Path | None = None) -> Path | None:
    if override is not None:
        return override if override.exists() else None
    for name in CUDA_BIN_NAMES:
        path = ROOT / name
        if path.exists():
            return path
    return None


def detect_cuda_gpus() -> list[CudaGpu]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []

    gpus: list[CudaGpu] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        name = parts[1] or f"CUDA GPU {index}"
        memory_mb = None
        if len(parts) > 2:
            raw_memory = parts[2].lower().replace("mib", "").replace("mb", "").strip()
            try:
                memory_mb = int(raw_memory)
            except ValueError:
                memory_mb = None
        driver_version = parts[3] if len(parts) > 3 and parts[3] else None
        gpus.append(CudaGpu(index=index, name=name, memory_mb=memory_mb, driver_version=driver_version))
    return gpus


def select_backend(requested: str, metal_bin: Path, cuda_bin: Path | None, cuda_gpus: list[CudaGpu]) -> str:
    system = platform.system().lower()
    if requested != "auto":
        return requested
    if system == "darwin" and metal_bin.exists():
        return "metal"
    if cuda_bin is not None and cuda_gpus:
        return "cuda"
    raise SystemExit(
        "no supported GPU backend detected; use the bundled Metal/CUDA miner release and install only the NVIDIA driver where needed"
    )


def is_executable(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def chmod_executable(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        return False
    return is_executable(path)


def find_libcuda_linux() -> str | None:
    for pattern in [
        "/usr/lib*/libcuda.so*",
        "/usr/lib/*/libcuda.so*",
        "/usr/local/cuda/lib*/libcuda.so*",
    ]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    if shutil.which("ldconfig"):
        try:
            proc = subprocess.run(["ldconfig", "-p"], text=True, capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if "libcuda.so" in line:
                    return line.strip()
    return None


def check_pool(pool_url: str) -> DoctorCheck:
    url = pool_url.rstrip("/") + "/stats_public"
    try:
        req = urllib.request.Request(url, headers={"user-agent": "hash-miner-doctor/0.1"})
        with urllib.request.urlopen(req, timeout=5) as res:
            body = res.read(1 << 20)
            status_code = res.status
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return DoctorCheck("pool", "fail", f"{url} returned non-JSON HTTP {status_code}", "Check pool URL and reverse proxy.")
        if status_code == 200 and isinstance(data, dict) and data.get("status") == "ok":
            return DoctorCheck("pool", "ok", f"Pool reachable: {url}")
        return DoctorCheck("pool", "fail", f"{url} returned HTTP {status_code} status={data.get('status') if isinstance(data, dict) else 'unknown'}")
    except urllib.error.URLError as exc:
        return DoctorCheck("pool", "fail", f"Pool unreachable: {url}; reason={exc.reason}", "Start pool_server.py locally or check network/firewall.")
    except TimeoutError:
        return DoctorCheck("pool", "fail", f"Pool check timed out: {url}", "Check network/firewall.")
    except OSError as exc:
        return DoctorCheck("pool", "fail", f"Pool unreachable: {url}; reason={exc}", "Check pool URL.")


def choose_doctor_backend(requested: str, system: str, metal_ready: bool, cuda_ready: bool) -> str | None:
    if requested == "metal":
        return "metal"
    if requested == "cuda":
        return "cuda"
    if system == "darwin" and metal_ready:
        return "metal"
    if cuda_ready:
        return "cuda"
    return None


def status_from_checks(checks: list[DoctorCheck], selected_backend: str | None) -> str:
    blocker_names = {"address", "pool", "backend"}
    if selected_backend == "metal":
        blocker_names.add("metal_binary")
    if selected_backend == "cuda":
        blocker_names.update({"nvidia_smi", "cuda_binary", "libcuda"})
    for check in checks:
        if check.name in blocker_names and check.status == "fail":
            return "blocked"
    if selected_backend is None:
        return "blocked"
    if any(check.status == "fail" for check in checks):
        return "degraded"
    if any(check.status == "warn" for check in checks):
        return "degraded"
    return "ready"


def run_doctor(args: argparse.Namespace) -> DoctorResult:
    checks: list[DoctorCheck] = []
    system = platform.system()
    machine = platform.machine()
    checks.append(DoctorCheck("os", "ok", f"{system or 'unknown'} {machine or 'unknown'}"))
    checks.append(DoctorCheck("paths", "ok", f"cwd={Path.cwd()} launcher={Path(__file__).resolve()}"))

    if not args.address:
        checks.append(
            DoctorCheck(
                "address",
                "warn",
                "address not provided; required before mining",
                "Pass --address 0x... when starting normal mining.",
            )
        )
    elif is_address(args.address):
        checks.append(DoctorCheck("address", "ok", "Address valid"))
    else:
        checks.append(DoctorCheck("address", "fail", "Invalid EVM address", "Use a checksummed or lowercase 0x-prefixed ETH address."))

    checks.append(check_pool(args.pool_url))
    if args.miner_token:
        checks.append(DoctorCheck("miner_token", "ok", "Miner token configured"))
    else:
        checks.append(DoctorCheck("miner_token", "warn", "Miner token not configured", "The public SYNTH pool does not require a miner token."))

    system_l = system.lower()
    metal_ready = False
    cuda_ready = False
    cuda_bin = find_cuda_bin(args.cuda_bin)
    cuda_gpus = detect_cuda_gpus()

    if system_l == "darwin":
        cpu_kind = "Apple Silicon" if machine in {"arm64", "aarch64"} else "Intel"
        checks.append(DoctorCheck("mac_cpu", "ok", cpu_kind))
        if args.metal_bin.exists():
            if is_executable(args.metal_bin):
                metal_ready = True
                checks.append(DoctorCheck("metal_binary", "ok", f"Metal miner executable: {args.metal_bin}"))
            elif args.install_missing and chmod_executable(args.metal_bin):
                metal_ready = True
                checks.append(DoctorCheck("metal_binary", "ok", f"Fixed executable bit: {args.metal_bin}"))
            else:
                checks.append(DoctorCheck("metal_binary", "fail", f"Metal miner is not executable: {args.metal_bin}", "Run with --install-missing to chmod +x."))
        else:
            checks.append(DoctorCheck("metal_binary", "fail", f"Metal miner missing: {args.metal_bin}", "Download the macOS arm64 miner release. TODO: release URL."))

    if args.backend == "metal" and system_l != "darwin":
        checks.append(DoctorCheck("metal_binary", "fail", "Metal backend requires macOS", "Use the CUDA miner release on NVIDIA Windows/Linux hosts."))

    if args.backend == "cuda" and system_l not in {"windows", "linux"}:
        if cuda_gpus:
            detail = "; ".join(
                f"gpu{gpu.index} {gpu.name} vram={gpu.memory_mb or 'unknown'}MiB driver={gpu.driver_version or 'unknown'}"
                for gpu in cuda_gpus
            )
            checks.append(DoctorCheck("nvidia_smi", "ok", detail))
        else:
            checks.append(DoctorCheck("nvidia_smi", "fail", "No NVIDIA GPU detected by nvidia-smi", "Use auto/metal on Apple Silicon, or run the CUDA release on Windows/Linux NVIDIA hosts."))
        if cuda_bin is not None:
            checks.append(DoctorCheck("cuda_binary", "ok", f"CUDA miner binary found: {cuda_bin}"))
        else:
            checks.append(DoctorCheck("cuda_binary", "fail", "CUDA miner binary missing", "Download the CUDA miner release for Windows/Linux. TODO: release URL."))

    if system_l == "windows":
        if cuda_gpus:
            cuda_ready = cuda_bin is not None
            detail = "; ".join(
                f"gpu{gpu.index} {gpu.name} vram={gpu.memory_mb or 'unknown'}MiB driver={gpu.driver_version or 'unknown'}"
                for gpu in cuda_gpus
            )
            checks.append(DoctorCheck("nvidia_smi", "ok", detail))
        else:
            checks.append(DoctorCheck("nvidia_smi", "fail", "nvidia-smi not found or no NVIDIA GPU detected", "Install/update the NVIDIA driver. Do not install CUDA Toolkit for normal mining."))
        if cuda_bin is not None:
            checks.append(DoctorCheck("cuda_binary", "ok", f"CUDA miner binary found: {cuda_bin}"))
        else:
            checks.append(DoctorCheck("cuda_binary", "fail", "CUDA miner binary missing: hash_gpu_cuda.exe", "Download the Windows CUDA miner release. TODO: release URL."))

    if system_l == "linux":
        if cuda_gpus:
            detail = "; ".join(
                f"gpu{gpu.index} {gpu.name} vram={gpu.memory_mb or 'unknown'}MiB driver={gpu.driver_version or 'unknown'}"
                for gpu in cuda_gpus
            )
            checks.append(DoctorCheck("nvidia_smi", "ok", detail))
        else:
            checks.append(DoctorCheck("nvidia_smi", "fail", "nvidia-smi not found or no NVIDIA GPU detected", "Install/update the NVIDIA driver with explicit user approval."))
        libcuda = find_libcuda_linux()
        if libcuda:
            checks.append(DoctorCheck("libcuda", "ok", f"libcuda found: {libcuda}"))
        else:
            checks.append(DoctorCheck("libcuda", "fail", "libcuda.so not found", "Install NVIDIA driver; do not install CUDA Toolkit for normal mining unless developing."))
        if cuda_bin is not None:
            if is_executable(cuda_bin):
                checks.append(DoctorCheck("cuda_binary", "ok", f"CUDA miner executable: {cuda_bin}"))
                cuda_ready = bool(cuda_gpus)
            elif args.install_missing and chmod_executable(cuda_bin):
                checks.append(DoctorCheck("cuda_binary", "ok", f"Fixed executable bit: {cuda_bin}"))
                cuda_ready = bool(cuda_gpus)
            else:
                checks.append(DoctorCheck("cuda_binary", "fail", f"CUDA miner is not executable: {cuda_bin}", "Run with --install-missing to chmod +x."))
        else:
            checks.append(DoctorCheck("cuda_binary", "fail", "CUDA miner binary missing: hash_gpu_cuda", "Download the Linux CUDA miner release. TODO: release URL."))

    # On non-CUDA macOS this is intentionally not a failure; normal users should
    # not install CUDA Toolkit, nvcc, Xcode, or VS Build Tools for mining.
    selected_backend = choose_doctor_backend(args.backend, system_l, metal_ready, cuda_ready)
    if selected_backend is None:
        checks.append(
            DoctorCheck(
                "backend",
                "fail",
                "No runnable GPU backend selected",
                "Use the bundled Metal miner on macOS or a release CUDA miner with NVIDIA driver on Windows/Linux.",
            )
        )
    else:
        checks.append(DoctorCheck("backend", "ok", f"Backend: {selected_backend}"))

    return DoctorResult(status=status_from_checks(checks, selected_backend), selected_backend=selected_backend, checks=checks)


def doctor_as_dict(result: DoctorResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "selected_backend": result.selected_backend,
        "checks": [
            {
                "name": check.name,
                "status": "ok" if check.status == "ok" else ("fail" if check.status == "fail" else "warn"),
                "detail": check.detail,
                **({"hint": check.hint} if check.hint else {}),
            }
            for check in result.checks
        ],
    }


def print_doctor(result: DoctorResult) -> None:
    for check in result.checks:
        label = "OK" if check.status == "ok" else ("FAIL" if check.status == "fail" else "WARN")
        print(f"[{label}] {check.name}: {check.detail}")
        if check.hint:
            print(f"[HINT] {check.hint}")
    print(f"status: {result.status}")
    print(f"selected_backend: {result.selected_backend or 'none'}")


def format_command(cmd: list[str]) -> str:
    redacted: list[str] = []
    redact_next = False
    for part in cmd:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(part)
        if part == "--miner-token":
            redact_next = True
    return " ".join(redacted)


def parse_gpu_selection(value: str, available: list[CudaGpu]) -> list[CudaGpu]:
    if value == "all":
        return available
    wanted: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            wanted.add(int(part))
        except ValueError as exc:
            raise SystemExit("--gpus must be 'all' or a comma-separated list of GPU indexes") from exc
    selected = [gpu for gpu in available if gpu.index in wanted]
    missing = wanted - {gpu.index for gpu in selected}
    if missing:
        raise SystemExit(f"requested CUDA GPU indexes not detected: {sorted(missing)}")
    return selected


def run_metal(args: argparse.Namespace, address: str) -> int:
    if args.gpus not in {"all", "0", "metal", "metal0"}:
        raise SystemExit("Metal MVP exposes one logical device; use --gpus all")
    worker_name = f"{args.worker_prefix}-metal"
    batch = args.batch if args.batch is not None else DEFAULT_METAL_BATCH
    iters = args.iters if args.iters is not None else DEFAULT_METAL_ITERS
    group = args.group if args.group is not None else DEFAULT_METAL_GROUP
    cmd = [
        sys.executable,
        str(ROOT / "hash_pool_miner.py"),
        "--pool-url",
        args.pool_url,
        "--worker-id",
        str(args.worker_id_base),
        "--worker-name",
        worker_name,
        "--payout-address",
        address,
        "--miner-version",
        MINER_VERSION,
        "--backend",
        "metal",
        "--device-name",
        "Apple Metal",
        "--device-id",
        "metal0",
        "--slice-seconds",
        str(args.slice_seconds),
        "--metal-bin",
        str(args.metal_bin),
        "--kernel",
        args.kernel,
        "--batch",
        str(batch),
        "--iters",
        str(iters),
        "--group",
        str(group),
        "--inflight",
        str(args.inflight),
        "--rounds",
        str(args.rounds),
    ]
    if args.miner_token:
        cmd.extend(["--miner-token", args.miner_token])

    print("HASH miner launcher")
    print("  safety: no private keys, no transaction signing, no broadcasting")
    print(f"  payout_address: {address}")
    print(f"  pool_fee:       {POOL_FEE_PERCENT}%")
    print(f"  backend:        metal")
    print(f"  worker:         {worker_name} preferred_id={args.worker_id_base} (pool assigns active id)")
    print(f"  command:        {format_command(cmd)}")
    sys.stdout.flush()
    return subprocess.run(cmd).returncode


def run_cuda(args: argparse.Namespace, address: str, cuda_bin: Path, cuda_gpus: list[CudaGpu]) -> int:
    selected = parse_gpu_selection(args.gpus, cuda_gpus)
    if not selected:
        raise SystemExit("no CUDA GPUs selected")
    batch = args.batch if args.batch is not None else DEFAULT_CUDA_BATCH
    iters = args.iters if args.iters is not None else DEFAULT_CUDA_ITERS
    group = args.group if args.group is not None else DEFAULT_CUDA_GROUP
    streams = args.streams if args.streams is not None else DEFAULT_CUDA_STREAMS

    print("HASH miner launcher")
    print("  safety: no private keys, no transaction signing, no broadcasting")
    print(f"  payout_address: {address}")
    print(f"  pool_fee:       {POOL_FEE_PERCENT}%")
    print("  backend:        cuda")
    print(f"  cuda_binary:    {cuda_bin}")
    print(f"  cuda_params:    batch={batch} iters={iters} group={group} streams={streams}")
    print("  workers:")
    commands: list[list[str]] = []
    for offset, gpu in enumerate(selected):
        worker_id = args.worker_id_base + offset
        worker_name = f"{args.worker_prefix}-gpu{gpu.index}"
        cmd = [
            sys.executable,
            str(ROOT / "hash_cuda_pool_miner.py"),
            "--pool-url",
            args.pool_url,
            "--worker-id",
            str(worker_id),
            "--worker-name",
            worker_name,
            "--payout-address",
            address,
            "--miner-version",
            MINER_VERSION,
            "--backend",
            "cuda",
            "--device",
            str(gpu.index),
            "--device-name",
            gpu.name,
            "--device-id",
            f"cuda{gpu.index}",
            "--slice-seconds",
            str(args.slice_seconds),
            "--cuda-bin",
            str(cuda_bin),
            "--batch",
            str(batch),
            "--iters",
            str(iters),
            "--group",
            str(group),
            "--streams",
            str(streams),
            "--rounds",
            str(args.rounds),
        ]
        if args.miner_token:
            cmd.extend(["--miner-token", args.miner_token])
        commands.append(cmd)
        print(f"    - preferred_worker_id={worker_id} worker_name={worker_name} gpu={gpu.index} name={gpu.name}")
        print(f"      command: {format_command(cmd)}")
    sys.stdout.flush()

    if len(commands) == 1:
        return subprocess.run(commands[0]).returncode

    procs: list[subprocess.Popen[bytes]] = []
    try:
        for cmd in commands:
            procs.append(subprocess.Popen(cmd))
        return_codes = [proc.wait() for proc in procs]
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
        print("\nstopping CUDA workers")
        return 130
    return 0 if all(code == 0 for code in return_codes) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="HASH256 professional pool miner launcher.")
    parser.add_argument("--address", help="ETH payout address. Never provide a private key.")
    parser.add_argument("--pool-url", default=DEFAULT_POOL_URL)
    parser.add_argument("--worker-prefix", default=socket.gethostname())
    parser.add_argument("--backend", choices=["auto", "metal", "cuda"], default="auto")
    parser.add_argument("--gpus", default="all", help="'all' or comma-separated CUDA GPU indexes. Metal MVP uses one device.")
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--slice-seconds", type=float, default=30.0)
    parser.add_argument("--miner-token", default=os.environ.get("HASH_POOL_MINER_TOKEN"))
    parser.add_argument("--doctor", action="store_true", help="Check runtime dependencies and exit without mining.")
    parser.add_argument("--check-only", action="store_true", help="Alias for --doctor.")
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Only fix small safe local issues such as chmod +x; never installs drivers or toolchains.",
    )
    parser.add_argument("--json", action="store_true", help="Emit doctor result as JSON for GUI/front-end use.")
    parser.add_argument("--metal-bin", type=Path, default=DEFAULT_METAL_BIN)
    parser.add_argument("--cuda-bin", type=Path, help="Path to hash_gpu_cuda or hash_gpu_cuda.exe.")
    parser.add_argument("--worker-id-base", type=int, default=0, help="Deprecated local worker-id hint base; pool-server assigns active IDs.")
    parser.add_argument("--kernel", choices=["compact", "scalar", "u64", "u32"], default="compact")
    parser.add_argument("--batch", type=int, help="Override backend batch size. CUDA default is tuned separately from Metal.")
    parser.add_argument("--iters", type=int, help="Override backend iterations per thread.")
    parser.add_argument("--group", type=int, help="Override backend thread group/block size.")
    parser.add_argument("--inflight", type=int, default=2)
    parser.add_argument("--streams", type=int, help="CUDA async stream count. Default 2.")
    args = parser.parse_args()

    doctor_mode = args.doctor or args.check_only
    if not doctor_mode and not args.address:
        raise SystemExit("missing --address")
    if args.worker_id_base < 0:
        raise SystemExit("--worker-id-base must be >= 0")
    if args.rounds < 0:
        raise SystemExit("--rounds must be >= 0")
    if args.slice_seconds <= 0:
        raise SystemExit("--slice-seconds must be > 0")
    if args.batch is not None and args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.iters is not None and args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.group is not None and args.group <= 0:
        raise SystemExit("--group must be positive")
    if args.inflight <= 0:
        raise SystemExit("--inflight must be positive")
    if args.streams is not None and args.streams <= 0:
        raise SystemExit("--streams must be positive")

    doctor = run_doctor(args)
    if doctor_mode:
        if args.json:
            print(json.dumps(doctor_as_dict(doctor), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print_doctor(doctor)
        return 2 if doctor.status == "blocked" else 0

    if doctor.status == "blocked":
        if args.json:
            print(json.dumps(doctor_as_dict(doctor), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print_doctor(doctor)
        return 2
    if doctor.status == "degraded":
        print("doctor status: degraded; continuing with runnable backend")
        for check in doctor.checks:
            if check.status != "ok":
                label = "FAIL" if check.status == "fail" else "WARN"
                print(f"[{label}] {check.name}: {check.detail}")
                if check.hint:
                    print(f"[HINT] {check.hint}")

    address = to_checksum_address(args.address)
    cuda_gpus = detect_cuda_gpus()
    cuda_bin = find_cuda_bin(args.cuda_bin)
    backend = doctor.selected_backend

    if backend == "metal":
        return run_metal(args, address)

    if backend == "cuda":
        if cuda_bin is None:
            raise SystemExit("missing CUDA runner: download the release hash_gpu_cuda binary for your OS")
        if not cuda_gpus:
            raise SystemExit("no NVIDIA GPUs detected by nvidia-smi")
        return run_cuda(args, address, cuda_bin, cuda_gpus)

    raise SystemExit(f"unsupported backend: {backend}")


if __name__ == "__main__":
    raise SystemExit(main())
