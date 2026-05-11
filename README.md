# SYNTH Miner

Official public miner client for the HASH256 pool.

This repository contains the miner-side code only:

- one-command launcher: `hash_miner.py`
- Apple Metal runner for macOS Apple Silicon
- CUDA runner and wrapper for NVIDIA Windows/Linux hosts
- dependency checks, worker registration, nonce leases, share submission, and candidate submission

It does not contain pool-server code, submitter keys, admin APIs, settlement jobs, deployment config, or private scheduling strategy.

## Quick Start

### macOS Apple Silicon

```bash
git clone https://github.com/V-SK/hash256-miner.git
cd hash256-miner
./scripts/install_macos.sh
./scripts/run_miner.sh 0xYourPayoutAddress
```

### Windows NVIDIA

Open PowerShell in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
.\scripts\run_miner.ps1 -Address 0xYourPayoutAddress
```

The current bundled Windows CUDA binary is an initial NVIDIA build. If your GPU generation is not supported by the bundled binary, install CUDA Toolkit and rebuild from `hash_gpu_cuda.cu`.

### Linux NVIDIA

```bash
git clone https://github.com/V-SK/hash256-miner.git
cd hash256-miner
./scripts/install_linux_cuda.sh
./scripts/run_miner.sh 0xYourPayoutAddress --backend cuda
```

Linux requires a working NVIDIA driver. If no CUDA binary is bundled for your host, the install script will compile one when `nvcc` is available.
See [docs/LINUX.md](docs/LINUX.md) for CUDA architecture selection and multi-GPU commands.
Linux uses the native `hash_gpu_cuda` binary. If you see a command trying to run
`hash_gpu_cuda.exe` on Linux, pull the latest repo and rebuild with
`./scripts/build_cuda_linux.sh`.

Linux optimization helpers:

```bash
./scripts/build_cuda_linux.sh
./scripts/tune_linux_nvidia.sh
BENCH_SECONDS=30 ./scripts/benchmark_linux_cuda.sh
```

## Normal Usage

```bash
python3 hash_miner.py --address 0xYourPayoutAddress
```

The default pool URL is:

```text
https://synth-miner.vercel.app/api/pool
```

Use a custom pool only if instructed:

```bash
python3 hash_miner.py --address 0xYourPayoutAddress --pool-url https://example.com/api/pool
```

## Doctor

Run a local dependency check without mining:

```bash
python3 hash_miner.py --doctor
python3 hash_miner.py --doctor --json
```

The miner checks:

- payout address format
- pool reachability
- Apple Metal binary on macOS
- `nvidia-smi`, CUDA binary, and `libcuda` on NVIDIA hosts
- selected backend

## Professional Options

```bash
python3 hash_miner.py --address 0xYourPayoutAddress \
  --backend auto \
  --gpus all \
  --slice-seconds 30 \
  --worker-prefix rig-a
```

CUDA tuning defaults are optimized for the first 3070 Ti laptop test build:

```text
batch=33554432 iters=8 group=256 streams=2
```

Override them only if you are benchmarking:

```bash
python3 hash_miner.py --address 0xYourPayoutAddress \
  --backend cuda --batch 33554432 --iters 8 --group 256 --streams 2
```

## Rewards

Pool fee: 2%.

Rewards are credited to your payout address by the pool and can be viewed or claimed on:

```text
https://synth-miner.vercel.app/
```

The miner never needs your private key. It only needs your public payout address.

## Security Model

The miner is intentionally limited:

- no private key input
- no transaction signing
- no transaction broadcasting
- no contract deployment
- server-assigned worker sessions
- server-assigned nonce leases
- server-side share and candidate verification
- duplicate and sequence protection on the pool side

Debug mode may print nonce/digest/calldata for local troubleshooting. Normal mode redacts detailed proof material.

## 中文说明

这是 SYNTH Miner 的公开矿工客户端仓库，只包含矿工侧代码，不包含矿池服务端、自动提交器、管理员 API、结算任务、私有 RPC 配置或内部调度策略。

普通用户只需要填写自己的收款地址：

```bash
python3 hash_miner.py --address 0x你的收款地址
```

矿工程序不会读取私钥、不会签名、不会广播交易。收益会进入矿池结算系统，矿工在网站上查看并领取：

```text
https://synth-miner.vercel.app/
```

## Source Availability

Copyright (c) V-SK. Source is published for miner transparency and for running against the official SYNTH Miner pool. No license is granted to operate a competing pool, redistribute modified binaries, or reuse the code commercially without written permission.
