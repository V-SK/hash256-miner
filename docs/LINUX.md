# Linux NVIDIA Mining

Linux is supported through the CUDA backend.

## Requirements

- x86_64 Linux
- NVIDIA GPU
- NVIDIA driver with `nvidia-smi`
- Python 3 with `venv`
- either a matching `hash_gpu_cuda` release binary or CUDA Toolkit with `nvcc`

The miner does not need a private key and does not sign or broadcast transactions.

## Ubuntu Quick Start

```bash
git clone https://github.com/V-SK/hash256-miner.git
cd hash256-miner
./scripts/install_linux_cuda.sh
./scripts/run_miner.sh 0xYourPayoutAddress --backend cuda
```

If the install script cannot find `hash_gpu_cuda` and cannot find `nvcc`, install CUDA Toolkit or download a Linux release binary.

Do not use the Windows `hash_gpu_cuda.exe` on Linux. Linux miners need the
native no-extension binary named `hash_gpu_cuda`; the launcher intentionally
ignores `.exe` files on Linux so a wrong release artifact fails early.
The Linux installer also fails early if only `hash_gpu_cuda.exe` is present or
if neither a native `hash_gpu_cuda` nor `nvcc` is available.

## Build Locally

Use the CUDA arch that matches your GPU. The build script auto-detects the first NVIDIA GPU when possible:

```bash
./scripts/build_cuda_linux.sh
```

Override manually when needed:

```bash
CUDA_ARCH=sm_86 ./scripts/build_cuda_linux.sh
```

```bash
CUDA_ARCH=sm_75 ./scripts/build_cuda_linux.sh   # Turing / RTX 20
CUDA_ARCH=sm_86 ./scripts/build_cuda_linux.sh   # Ampere / RTX 30
CUDA_ARCH=sm_89 ./scripts/build_cuda_linux.sh   # Ada / RTX 40
CUDA_ARCH=sm_120 ./scripts/build_cuda_linux.sh  # Blackwell / RTX 50, if supported by your CUDA Toolkit
```

Then run:

```bash
./scripts/run_miner.sh 0xYourPayoutAddress --backend cuda
```

## Linux Performance Tuning

The CUDA kernel is the same register-resident v2 kernel used by the Windows CUDA build. On Linux, the biggest wins usually come from native architecture compilation and stable GPU clocks.

Dry-run safe tuning commands:

```bash
./scripts/tune_linux_nvidia.sh
```

Apply safe tuning:

```bash
./scripts/tune_linux_nvidia.sh --apply
```

Optional power and clock controls:

```bash
./scripts/tune_linux_nvidia.sh --apply --power-limit 320
./scripts/tune_linux_nvidia.sh --apply --lock-graphics-clock 2500
./scripts/tune_linux_nvidia.sh --apply --unlock-clocks
```

Only use power/clock values that your card supports. Consumer GeForce cards may reject some clock commands; that is normal.

Benchmark a Linux host and pick the best parameters:

```bash
BENCH_SECONDS=30 ./scripts/benchmark_linux_cuda.sh
```

The script writes a CSV and prints the top configurations. Use the winning row as:

```bash
python3 hash_miner.py --address 0xYourPayoutAddress \
  --backend cuda \
  --batch 33554432 \
  --iters 8 \
  --group 256 \
  --streams 2
```

## Doctor

```bash
python3 hash_miner.py --doctor --backend cuda
```

Healthy Linux CUDA output should include:

- `nvidia_smi: OK`
- `libcuda: OK`
- `cuda_binary: OK`
- `backend: cuda`

## Multi-GPU

Run all detected GPUs:

```bash
python3 hash_miner.py --address 0xYourPayoutAddress --backend cuda --gpus all
```

Run selected GPUs:

```bash
python3 hash_miner.py --address 0xYourPayoutAddress --backend cuda --gpus 0,2
```
