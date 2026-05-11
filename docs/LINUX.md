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

## Build Locally

Use the CUDA arch that matches your GPU. The default is `sm_86`, which is suitable for Ampere cards such as RTX 30 series.

```bash
CUDA_ARCH=sm_86 ./scripts/build_cuda_linux.sh
```

Examples:

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
