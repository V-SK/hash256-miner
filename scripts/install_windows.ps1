param(
  [string]$Python = "py",
  [string]$CudaArch = "sm_86"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Run-Python {
  param([string[]]$Args)
  if ($Python -eq "py") {
    & py -3 @Args
  } else {
    & $Python @Args
  }
}

Run-Python @("-m", "venv", ".venv")
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".\hash_gpu_cuda.exe")) {
  $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  if ($nvcc) {
    & $nvcc.Source -O3 -std=c++17 -arch=$CudaArch hash_gpu_cuda.cu -o hash_gpu_cuda.exe
  } else {
    Write-Host "hash_gpu_cuda.exe is missing and nvcc.exe was not found."
    Write-Host "Install NVIDIA CUDA Toolkit or download a matching binary release."
  }
}

& .\.venv\Scripts\python.exe .\hash_miner.py --doctor --backend cuda
