param(
  [string]$Python = "py",
  [string]$CudaArch = "sm_86"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Run-Python {
  param([string[]]$PythonArgs)
  if ($Python -eq "py") {
    & py -3 @PythonArgs
  } else {
    & $Python @PythonArgs
  }
}

function Get-VenvPython {
  $candidates = @(
    ".\.venv\Scripts\python.exe",
    ".\.venv\bin\python.exe",
    ".\.venv\bin\python"
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }
  throw "venv python not found under .venv"
}

Run-Python @("-m", "venv", ".venv")
$VenvPython = Get-VenvPython
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

if (-not (Test-Path ".\hash_gpu_cuda.exe")) {
  $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  if ($nvcc) {
    & $nvcc.Source -O3 -std=c++17 -arch=$CudaArch hash_gpu_cuda.cu -o hash_gpu_cuda.exe
  } else {
    Write-Host "hash_gpu_cuda.exe is missing and nvcc.exe was not found."
    Write-Host "Install NVIDIA CUDA Toolkit or download a matching binary release."
  }
}

& $VenvPython .\hash_miner.py --doctor --backend cuda
