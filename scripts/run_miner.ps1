param(
  [Parameter(Mandatory = $true)]
  [string]$Address,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python) -and (Test-Path ".\.venv\bin\python.exe")) {
  $Python = ".\.venv\bin\python.exe"
}
if (-not (Test-Path $Python) -and (Test-Path ".\.venv\bin\python")) {
  $Python = ".\.venv\bin\python"
}
if (-not (Test-Path $Python)) {
  $Python = "python"
}

& $Python .\hash_miner.py --address $Address @ExtraArgs
