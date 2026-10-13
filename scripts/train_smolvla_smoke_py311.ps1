$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TrainScript = Join-Path $PSScriptRoot "train_smolvla_py311_8gb.ps1"

& $TrainScript -Steps 10 -SaveFreq 10 -RunName "smolvla_lora_r8_smoke"

if ($LASTEXITCODE -ne 0) {
    throw "SmolVLA smoke training failed with exit code $LASTEXITCODE"
}

Write-Host "Smoke training passed."
