$ErrorActionPreference = "Stop"

$TrainScript = Join-Path $PSScriptRoot "train_smolvla_py311_8gb_v2.ps1"

& $TrainScript -Steps 10 -SaveFreq 10 -RunName "smolvla_lora_r8_smoke_windows"

if ($LASTEXITCODE -ne 0) {
    throw "SmolVLA smoke training failed with exit code $LASTEXITCODE"
}

Write-Host "Smoke training and checkpoint save passed."
