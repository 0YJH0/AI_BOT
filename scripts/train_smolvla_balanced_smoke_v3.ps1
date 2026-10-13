$ErrorActionPreference = "Stop"

$TrainScript = Join-Path $PSScriptRoot "train_smolvla_balanced_8gb_v3.ps1"
& $TrainScript `
    -Steps 10 `
    -SaveFreq 10 `
    -LoraRank 16 `
    -RunName "smolvla_balanced504_aug_lora16_smoke"

if ($LASTEXITCODE -ne 0) {
    throw "Balanced VLA smoke training failed with exit code $LASTEXITCODE"
}

Write-Host "Balanced-data augmentation smoke test passed."
