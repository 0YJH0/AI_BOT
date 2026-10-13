param([int]$Port = 8776)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
& (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") `
    -TrainingRun "smolvla_relative4d_fresh_lora16_20261011_v13" `
    -CheckpointStep 30000 -Levels @("easy", "medium") -EpisodesPerLevel 3 -MaxSteps 600 `
    -ValidationReport "results\vla\relative4d_20261011_v13\validation_scan_30000.json" `
    -DatasetExportRoot (Join-Path $Root "exports\lerobot\pick_lift_relative4d_20261011_v13") `
    -InputInterface relative4d -RecordPhaseDiagnostics -Port $Port `
    -OutputRoot "results\vla\relative4d_30k_diagnostic_20261012_v14" `
    -AllowValidationGateOverride `
    -ValidationGateOverrideReason "User requested 3 Easy and 3 Medium diagnostic closed-loop episodes for the 30k policy despite its 2.33 cm offline error; the 2 cm gate has not passed."
