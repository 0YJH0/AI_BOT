param([int]$Port = 8778)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
& (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
    -TrainingRun 'smolvla_relative4d_close_lift_lora16_20261012_v14' `
    -CheckpointStep 6000 -Levels @('easy','medium') -EpisodesPerLevel 3 -MaxSteps 600 `
    -ValidationReport 'results\vla\relative4d_close_lift_20261012_v14\validation_scan.json' `
    -DatasetExportRoot (Join-Path $Root 'exports\lerobot\pick_lift_relative4d_20261011_v13') `
    -InputInterface relative4d -RecordPhaseDiagnostics -Port $Port `
    -OutputRoot 'results\vla\relative4d_close_lift_paired_20261012_v14' `
    -AllowValidationGateOverride `
    -ValidationGateOverrideReason 'User requested the same six Easy/Medium diagnostic scenes for the v14 best checkpoint (30k + 6k); offline 2.203 cm has not passed the 2 cm gate.'
