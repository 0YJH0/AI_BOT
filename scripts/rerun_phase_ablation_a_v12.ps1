param([string]$RunTag="20261011_v12", [int]$Port=8774)
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$') {throw "Unsafe run tag"}
$Output=Join-Path $Root ("results\vla\phase_ablation_a_repair_" + $RunTag)
if (Test-Path -LiteralPath $Output) {throw "Refusing to overwrite $Output"}
New-Item -ItemType Directory -Path $Output | Out-Null
function Set-Stage([string]$Stage,[string]$Message) {
    @{stage=$Stage; message=$Message; planned_episodes=6; checkpoint_step=6000;
      original_results="results/vla/phase_ablation_20261011_v11";
      updated_at_utc=[DateTime]::UtcNow.ToString("o"); retraining=$false
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "pipeline_status.json") -Encoding UTF8
}
try {
    Set-Stage "RERUN_A_MEDIUM_HARD" "Same weights and scene seeds; repaired coherent consecutive-frame initialization."
    & (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") -RunTag "20261011_truth_v9" `
        -TrainingRun "smolvla_execution_aligned_full_lora16_20261011_truth_v9" -CheckpointStep 6000 `
        -ValidationReport "results\vla\fresh_lora_comparison_20261011_v10\continued_validation240.json" `
        -OutputRoot $Output -Port $Port -Levels medium,hard -EpisodesPerLevel 3 -MaxSteps 600 `
        -PhaseMechanism legacy -OrientationMode model -RecordPhaseDiagnostics
    Set-Stage "RERUN_COMPLETE" "Six supplemental episodes completed; original experiment remains untouched."
} catch {
    Set-Stage "FAILED" $_.Exception.Message
    throw
}
