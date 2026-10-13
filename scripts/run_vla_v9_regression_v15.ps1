param([int]$Port=8780)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Output=Join-Path $Root ('results\vla\v9_regression_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'_v15')
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$env:PYTHONUTF8='1'
function Stage([string]$Name,[string]$Message) {
    @{stage=$Name; message=$Message; results=$Output; retraining=$false;
      updated_at_utc=[DateTime]::UtcNow.ToString('o'); planned_episodes=6
    } | ConvertTo-Json | Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
}
try {
    Stage 'LOCKING_BASELINE' 'Freeze v9 checkpoint, processors, base model and current execution source hashes.'
    & $Python scripts\audit_vla_v9_regression.py --output $Output --freeze
    if ($LASTEXITCODE -ne 0) {throw 'Baseline lock failed'}
    Stage 'V9_BASELINE_RUNNING' 'Same v9 weights, legacy phase and model orientation; current corrected camera readiness. Easy/Medium 3 each.'
    & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
        -TrainingRun 'smolvla_execution_aligned_full_lora16_20261011_truth_v9' -CheckpointStep 6000 `
        -ValidationReport 'results\vla\fresh_lora_comparison_20261011_v10\continued_validation240.json' `
        -OutputRoot $Output -Port $Port -Levels easy,medium -EpisodesPerLevel 3 -MaxSteps 600 `
        -PhaseMechanism legacy -OrientationMode model -InputInterface legacy -RecordPhaseDiagnostics
    Stage 'AUDITING_REGRESSIONS' 'Verify frozen artifacts and compare evidence without single-factor claims for bundled modifications.'
    & $Python scripts\audit_vla_v9_regression.py --output $Output
    if ($LASTEXITCODE -ne 0) {throw 'Regression audit failed; see logs'}
    Stage 'REGRESSION_REVIEW_COMPLETE' 'Read regression_report.json. No training launched; unisolated factors remain explicitly pending.'
} catch { Stage 'FAILED' $_.Exception.Message; throw }
