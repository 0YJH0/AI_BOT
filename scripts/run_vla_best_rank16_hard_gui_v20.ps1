param([ValidateRange(1024,65535)][int]$Port=8788)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Output=Join-Path $Root ('results\vla\best_rank16_hard_gui_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'_v20')
$Run='smolvla_v9_chunk4_20261012_195615_610_v17'
$Reference=Join-Path $Root 'results\vla\rank32_hard_gui_20261012_225337_352_v19'
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
function Stage([string]$Name,[string]$Message) {
    @{stage=$Name;message=$Message;results=$Output;training_run=$Run;checkpoint_step=6000;
      selection='Best observed total success in recent six-scene Easy/Medium tests: r16 chunk4, 3/6';
      planned_episodes=5;levels=@('hard');gui_visible=$true;reference=$Reference;
      updated_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json |
        Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "BEST_HARD_GUI_STAGE=$Name RESULTS=$Output"
}
try {
    Stage 'RUNNING_HARD_GUI' 'r16 chunk4 checkpoint; same five Hard scenes and GUI protocol as r32.'
    & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
        -TrainingRun $Run -CheckpointStep 6000 `
        -ValidationReport 'results\vla\v9_chunk_ablation_20261012_195615_610_v17\chunk4_validation.json' `
        -DatasetExportRoot (Join-Path $Root 'exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9') `
        -OutputRoot $Output -Port $Port -Levels hard -EpisodesPerLevel 5 -MaxSteps 600 `
        -InputInterface legacy -PhaseMechanism legacy -OrientationMode model `
        -GraspEnableDistanceM 0.075 -RecordPhaseDiagnostics -ShowWindow -Realtime
    Stage 'SUMMARIZING' 'Verify r16 checkpoint and paired scenes/settings against completed r32 Hard run.'
    & $Python scripts\summarize_vla_hard_r32_v19.py --root $Output --expected-rank 16 --compare-root $Reference
    if ($LASTEXITCODE -ne 0) {throw 'Hard paired summary failed'}
    Stage 'COMPLETE' 'Five visible Hard trials complete; hard_summary.json and paired_hard_comparison.json saved.'
} catch {Stage 'FAILED' $_.Exception.Message;throw}
