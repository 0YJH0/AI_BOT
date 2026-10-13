param([ValidateRange(1024,65535)][int]$Port=8787)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Output=Join-Path $Root ('results\vla\rank32_hard_gui_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'_v19')
$Run='smolvla_chunk4_rank32_20261012_212747_052_v18'
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
function Stage([string]$Name,[string]$Message) {
    @{stage=$Name;message=$Message;results=$Output;training_run=$Run;checkpoint_step=6000;
      planned_episodes=5;levels=@('hard');gui_visible=$true;
      updated_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json |
        Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "HARD_GUI_STAGE=$Name RESULTS=$Output"
}
try {
    Stage 'RUNNING_HARD_GUI' 'Five fixed-seed Hard trials; r32 GPU inference, visible Isaac window, 10cm lift.'
    & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
        -TrainingRun $Run -CheckpointStep 6000 `
        -ValidationReport 'results\vla\rank32_plateau_20261012_212747_052_v18\validation.json' `
        -DatasetExportRoot (Join-Path $Root 'exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9') `
        -OutputRoot $Output -Port $Port -Levels hard -EpisodesPerLevel 5 -MaxSteps 600 `
        -InputInterface legacy -PhaseMechanism legacy -OrientationMode model `
        -GraspEnableDistanceM 0.075 -RecordPhaseDiagnostics -ShowWindow -Realtime
    Stage 'SUMMARIZING' 'Verify checkpoint and aggregate per-episode grasp/lift diagnostics.'
    & $Python scripts\summarize_vla_hard_r32_v19.py --root $Output
    if ($LASTEXITCODE -ne 0) {throw 'Hard summary failed'}
    Stage 'COMPLETE' 'Five visible Hard grasp trials completed; see hard_summary.json.'
} catch {Stage 'FAILED' $_.Exception.Message;throw}
