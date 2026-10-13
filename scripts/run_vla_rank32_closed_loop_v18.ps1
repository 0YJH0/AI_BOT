param([ValidateRange(1024,65535)][int]$Port=8786)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Tag=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Output=Join-Path $Root "results\vla\rank32_closed_loop_${Tag}_v18"
$TrainingRun='smolvla_chunk4_rank32_20261012_212747_052_v18'
$Validation=Join-Path $Root 'results\vla\rank32_plateau_20261012_212747_052_v18\validation.json'
$Baseline=Join-Path $Root 'results\vla\v9_chunk_ablation_20261012_195615_610_v17\chunk4_closed_loop\visual_only_benchmark_20261012_210112_048'
$Export=Join-Path $Root 'exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9'
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
function Stage([string]$Name,[string]$Message) {
    @{stage=$Name;message=$Message;results=$Output;training_run=$TrainingRun;checkpoint_step=6000;
      episodes_per_level=3;levels=@('easy','medium');planned_total_episodes=6;baseline=$Baseline;
      updated_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 4 |
        Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "RANK32_CLOSED_LOOP_STAGE=$Name RESULTS=$Output"
}
try {
    Stage 'BENCHMARK_R32' 'Same six Easy/Medium scenes, legacy interface, model orientation, 10 cm true lift.'
    & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
        -TrainingRun $TrainingRun -CheckpointStep 6000 -DatasetExportRoot $Export `
        -ValidationReport $Validation -OutputRoot $Output -Port $Port `
        -Levels easy,medium -EpisodesPerLevel 3 -MaxSteps 600 -InputInterface legacy `
        -PhaseMechanism legacy -OrientationMode model -GraspEnableDistanceM 0.075 -RecordPhaseDiagnostics
    $Reports=@(Get-ChildItem -LiteralPath $Output -Recurse -Filter 'vla_benchmark_report.json' -File)
    if ($Reports.Count -ne 1) {throw 'Expected exactly one completed report'}
    Stage 'COMPARING' 'Verify scene and execution settings; compare with pre-training r16 checkpoint.'
    & $Python scripts\compare_vla_close_lift_pair.py --baseline $Baseline `
        --candidate $Reports[0].Directory.FullName --output (Join-Path $Output 'paired_comparison.json')
    if ($LASTEXITCODE -ne 0) {throw 'Paired comparison failed'}
    Stage 'COMPLETE' 'Six episodes and paired comparison complete. Rank and additional training both changed.'
} catch {Stage 'FAILED' $_.Exception.Message;throw}
