param([string]$RunTag='')
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $RunTag) { $RunTag=Get-Date -Format 'yyyyMMdd_HHmmss_fff' }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid tag' }
$Output=Join-Path $Root "results\vla\best_v17_horizon2_$RunTag"
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$Run='smolvla_v9_chunk4_20261012_195615_610_v17'
$Baseline=Join-Path $Root 'results\vla\v9_chunk_ablation_20261012_195615_610_v17\chunk4_closed_loop\visual_only_benchmark_20261012_210112_048'
$Validation=Join-Path $Root 'results\vla\v9_chunk_ablation_20261012_195615_610_v17\chunk4_validation.json'
$Checkpoint=Join-Path $Root "results\vla\training\$Run\checkpoints\006000\pretrained_model"
$Hash=(Get-FileHash (Join-Path $Checkpoint 'adapter_model.safetensors')).Hash
$BaselineContext=Get-Content (Join-Path $Baseline 'test_context.json') -Raw | ConvertFrom-Json
if ($Hash -ne $BaselineContext.checkpoint_sha256) { throw 'Baseline checkpoint mismatch' }
$Probe=New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback,0)
try { $Probe.Start();$Port=$Probe.LocalEndpoint.Port } finally { $Probe.Stop() }
function Stage([string]$Name,[string]$Message) {
 @{stage=$Name;message=$Message;results=$Output;baseline=$Baseline;checkpoint=$Checkpoint;
   checkpoint_sha256=$Hash;training_run=$Run;checkpoint_step=6000;chunk_size=4;execution_horizon=2;
   action_repeat=5;levels=@('easy','medium');episodes_per_level=3;planned_episodes=6;
   retraining=$false;gui_visible=$true;port=$Port;input='RGB_language_proprio_visual_estimate_phase';
   action='delta_xyz_target_quaternion_gripper';minimum_actual_lift_m=0.10;
   updated_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 4 |
   Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
 Write-Output "BEST_V17_H2_STAGE=$Name RESULTS=$Output"
}
try {
 Stage 'RUNNING' 'Historical best on paired six Easy/Medium scenes; only execution horizon changes 1 to 2. No retraining.'
 & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
  -TrainingRun $Run -CheckpointStep 6000 -ValidationReport $Validation -OutputRoot $Output -Port $Port `
  -Levels easy,medium -EpisodesPerLevel 3 -MaxSteps 600 -ExecutionHorizon 2 `
  -InputInterface legacy -PhaseMechanism legacy -StudentPhaseMode legacy -OrientationMode model `
  -GraspEnableDistanceM 0.075 -RecordPhaseDiagnostics -ShowWindow -Realtime
 if ((Get-FileHash (Join-Path $Checkpoint 'adapter_model.safetensors')).Hash -ne $Hash) { throw 'Checkpoint changed' }
 $Reports=@(Get-ChildItem -LiteralPath $Output -Recurse -Filter 'vla_benchmark_report.json' -File)
 if ($Reports.Count -ne 1) { throw 'Expected one complete benchmark report' }
 Stage 'COMPARING' 'Compare matched scenes, true lift and first closure; count requests separately from executed actions.'
 & 'C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe' scripts\compare_vla_execution_horizons.py `
  --baseline $Baseline --candidate $Reports[0].Directory.FullName --output (Join-Path $Output 'comparison.json')
 if ($LASTEXITCODE -ne 0) { throw 'Paired comparison failed' }
 Stage 'COMPLETE' 'Six diagnostic episodes and paired comparison saved. Original checkpoint unchanged.'
} catch { Stage 'FAILED' $_.Exception.Message; throw }
