param([string]$RunTag = "", [int]$EpisodesPerLevel = 3,
 [ValidateSet(1,2,4)][int]$ExecutionHorizon = 1)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $RunTag) { $RunTag = Get-Date -Format 'yyyyMMdd_HHmmss_fff' }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $EpisodesPerLevel -lt 1) { throw 'Invalid run settings' }
$TrainingRun = 'smolvla_rgb_language_proprio_future_joint_lora16_20261013_031623_195_v24'
$Checkpoint = Join-Path $Root "results\vla\training\$TrainingRun\checkpoints\030000\pretrained_model"
$Validation = Join-Path $Root 'results\vla\future_joint_20261013_031623_195_v24\validation_scan.json'
$Scan = Get-Content $Validation -Raw | ConvertFrom-Json
if ($Scan.status -ne 'JOINT_CHECKPOINT_SCAN_COMPLETE' -or @($Scan.results | Where-Object { $_.step -eq 30000 }).Count -ne 1) {
 throw 'Validated 30k joint checkpoint required'
}
$Config = Get-Content (Join-Path $Checkpoint 'config.json') -Raw | ConvertFrom-Json
$Contract = Get-Content (Join-Path $Checkpoint 'interface_contract.json') -Raw | ConvertFrom-Json
if ($Config.chunk_size -ne 4 -or $Config.input_features.'observation.state'.shape[0] -ne 25 -or
 $Contract.execution -ne 'direct_PD_joint_target_no_student_IK') { throw 'Unexpected joint policy interface' }
$Output = Join-Path $Root "results\vla\joint_closed_loop_${RunTag}_v24"
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$Hash = (Get-FileHash (Join-Path $Checkpoint 'adapter_model.safetensors')).Hash
$Probe = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
try { $Probe.Start(); $Port = $Probe.LocalEndpoint.Port } finally { $Probe.Stop() }
$Server = $null
$Isaac = $null
$Complete = $false
function Stage([string]$Name, [string]$Message) {
 @{stage=$Name;message=$Message;checkpoint=$Checkpoint;checkpoint_sha256=$Hash;checkpoint_step=30000;
   training_run=$TrainingRun;results=$Output;levels=@('easy','medium');episodes_per_level=$EpisodesPerLevel;
   planned_episodes=2*$EpisodesPerLevel;chunk_size=4;execution_horizon=$ExecutionHorizon;action_repeat=5;
   checkpoint_reference_execution_horizon=1;execution_horizon_override=($ExecutionHorizon -ne 1);
   minimum_actual_lift_m=0.10;gui_visible=$true;retraining=$false;inference_device='cuda';port=$Port;
   input='RGB_language_robot_proprioception';action='absolute_7_joint_radians_plus_gripper';
   scripted_fallback=$false;target_or_phase_in_model_input=$false;scope='small_diagnostic_not_statistical_benchmark';
   updated_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 4 |
   Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
 Write-Output "JOINT_DIAGNOSTIC_STAGE=$Name RESULTS=$Output"
}
$env:OMNI_KIT_ACCEPT_EULA='YES'
$env:PYTHONPATH=Join-Path $Root 'src'
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$env:HF_HUB_OFFLINE='1'
try {
 Stage 'STARTING_GPU_SERVER' 'Preserved 30k weights; no training; GUI diagnostic requested.'
 $ServerArgs=@('scripts\serve_smolvla_windows_v044_v4.py','--checkpoint',('"'+$Checkpoint+'"'),
  '--port',$Port,'--execution-horizon',$ExecutionHorizon,'--device','cuda','--seed',20261011,'--gpu-memory-fraction',0.35,'--profile-memory')
 $Server=Start-Process 'C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe' -ArgumentList $ServerArgs `
  -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $Output 'server_stdout.log') -RedirectStandardError (Join-Path $Output 'server_stderr.log')
 $Deadline=[DateTime]::UtcNow.AddSeconds(180)
 $Ready=$false
 while ([DateTime]::UtcNow -lt $Deadline) {
  $Server.Refresh()
  if ($Server.HasExited) { throw 'GPU inference server exited; inspect server logs' }
  if (Select-String -LiteralPath (Join-Path $Output 'server_stdout.log') -Pattern 'SMOLVLA_SERVER_READY=' -Quiet) { $Ready=$true;break }
  Start-Sleep -Milliseconds 500
 }
 if (-not $Ready) { throw 'GPU inference server readiness timeout' }
 Stage 'ISAAC_GUI_DIAGNOSTIC' 'Easy/Medium three scenes each, direct joint PD, model-only decisions, true lift evaluation.'
 $IsaacArgs=@('scripts\run_vla_benchmark.py','--levels','easy','medium','--episodes-per-level',$EpisodesPerLevel,
  '--max-steps',600,'--minimum-true-lift-m',0.10,'--pure-vla','--require-vla','--proprio-only','--joint-target-actions',
  '--vla-port',$Port,'--vla-timeout',60,'--execution-horizon',$ExecutionHorizon,'--action-repeat',5,
  '--camera-ready-timeout-s',180,'--visual-init-attempts',1,'--progress-every',100,
  '--record-phase-diagnostics','--device','cuda:0','--rendering_mode','performance','--realtime','--keep-open',
  '--output-dir',('"'+$Output+'"'))
 $Isaac=Start-Process 'C:\Users\yjh\anaconda3\envs\isaaclab23\pythonw.exe' -ArgumentList $IsaacArgs `
  -WorkingDirectory $Root -WindowStyle Normal -PassThru `
  -RedirectStandardOutput (Join-Path $Output 'isaac_stdout.log') -RedirectStandardError (Join-Path $Output 'isaac_stderr.log')
 @{server_pid=$Server.Id;isaac_pid=$Isaac.Id;port=$Port} | ConvertTo-Json | Set-Content (Join-Path $Output 'processes.json') -Encoding UTF8
 $null=$Isaac.Handle
 while (-not $Isaac.WaitForExit(1000)) {
  if (-not $Complete -and (Select-String -LiteralPath (Join-Path $Output 'isaac_stdout.log') -Pattern 'VLA_BENCHMARK_SUMMARY=' -Quiet)) {
   $Report=Get-Content (Join-Path $Output 'vla_benchmark_report.json') -Raw | ConvertFrom-Json
   if ($Report.status -ne 'VLA_BENCHMARK_OK') { throw 'Incomplete benchmark report' }
   if ((Get-FileHash (Join-Path $Checkpoint 'adapter_model.safetensors')).Hash -ne $Hash) { throw 'Checkpoint changed' }
   $Complete=$true
   Stage 'COMPLETE_GUI_OPEN' 'Reports saved. GUI renders final scene without further actions; close Isaac window when finished.'
   $Server.Refresh()
   if (-not $Server.HasExited) { Stop-Process -InputObject $Server; $null=$Server.WaitForExit(10000) }
  }
 }
 if (-not $Complete) { throw 'Isaac exited before completing all diagnostic episodes; inspect logs' }
 Stage 'COMPLETE' 'Six diagnostic episodes saved; visualization window closed; checkpoint preserved.'
} catch { Stage 'FAILED' $_.Exception.Message; throw }
finally {
 foreach ($Child in @($Server,$Isaac)) {
  if ($null -ne $Child) { $Child.Refresh();if (-not $Child.HasExited) { Stop-Process -InputObject $Child -ErrorAction SilentlyContinue } }
 }
}
