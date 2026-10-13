param([string]$RunTag = "", [switch]$SmokeOnly)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $RunTag) { $RunTag = (Get-Date -Format "yyyyMMdd_HHmmss_fff") + "_v24" }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$') { throw "Unsafe run tag" }
$Python = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Evidence = Join-Path $Root "results\vla\future_joint_reuse_20261013_v24"
$Results = Join-Path $Root "results\vla\future_joint_$RunTag"
$SourceExport = Join-Path $Root "exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9"
$Export = Join-Path $Root "exports\lerobot\pick_lift_future_joint_$RunTag"
$SmokeExport = Join-Path $Evidence "smoke_export"
$Base = Join-Path $Root "results\vla\models\smolvla_base"
$Run = "smolvla_rgb_language_proprio_future_joint_lora16_$RunTag"
foreach ($Path in @($Results,$Export)) {
 if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite $Path" }
}
$Replay = Get-Content (Join-Path $Evidence "replay\vla_benchmark_episodes.csv") | ConvertFrom-Csv
if ($Replay.Count -ne 3 -or @($Replay | Where-Object { $_.success -ne 'True' -or [double]$_.maximum_lift_cm -lt 10 }).Count -ne 0) {
 throw "Successful Easy/Medium/Hard label replay is required"
}
$SmokeAudit = Get-Content (Join-Path $SmokeExport "export_audit.json") -Raw | ConvertFrom-Json
if ($SmokeAudit.status -ne 'FUTURE_JOINT_EXPORT_VERIFIED') { throw "Smoke export unverified" }
New-Item -ItemType Directory -Path $Results | Out-Null
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
foreach ($Name in @("LEROBOT_PLATEAU_CONFIG_JSON", "LEROBOT_LOSS_PLATEAU_JSON", "LEROBOT_INTERFACE_CONTRACT",
 "LEROBOT_JOINT_INTERFACE_CONTRACT", "LEROBOT_CHUNK_INITIAL_AUDIT_PATH", "LEROBOT_CHUNK_INITIAL_AUDIT_EXPECTED_CHUNK_SIZE",
 "LEROBOT_CHUNK_ABLATION_AUDIT", "LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK", "LEROBOT_LIFT_PHASE_METADATA")) {
 Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
}
function Set-Stage([string]$Stage,[string]$Message) {
 @{stage=$Stage; message=$Message; results=$Results; source_export=$SourceExport; dataset=$Export; training_run=$Run
   steps=30000; state_dim=25; action_dim=8; previous_adapter_loaded=$false; base=$Base
   label_source="next_decision_measured_arm_q_plus_current_gripper_command"; actual_recorded_joint_command=$false
   target_or_phase_in_student_input=$false; future_horizon_s=0.1; original_splits_preserved=$true
   label_replay_successes=3; label_replay_episodes=3; student_closed_loop_tested=$false
   updated_at_utc=[DateTime]::UtcNow.ToString("o")} | ConvertTo-Json | Set-Content (Join-Path $Results "pipeline_status.json") -Encoding UTF8
 Write-Host "FUTURE_JOINT_STAGE=$Stage $Message"
}
function Invoke-Logged([string]$Exe,[string[]]$Arguments,[string]$Log) {
 $Old=$ErrorActionPreference
 try {
  $ErrorActionPreference="Continue"
  & $Exe @Arguments 2>&1 | ForEach-Object {
   if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_.ToString() }
  } | Tee-Object -FilePath $Log
  $Code=$LASTEXITCODE
 } finally { $ErrorActionPreference=$Old }
 if ($Code -ne 0) { throw "Command failed ($Code); see $Log" }
}
function Make-Plan([string]$Dataset,[string]$Plan,[string]$Log) {
 Invoke-Logged $Python @("scripts\build_vla_sampling_plan.py","--dataset-root",(Join-Path $Dataset "train"),"--output",$Plan,
  "--balance-levels","--easy-weight","1","--medium-weight","1","--hard-weight","1",
  "--phase-0-weight","0.5","--phase-1-weight","1","--phase-2-weight","1.5","--phase-3-weight","1.5","--phase-4-weight","2") $Log
}
function Train([string]$Dataset,[string]$Plan,[string]$Name,[int]$Steps,[int]$Save,[int]$Warmup) {
 & (Join-Path $PSScriptRoot "train_smolvla_dagger_recovery_8gb_v7.ps1") -FreshLora -AdapterCheckpoint $Base `
  -DatasetRoot (Join-Path $Dataset "train") -DatasetRepoId "local/isaac-future-joint-v24-train" -SamplingPlan $Plan `
  -RunName $Name -Steps $Steps -SaveFreq $Save -BatchSize 1 -Seed 1000 -ChunkSize 4 -ActionSteps 1 `
  -SchedulerWarmupSteps $Warmup -SchedulerDecaySteps $Steps -PeakLearningRate 0.00001 -DecayLearningRate 0.000001 `
  -ZAxisWeight 1 -GripperIndex 7 -GripperWeight 1 -JointInterfaceContract (Join-Path $Dataset "train\interface_contract.json")
 $Init=Get-Content (Join-Path $Root "results\vla\training_logs\$Name.initialization.json") -Raw | ConvertFrom-Json
 if ($Init.status -ne "FRESH_LORA_INITIALIZATION_VERIFIED" -or $Init.old_adapter_loaded) { throw "Fresh LoRA initialization not verified" }
}
try {
 $BaseHash=(Get-FileHash (Join-Path $Base "model.safetensors")).Hash
 Set-Stage "SMOKE_TRAINING" "20-update smoke check of future-joint labels, normalization and checkpoint interface."
 $SmokePlan=Join-Path $Results "smoke_sampling_plan.json"
 Make-Plan $SmokeExport $SmokePlan (Join-Path $Results "smoke_sampling.log")
 Train $SmokeExport $SmokePlan ($Run+"_smoke") 20 20 3
 Set-Stage "SMOKE_VALIDATION" "Check joint/gripper output denormalization on independent validation frames."
 Invoke-Logged $Python @("scripts\scan_vla_joint_v23.py","--checkpoint-root",(Join-Path $Root "results\vla\training\${Run}_smoke\checkpoints"),
  "--dataset-root",(Join-Path $SmokeExport "validation"),"--output",(Join-Path $Results "smoke_validation.json"),"--steps","20","--samples","4") (Join-Path $Results "smoke_validation.log")
 if ($SmokeOnly) { Set-Stage "SMOKE_COMPLETE" "Export/training/validation path verified; no full training."; return }
 Set-Stage "EXPORTING_EXISTING_DATA" "Reuse 600 original successful episodes with unchanged split membership; no recollection."
 Invoke-Logged $Python @("scripts\export_vla_future_joint_v24.py","--source-export",$SourceExport,"--output-root",$Export) (Join-Path $Results "export.log")
 $Audit=Get-Content (Join-Path $Export "export_audit.json") -Raw | ConvertFrom-Json
 if ($Audit.status -ne "FUTURE_JOINT_EXPORT_VERIFIED" -or $Audit.smoke_subset) { throw "Full export audit failed" }
 $Frames=($Audit.splits | Where-Object { $_.split -eq 'train' }).frames
 if ($Frames -ne 26828) { throw "Unexpected training frame count: $Frames" }
 $Plan=Join-Path $Results "train_sampling_plan.json"
 Make-Plan $Export $Plan (Join-Path $Results "sampling.log")
 Set-Stage "FRESH_FINETUNING" "Original SmolVLA + new rank-16 LoRA, 30000 updates; future states are labels only."
 Train $Export $Plan $Run 30000 6000 500
 Set-Stage "FIXED_JOINT_VALIDATION" "Fixed 240 validation frames at 6k/12k/18k/24k/30k; joint radians, gripper and hold-current-q baseline."
 Invoke-Logged $Python @("scripts\scan_vla_joint_v23.py","--checkpoint-root",(Join-Path $Root "results\vla\training\$Run\checkpoints"),
  "--dataset-root",(Join-Path $Export "validation"),"--output",(Join-Path $Results "validation_scan.json"),
  "--steps","6000","12000","18000","24000","30000","--samples","240") (Join-Path $Results "validation.log")
 if ((Get-FileHash (Join-Path $Base "model.safetensors")).Hash -ne $BaseHash) { throw "Base model changed" }
 Set-Stage "TRAINING_AND_VALIDATION_COMPLETE" "Future-joint model trained and validated; student closed-loop performance remains to be tested."
} catch { Set-Stage "FAILED" $_.Exception.Message; throw }
