param([string]$RunTag = "", [int]$Steps = 30000, [switch]$PilotOnly, [switch]$SmokeOnly)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $RunTag) { $RunTag = (Get-Date -Format "yyyyMMdd_HHmmss_fff") + "_v23" }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $Steps -lt 1) { throw "Invalid settings" }
$IsaacPython = "C:\Users\yjh\anaconda3\envs\isaaclab23\python.exe"
$Python = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Results = Join-Path $Root "results\vla\joint_fresh_$RunTag"
$Source = Join-Path $Root "dataset\vla_joint_execution_aligned_$RunTag"
$Export = Join-Path $Root "exports\lerobot\pick_lift_joint_$RunTag"
$Base = Join-Path $Root "results\vla\models\smolvla_base"
$Run = "smolvla_rgb_language_proprio_joint_fresh_lora16_$RunTag"
foreach ($Path in @($Results,$Source,$Export)) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite $Path" }
}
New-Item -ItemType Directory -Path $Results | Out-Null
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
foreach ($Name in @("LEROBOT_PLATEAU_CONFIG_JSON", "LEROBOT_LOSS_PLATEAU_JSON", "LEROBOT_INTERFACE_CONTRACT",
 "LEROBOT_JOINT_INTERFACE_CONTRACT", "LEROBOT_CHUNK_INITIAL_AUDIT_PATH", "LEROBOT_CHUNK_INITIAL_AUDIT_EXPECTED_CHUNK_SIZE",
 "LEROBOT_CHUNK_ABLATION_AUDIT", "LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK", "LEROBOT_LIFT_PHASE_METADATA")) {
 Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
}
function Set-Stage([string]$Stage,[string]$Message) {
 @{stage=$Stage; message=$Message; results=$Results; source=$Source; dataset=$Export; training_run=$Run
   steps=$Steps; state_dim=25; action_dim=8; previous_adapter_loaded=$false; base=$Base
   action_semantics="absolute_7_arm_joint_radians_plus_gripper"; target_or_phase_in_student_input=$false
   updated_at_utc=[DateTime]::UtcNow.ToString("o")} | ConvertTo-Json | Set-Content (Join-Path $Results "pipeline_status.json") -Encoding UTF8
 Write-Host "JOINT_PIPELINE_STAGE=$Stage $Message"
}
function Invoke-Logged([string]$Exe,[string[]]$Arguments,[string]$Log) {
 $Old = $ErrorActionPreference
 try {
  $ErrorActionPreference = "Continue"
  & $Exe @Arguments 2>&1 | ForEach-Object {
   if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_.ToString() }
  } | Tee-Object -FilePath $Log
  $Code = $LASTEXITCODE
 } finally { $ErrorActionPreference = $Old }
 if ($Code -ne 0) { throw "Command failed ($Code); see $Log" }
}
function Collect([string]$Destination,[int]$Count,[int]$Attempts,[string]$Report,[string]$Log,[int]$SeedOffset) {
 $Arguments = @("scripts\run_vla_benchmark.py", "--headless", "--levels","easy","medium","hard",
  "--max-steps","600","--collect-full-trajectories",$Destination,
  "--collection-target-per-level",[string]$Count,"--collection-max-attempts-per-level",[string]$Attempts,
  "--collection-seed-offset",[string]$SeedOffset,"--collection-teacher","truth","--joint-target-actions",
  "--vla-position-target-mode","delta","--execution-horizon","1","--action-repeat","5",
  "--visual-init-attempts","1","--camera-ready-timeout-s","45","--visual-invalid-pose-timeout-s","8",
  "--device","cuda:0","--rendering_mode","performance","--output-dir",$Report)
 Invoke-Logged $IsaacPython $Arguments $Log
 $Summary = Get-Content (Join-Path $Destination "collection_summary.json") -Raw | ConvertFrom-Json
 if ($Summary.status -ne "FULL_TRAJECTORY_COLLECTION_OK") { throw "Incomplete collection; training prohibited" }
}
try {
 Set-Stage "TEACHER_JOINT_PILOT" "Require two audited successful direct-joint trajectories per level before large collection."
 $BaseHash = (Get-FileHash (Join-Path $Base "model.safetensors")).Hash
 Collect (Join-Path $Results "pilot_dataset") 2 12 (Join-Path $Results "pilot_reports") (Join-Path $Results "pilot.log") 20263013
 if ($PilotOnly) { Set-Stage "PILOT_COMPLETE" "Six physically successful direct-joint demonstrations verified; no training."; return }
 if ($SmokeOnly) { $Target=4; $Train=2; $Val=1; $Test=1; $Attempts=30 }
 else { $Target=200; $Train=170; $Val=15; $Test=15; $Attempts=1000 }
 Set-Stage "COLLECTING_JOINT_DEMONSTRATIONS" "New live command labels; preserve old datasets; successful Easy/Medium/Hard quotas."
 Collect $Source $Target $Attempts (Join-Path $Results "collection_reports") (Join-Path $Results "collection.log") 20264013
 Set-Stage "EXPORTING_AND_AUDITING" "Robot-only inputs; identical executed joint target labels, disjoint physical scene splits."
 Invoke-Logged $Python @("scripts\export_vla_joint_v23.py","--source-root",$Source,"--output-root",$Export,
  "--train-per-level",[string]$Train,"--validation-per-level",[string]$Val,"--test-per-level",[string]$Test) (Join-Path $Results "export.log")
 $Plan=Join-Path $Results "train_sampling_plan.json"
 Invoke-Logged $Python @("scripts\build_vla_sampling_plan.py","--dataset-root",(Join-Path $Export "train"),"--output",$Plan,
  "--balance-levels","--easy-weight","1","--medium-weight","1","--hard-weight","1",
  "--phase-0-weight","0.5","--phase-1-weight","1","--phase-2-weight","1.5","--phase-3-weight","1.5","--phase-4-weight","2") (Join-Path $Results "sampling.log")
 $env:LEROBOT_JOINT_INTERFACE_CONTRACT=Join-Path $Export "train\interface_contract.json"
 Set-Stage "FRESH_FINETUNING" "Original SmolVLA + fresh rank-16 LoRA; no Cartesian Z weighting on joint dimensions."
 $Save = if ($SmokeOnly) { $Steps } else { 6000 }
 $Warmup = if ($SmokeOnly) { [Math]::Min(3,$Steps-1) } else { 500 }
 & (Join-Path $PSScriptRoot "train_smolvla_dagger_recovery_8gb_v7.ps1") -FreshLora -AdapterCheckpoint $Base `
  -DatasetRoot (Join-Path $Export "train") -DatasetRepoId "local/isaac-joint-v23-train" -SamplingPlan $Plan `
  -RunName $Run -Steps $Steps -SaveFreq $Save -BatchSize 1 -Seed 1000 -ChunkSize 4 -ActionSteps 1 `
  -SchedulerWarmupSteps $Warmup -SchedulerDecaySteps $Steps -PeakLearningRate 0.00001 -DecayLearningRate 0.000001 `
  -ZAxisWeight 1 -GripperIndex 7 -GripperWeight 1 -JointInterfaceContract (Join-Path $Export "train\interface_contract.json")
 $Init=Get-Content (Join-Path $Root "results\vla\training_logs\$Run.initialization.json") -Raw | ConvertFrom-Json
 if ($Init.status -ne "FRESH_LORA_INITIALIZATION_VERIFIED" -or $Init.old_adapter_loaded) { throw "Fresh adapter audit failed" }
 Set-Stage "FIXED_JOINT_VALIDATION" "Joint radians/gripper validation; never misreport arm joint errors as Cartesian centimeters."
 $ScanArgs=@("scripts\scan_vla_joint_v23.py","--checkpoint-root",(Join-Path $Root "results\vla\training\$Run\checkpoints"),
  "--dataset-root",(Join-Path $Export "validation"),"--output",(Join-Path $Results "validation_scan.json"),"--steps")
 if ($SmokeOnly) { $ScanArgs += @([string]$Steps,"--samples","4") }
 else { $ScanArgs += @("6000","12000","18000","24000","30000","--samples","240") }
 Invoke-Logged $Python $ScanArgs (Join-Path $Results "validation.log")
 if ((Get-FileHash (Join-Path $Base "model.safetensors")).Hash -ne $BaseHash) { throw "Base model changed" }
 Set-Stage "TRAINING_AND_VALIDATION_COMPLETE" "Joint model ready; no previous files overwritten; student closed-loop benchmark not run by this training request."
} catch { Set-Stage "FAILED" $_.Exception.Message; throw }
