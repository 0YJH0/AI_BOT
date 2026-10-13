param(
    [int]$Steps = 10000,
    [int]$SaveFreq = 1000,
    [int]$BatchSize = 1,
    [int]$SchedulerWarmupSteps = 500,
    [int]$SchedulerDecaySteps = 10000,
    [double]$PeakLearningRate = 0.00001,
    [double]$DecayLearningRate = 0.000001,
    [double]$ZAxisWeight = 3.0,
    [string]$RunName = "smolvla_visual_fsm_delta_dagger15mm_lora16_v6",
    [string]$DatasetRoot = "",
    [string]$DatasetRepoId = "local/isaac-pick-lift-vla-visual-fsm-delta-weighted-dagger15mm-v6-train",
    [string]$SamplingPlan = "",
    [string]$AdapterCheckpoint = "",
    [switch]$FreshLora,
    [int]$Seed = 1000,
    [int]$ChunkSize = 4,
    [int]$ActionSteps = 0,
    [ValidateSet("normalized","per_sample")][string]$SampleWeightReduction = "normalized",
    [string]$InterfaceContract = "",
    [string]$JointInterfaceContract = "",
    [int]$GripperIndex = -1,
    [double]$GripperWeight = 1.0,
    [int]$ExpectedTrainableParams = 742656,
    [double]$LiftZAxisWeight = 0.0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Launcher = Join-Path $PSScriptRoot "lerobot_train_windows.py"
if ([string]::IsNullOrWhiteSpace($AdapterCheckpoint)) {
    if ($FreshLora) { $AdapterCheckpoint = Join-Path $ProjectRoot "results\vla\models\smolvla_base" }
    else { $AdapterCheckpoint = Join-Path $ProjectRoot "results\vla\training\smolvla_visual_fsm_delta_weighted_lora16_v5\checkpoints\027500\pretrained_model" }
} elseif (-not [System.IO.Path]::IsPathRooted($AdapterCheckpoint)) {
    $AdapterCheckpoint = Join-Path $ProjectRoot $AdapterCheckpoint
}
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw "Unsafe RunName" }
if ($FreshLora -and ((Test-Path (Join-Path $AdapterCheckpoint "adapter_config.json")) -or (Test-Path (Join-Path $AdapterCheckpoint "adapter_model.safetensors")))) {
    throw "Fresh LoRA must start from original base weights, not an adapter checkpoint"
}
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_visual_fsm_delta_weighted_dagger15mm_v6\train"
} elseif (-not [System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot $DatasetRoot
}
if ([string]::IsNullOrWhiteSpace($SamplingPlan)) {
    $SamplingPlan = Join-Path $ProjectRoot "results\vla\sampling_plans\v6_train_dagger15mm_sampling_plan.json"
} elseif (-not [System.IO.Path]::IsPathRooted($SamplingPlan)) {
    $SamplingPlan = Join-Path $ProjectRoot $SamplingPlan
}
$OutputDir = Join-Path $ProjectRoot ("results\vla\training\" + $RunName)
$LogDir = Join-Path $ProjectRoot "results\vla\training_logs"
$LogPath = Join-Path $LogDir ($RunName + ".log")

foreach ($RequiredPath in @($PythonExe, $Launcher, $AdapterCheckpoint, $DatasetRoot, $SamplingPlan)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required VLA artifact is missing: $RequiredPath"
    }
}
if (
    $Steps -lt 1 -or $SaveFreq -lt 1 -or $BatchSize -lt 1 -or
    $SchedulerWarmupSteps -lt 0 -or $SchedulerDecaySteps -lt 1 -or
    $PeakLearningRate -le 0.0 -or $DecayLearningRate -le 0.0 -or
    $DecayLearningRate -gt $PeakLearningRate -or $ZAxisWeight -le 0.0
) {
    throw "Training, scheduler, or loss parameters are invalid."
}
if (Test-Path -LiteralPath $OutputDir) {
    throw "Refusing to overwrite an existing run: $OutputDir"
}
if (Test-Path -LiteralPath $LogPath) {
    $LogPath = Join-Path $LogDir ($RunName + "_retry_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff") + ".log")
    if (Test-Path -LiteralPath $LogPath) { throw "Training log already exists: $LogPath" }
}
# Validate adapter-only policy metadata before creating the training log.
& $PythonExe $Launcher "--policy.path=$AdapterCheckpoint" --checkpoint-preflight-only
if ($LASTEXITCODE -ne 0) { throw "Input checkpoint preflight failed: $AdapterCheckpoint" }
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
if ($ChunkSize -lt 1) { throw "ChunkSize must be positive" }
if ($ActionSteps -eq 0) { $ActionSteps = $ChunkSize }
if ($ActionSteps -lt 1 -or $ActionSteps -gt $ChunkSize) { throw "ActionSteps must be between 1 and ChunkSize" }
if ($InterfaceContract) {
    $env:LEROBOT_INTERFACE_CONTRACT = [System.IO.Path]::GetFullPath($InterfaceContract)
} else { Remove-Item Env:LEROBOT_INTERFACE_CONTRACT -ErrorAction SilentlyContinue }
# A joint export carries an explicit semantic contract. This also supports a
# durable v23 driver already launched before the optional CLI parameter existed.
$DatasetJointContract = Join-Path $DatasetRoot "interface_contract.json"
if (-not $JointInterfaceContract -and (Test-Path -LiteralPath $DatasetJointContract)) {
    $DatasetContract = Get-Content $DatasetJointContract -Raw | ConvertFrom-Json
    if ($DatasetContract.version -eq "vla_joint_execution_aligned_v23") {
        $JointInterfaceContract = $DatasetJointContract
    }
}
if ($JointInterfaceContract) {
    if ($InterfaceContract -or $ZAxisWeight -ne 1 -or $LiftZAxisWeight -ne 0 -or $GripperIndex -ne 7) {
        throw "Joint labels require uniform arm weights, gripper index 7, and no relative Cartesian contract"
    }
    $env:LEROBOT_JOINT_INTERFACE_CONTRACT = [System.IO.Path]::GetFullPath($JointInterfaceContract)
    $JointContract = Get-Content $env:LEROBOT_JOINT_INTERFACE_CONTRACT -Raw | ConvertFrom-Json
    if ($JointContract.version -ne "vla_joint_execution_aligned_v23" -or $JointContract.state_dim -ne 25) {
        throw "Invalid joint interface contract"
    }
} else { Remove-Item Env:LEROBOT_JOINT_INTERFACE_CONTRACT -ErrorAction SilentlyContinue }
$env:LEROBOT_VLA_SAMPLING_PLAN = $SamplingPlan
if ($FreshLora) {
    Remove-Item Env:LEROBOT_TRAIN_EXISTING_PEFT -ErrorAction SilentlyContinue
    Remove-Item Env:LEROBOT_EXPECTED_PEFT_TRAINABLE_PARAMS -ErrorAction SilentlyContinue
    $env:LEROBOT_FRESH_LORA_AUDIT_PATH = Join-Path $LogDir ($RunName + ".initialization.json")
    if (Test-Path -LiteralPath $env:LEROBOT_FRESH_LORA_AUDIT_PATH) { throw "Initialization audit exists" }
} else {
    Remove-Item Env:LEROBOT_FRESH_LORA_AUDIT_PATH -ErrorAction SilentlyContinue
    $env:LEROBOT_TRAIN_EXISTING_PEFT = "1"
    if ($ExpectedTrainableParams -lt 1) { throw "ExpectedTrainableParams must be positive" }
    $env:LEROBOT_EXPECTED_PEFT_TRAINABLE_PARAMS = [string]$ExpectedTrainableParams
}
$env:LEROBOT_SMOLVLA_WEIGHTED_LOSS_JSON = @{
    z_axis_index = 2
    z_axis_weight = $ZAxisWeight
    sample_weight_key = "training.loss_weight"
    sample_weight_reduction = $SampleWeightReduction
    gripper_index = $(if ($GripperIndex -lt 0) { $null } else { $GripperIndex })
    gripper_weight = $GripperWeight
    lift_z_axis_weight = $(if ($LiftZAxisWeight -eq 0.0) { $null } else { $LiftZAxisWeight })
} | ConvertTo-Json -Compress
if ($LiftZAxisWeight -lt 0) { throw "LiftZAxisWeight must be nonnegative" }
if ($LiftZAxisWeight -gt 0) { $env:LEROBOT_LIFT_PHASE_METADATA = "1" }
else { Remove-Item Env:LEROBOT_LIFT_PHASE_METADATA -ErrorAction SilentlyContinue }

# Delta labels are robot-frame quantities, so use photometric augmentation
# only. RandomAffine remains explicitly disabled.
$PhotometricTfs = '{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
$env:LEROBOT_IMAGE_TRANSFORMS_TFS_JSON = $PhotometricTfs

Write-Host "SmolVLA targeted DAgger recovery fine-tuning (Windows / RTX 4060 8 GB)"
Write-Host "  initialization: $AdapterCheckpoint (fresh optimizer; fresh LoRA=$FreshLora)"
Write-Host "  adapter guard:  one default adapter, expected $ExpectedTrainableParams trainable params"
Write-Host "  dataset:        $DatasetRoot"
Write-Host "  sampling:       verified train-only plan: $SamplingPlan"
Write-Host "  loss:           dataset training.loss_weight; Z x$ZAxisWeight"
Write-Host "  optimizer:      peak=$PeakLearningRate, floor=$DecayLearningRate"
Write-Host "  schedule:       warmup=$SchedulerWarmupSteps, decay=$SchedulerDecaySteps"
Write-Host "  steps/batch:    $Steps / $BatchSize"
Write-Host "  output:         $OutputDir"
Write-Host "  log:            $LogPath"

$TrainArgs = @(
    "--policy.path=$AdapterCheckpoint",
    "--policy.input_features=null",
    "--policy.output_features=null",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--policy.chunk_size=$ChunkSize",
    "--policy.n_action_steps=$ActionSteps",
    "--policy.resize_imgs_with_padding=[320,320]",
    "--policy.freeze_vision_encoder=true",
    "--policy.train_expert_only=true",
    "--policy.train_state_proj=true",
    "--policy.use_amp=true",
    "--policy.optimizer_lr=$PeakLearningRate",
    "--policy.scheduler_warmup_steps=$SchedulerWarmupSteps",
    "--policy.scheduler_decay_steps=$SchedulerDecaySteps",
    "--policy.scheduler_decay_lr=$DecayLearningRate",
    "--dataset.repo_id=$DatasetRepoId",
    "--dataset.root=$DatasetRoot",
    "--dataset.image_transforms.enable=true",
    "--dataset.image_transforms.random_order=true",
    "--dataset.image_transforms.max_num_transforms=3",
    "--output_dir=$OutputDir",
    "--job_name=$RunName",
    "--batch_size=$BatchSize",
    "--seed=$Seed",
    "--num_workers=0",
    "--steps=$Steps",
    "--log_freq=100",
    "--eval_freq=0",
    "--save_checkpoint=true",
    "--save_freq=$SaveFreq",
    "--wandb.enable=false"
)
if ($FreshLora) {
    # LeRobot's PEFT CLI exposes rank, not alpha/dropout. Its defaults are
    # alpha=8/dropout=0; the before-training audit enforces both values.
    $TrainArgs += @("--policy.use_peft=false", "--peft.method_type=LORA", "--peft.r=16")
}

# The continued-adapter branch must not pass --peft.* (double-wrapping).
# FreshLora instead creates exactly one adapter; neither branch resumes an optimizer.
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $PythonExe $Launcher @TrainArgs 2>&1 | ForEach-Object {
    # Windows PowerShell wraps native stderr in ErrorRecord objects. Emit the
    # message as plain text and omit blank progress records from tqdm.
    if ($_ -is [System.Management.Automation.ErrorRecord]) { $Line = $_.Exception.Message }
    else { $Line = $_.ToString() }
    if (-not [string]::IsNullOrWhiteSpace($Line)) { $Line }
} | Tee-Object -FilePath $LogPath
$TrainExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($TrainExitCode -ne 0) {
    throw "DAgger-recovery SmolVLA fine-tuning failed with exit code $TrainExitCode"
}

Write-Host "Fine-tuning completed: $OutputDir\checkpoints"
Write-Host "Training log:         $LogPath"
