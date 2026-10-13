param(
    [int]$Steps = 30000,
    [int]$SaveFreq = 2500,
    [int]$LoraRank = 16,
    [int]$BatchSize = 2,
    [int]$SchedulerWarmupSteps = 1000,
    [int]$SchedulerDecaySteps = 0,
    [double]$ZAxisWeight = 3.0,
    [string]$RunName = "smolvla_visual_fsm_delta_weighted_lora16_v5",
    [string]$DatasetRoot = "",
    [string]$SamplingPlan = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Launcher = Join-Path $PSScriptRoot "lerobot_train_windows.py"
$ModelPath = Join-Path $ProjectRoot "results\vla\models\smolvla_base"
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_visual_fsm_delta_weighted_causal_clean_504_v5\train"
} elseif (-not [System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot $DatasetRoot
}
if ([string]::IsNullOrWhiteSpace($SamplingPlan)) {
    $SamplingPlan = Join-Path $ProjectRoot "results\vla\sampling_plans\smolvla_visual_fsm_delta_weighted_v5_train.json"
} elseif (-not [System.IO.Path]::IsPathRooted($SamplingPlan)) {
    $SamplingPlan = Join-Path $ProjectRoot $SamplingPlan
}
$OutputDir = Join-Path $ProjectRoot ("results\vla\training\" + $RunName)
$LogDir = Join-Path $ProjectRoot "results\vla\training_logs"
$LogPath = Join-Path $LogDir ($RunName + ".log")

foreach ($RequiredPath in @($PythonExe, $Launcher, $ModelPath, $DatasetRoot, $SamplingPlan)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required VLA artifact is missing: $RequiredPath"
    }
}
if ($SchedulerDecaySteps -eq 0) {
    $SchedulerDecaySteps = $Steps
}
if (
    $Steps -lt 1 -or $SaveFreq -lt 1 -or $LoraRank -lt 1 -or
    $BatchSize -lt 1 -or $SchedulerWarmupSteps -lt 0 -or
    $SchedulerDecaySteps -lt 1 -or $ZAxisWeight -le 0.0
) {
    throw "Training, scheduler, or loss parameters are invalid."
}
if (Test-Path -LiteralPath $OutputDir) {
    throw "Refusing to overwrite an existing run: $OutputDir"
}
if (Test-Path -LiteralPath $LogPath) {
    throw "Refusing to overwrite an existing training log: $LogPath"
}
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:LEROBOT_VLA_SAMPLING_PLAN = $SamplingPlan
$env:LEROBOT_SMOLVLA_WEIGHTED_LOSS_JSON = @{
    z_axis_index = 2
    z_axis_weight = $ZAxisWeight
    sample_weight_key = "training.loss_weight"
} | ConvertTo-Json -Compress

# Delta labels are robot-frame quantities, so geometric image transforms would
# invalidate their direction. Keep only photometric augmentation.
$PhotometricTfs = '{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
$env:LEROBOT_IMAGE_TRANSFORMS_TFS_JSON = $PhotometricTfs

Write-Host "SmolVLA delta-XYZ weighted training (Windows / RTX 4060 8 GB)"
Write-Host "  initialization: fresh LoRA from local SmolVLA base"
Write-Host "  position label: current-EEF-relative delta xyz, max 4 cm"
Write-Host "  loss:           real 8 action dims; Z x$ZAxisWeight"
Write-Host "  sample loss:    dataset Hard x1.25, LIFT x1.5"
Write-Host "  sampling:       verified train-only Hard/LIFT plan"
Write-Host "  chunk/horizon:  4 / 4"
Write-Host "  LoRA/batch:     $LoraRank / $BatchSize"
Write-Host "  steps:          $Steps"
Write-Host "  scheduler:      warmup=$SchedulerWarmupSteps, decay=$SchedulerDecaySteps"
Write-Host "  output:         $OutputDir"
Write-Host "  persistent log: $LogPath"

$TrainArgs = @(
    "--policy.path=$ModelPath",
    "--policy.input_features=null",
    "--policy.output_features=null",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--policy.chunk_size=4",
    "--policy.n_action_steps=4",
    "--policy.resize_imgs_with_padding=[320,320]",
    "--policy.freeze_vision_encoder=true",
    "--policy.train_expert_only=true",
    "--policy.train_state_proj=true",
    "--policy.use_amp=true",
    "--policy.scheduler_warmup_steps=$SchedulerWarmupSteps",
    "--policy.scheduler_decay_steps=$SchedulerDecaySteps",
    "--dataset.repo_id=local/isaac-pick-lift-vla-visual-fsm-delta-weighted-causal-clean-504-v5-train",
    "--dataset.root=$DatasetRoot",
    "--dataset.image_transforms.enable=true",
    "--dataset.image_transforms.random_order=true",
    "--dataset.image_transforms.max_num_transforms=3",
    "--output_dir=$OutputDir",
    "--job_name=$RunName",
    "--batch_size=$BatchSize",
    "--num_workers=0",
    "--steps=$Steps",
    "--log_freq=100",
    "--eval_freq=0",
    "--save_checkpoint=true",
    "--save_freq=$SaveFreq",
    "--wandb.enable=false",
    "--peft.method_type=LORA",
    "--peft.r=$LoraRank"
)
# Windows PowerShell turns ordinary native stderr lines (for example the
# optional torchcodec warning) into NativeCommandError records.  Temporarily
# keep those records non-terminating while preserving the real process exit
# code below.
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $PythonExe $Launcher @TrainArgs 2>&1 | Tee-Object -FilePath $LogPath
$TrainExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($TrainExitCode -ne 0) {
    throw "Delta-weighted SmolVLA training failed with exit code $TrainExitCode"
}

Write-Host "Training completed: $OutputDir\checkpoints"
Write-Host "Training log:       $LogPath"
