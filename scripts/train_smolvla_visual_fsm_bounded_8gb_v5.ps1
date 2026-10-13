param(
    [int]$Steps = 60000,
    [int]$SaveFreq = 2500,
    [int]$LoraRank = 16,
    [int]$BatchSize = 1,
    [int]$SchedulerWarmupSteps = 1000,
    [int]$SchedulerDecaySteps = 0,
    [string]$RunName = "smolvla_visual_fsm_bounded_lora16_v4",
    [string]$DatasetRoot = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Launcher = Join-Path $PSScriptRoot "lerobot_train_windows.py"
$ModelPath = Join-Path $ProjectRoot "results\vla\models\smolvla_base"
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_visual_fsm_bounded_causal_clean_504_v4\train"
} elseif (-not [System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot $DatasetRoot
}
$OutputDir = Join-Path $ProjectRoot ("results\vla\training\" + $RunName)

foreach ($RequiredPath in @($PythonExe, $Launcher, $ModelPath, $DatasetRoot)) {
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
    $SchedulerDecaySteps -lt 1
) {
    throw "Training and scheduler parameters are invalid."
}
if (Test-Path -LiteralPath $OutputDir) {
    throw "Refusing to overwrite an existing run: $OutputDir"
}

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"

# Absolute robot-frame action labels must not be combined with untracked image
# translations or rotations. LeRobot 0.4.4 requires overriding the complete
# transform dictionary, so retain photometric augmentation and disable affine.
$PhotometricTfs = '{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
$env:LEROBOT_IMAGE_TRANSFORMS_TFS_JSON = $PhotometricTfs

Write-Host "SmolVLA visual + bounded-FSM training (Windows / RTX 4060 8 GB)"
Write-Host "  initialization: fresh LoRA from the local SmolVLA base (no PEFT resume)"
Write-Host "  dataset:       visual/FSM/bounded causal v4 train split"
Write-Host "  augmentation:  photometric only; RandomAffine disabled"
Write-Host "  action chunk:  4"
Write-Host "  horizon:       4"
Write-Host "  vision encoder: frozen"
Write-Host "  LoRA rank:     $LoraRank"
Write-Host "  batch size:    $BatchSize"
Write-Host "  steps:         $Steps"
Write-Host "  scheduler:     warmup=$SchedulerWarmupSteps, decay=$SchedulerDecaySteps"
Write-Host "  save every:    $SaveFreq"
Write-Host "  output:        $OutputDir"

# Deliberately use --policy.path on the immutable base model and a new output
# directory. Do not pass --resume/config_path: LeRobot 0.4.4 can double-wrap a
# loaded PEFT adapter and reinitialize its LoRA matrices during resume.
& $PythonExe $Launcher `
    --policy.path="$ModelPath" `
    --policy.input_features=null `
    --policy.output_features=null `
    --policy.device=cuda `
    --policy.push_to_hub=false `
    --policy.chunk_size=4 `
    --policy.n_action_steps=4 `
    --policy.resize_imgs_with_padding=[320,320] `
    --policy.freeze_vision_encoder=true `
    --policy.train_expert_only=true `
    --policy.train_state_proj=true `
    --policy.use_amp=true `
    --policy.scheduler_warmup_steps=$SchedulerWarmupSteps `
    --policy.scheduler_decay_steps=$SchedulerDecaySteps `
    --dataset.repo_id=local/isaac-pick-lift-vla-visual-fsm-bounded-causal-clean-504-v4-train `
    --dataset.root="$DatasetRoot" `
    --dataset.image_transforms.enable=true `
    --dataset.image_transforms.random_order=true `
    --dataset.image_transforms.max_num_transforms=3 `
    --output_dir="$OutputDir" `
    --job_name="$RunName" `
    --batch_size=$BatchSize `
    --num_workers=0 `
    --steps=$Steps `
    --log_freq=100 `
    --eval_freq=0 `
    --save_checkpoint=true `
    --save_freq=$SaveFreq `
    --wandb.enable=false `
    --peft.method_type=LORA `
    --peft.r=$LoraRank

if ($LASTEXITCODE -ne 0) {
    throw "Visual/FSM/bounded SmolVLA training failed with exit code $LASTEXITCODE"
}

Write-Host "Training completed: $OutputDir\checkpoints"
