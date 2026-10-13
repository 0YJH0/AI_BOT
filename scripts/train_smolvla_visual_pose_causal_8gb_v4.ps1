param(
    [int]$Steps = 60000,
    [int]$SaveFreq = 2500,
    [int]$LoraRank = 16,
    [string]$RunName = "smolvla_visual_pose_causal_clean_lora16_v3",
    [string]$DatasetRoot = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Launcher = Join-Path $PSScriptRoot "lerobot_train_windows.py"
$ModelPath = Join-Path $ProjectRoot "results\vla\models\smolvla_base"
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_visual_pose_causal_clean_504_v3\train"
} elseif (-not [System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetRoot = Join-Path $ProjectRoot $DatasetRoot
}
$OutputDir = Join-Path $ProjectRoot ("results\vla\training\" + $RunName)

foreach ($RequiredPath in @($PythonExe, $Launcher, $ModelPath, $DatasetRoot)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required VLA artifact is missing: $RequiredPath"
    }
}
if ($Steps -lt 1 -or $SaveFreq -lt 1 -or $LoraRank -lt 1) {
    throw "Steps, SaveFreq, and LoraRank must be positive."
}

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"

# Absolute robot-frame action labels must not be combined with untracked image
# translations/rotations. Keep only photometric augmentation and explicitly set
# affine weight to zero. LeRobot 0.4.4 requires overriding the complete dict.
$PhotometricTfs = '{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
$env:LEROBOT_IMAGE_TRANSFORMS_TFS_JSON = $PhotometricTfs

Write-Host "SmolVLA causal + RGB-D-pose training (Windows / RTX 4060 8 GB)"
Write-Host "  dataset:       402 train episodes, balanced Easy/Medium/Hard"
Write-Host "  state:         25-D proprioception + RGB-D xyz/validity (29-D)"
Write-Host "  labels:        causal observation(t) -> action(t+1)"
Write-Host "  augmentation:  photometric only; RandomAffine disabled"
Write-Host "  action chunk:  4"
Write-Host "  LoRA rank:     $LoraRank"
Write-Host "  steps:         $Steps"
Write-Host "  save every:    $SaveFreq"
Write-Host "  output:        $OutputDir"

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
    --dataset.repo_id=local/isaac-pick-lift-vla-visual-pose-causal-clean-504-v3-train `
    --dataset.root="$DatasetRoot" `
    --dataset.image_transforms.enable=true `
    --dataset.image_transforms.random_order=true `
    --dataset.image_transforms.max_num_transforms=3 `
    --output_dir="$OutputDir" `
    --job_name="$RunName" `
    --batch_size=1 `
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
    throw "Visual-conditioned causal SmolVLA training failed with exit code $LASTEXITCODE"
}

Write-Host "Training completed: $OutputDir\checkpoints"
