param(
    [int]$Steps = 30000,
    [int]$SaveFreq = 2500,
    [int]$LoraRank = 16,
    [string]$RunName = "smolvla_balanced504_aug_lora16_v1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Launcher = Join-Path $PSScriptRoot "lerobot_train_windows.py"
$ModelPath = Join-Path $ProjectRoot "results\vla\models\smolvla_base"
$DatasetRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_balanced_504_v1\train"
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

Write-Host "SmolVLA balanced-data training (Windows / RTX 4060 8 GB)"
Write-Host "  dataset:     402 train episodes (134 Easy / 134 Medium / 134 Hard)"
Write-Host "  augmentation: enabled, up to 3 transforms in random order"
Write-Host "  resize:      320 x 320"
Write-Host "  LoRA rank:   $LoraRank"
Write-Host "  steps:       $Steps"
Write-Host "  save every:  $SaveFreq"
Write-Host "  output:      $OutputDir"

& $PythonExe $Launcher `
    --policy.path="$ModelPath" `
    --policy.input_features=null `
    --policy.output_features=null `
    --policy.device=cuda `
    --policy.push_to_hub=false `
    --policy.chunk_size=8 `
    --policy.n_action_steps=8 `
    --policy.resize_imgs_with_padding=[320,320] `
    --policy.freeze_vision_encoder=true `
    --policy.train_expert_only=true `
    --policy.train_state_proj=true `
    --policy.use_amp=true `
    --dataset.repo_id=local/isaac-pick-lift-vla-balanced-504-v1-train `
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
    throw "Balanced SmolVLA training failed with exit code $LASTEXITCODE"
}

Write-Host "Balanced training completed: $OutputDir\checkpoints"
