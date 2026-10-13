param(
    [int]$Steps = 5000,
    [int]$SaveFreq = 500,
    [string]$RunName = "smolvla_lora_r8_v1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$WindowsTrainLauncher = Join-Path $PSScriptRoot "lerobot_train_windows.py"
$ModelPath = Join-Path $ProjectRoot "results\vla\models\smolvla_base"
$DatasetRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_images_baseline_v1"
$OutputDir = Join-Path $ProjectRoot ("results\vla\training\" + $RunName)

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "VLA environment is missing. Expected: $PythonExe"
}
if (-not (Test-Path -LiteralPath $WindowsTrainLauncher)) {
    throw "Windows LeRobot launcher is missing. Expected: $WindowsTrainLauncher"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "SmolVLA base model is missing. Expected: $ModelPath"
}
if (-not (Test-Path -LiteralPath $DatasetRoot)) {
    throw "LeRobot dataset is missing. Expected: $DatasetRoot"
}

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"

Write-Host "SmolVLA LoRA training (RTX 4060 8 GB / Windows profile)"
Write-Host "  steps:       $Steps"
Write-Host "  save every:  $SaveFreq"
Write-Host "  output:      $OutputDir"

& $PythonExe $WindowsTrainLauncher `
    --policy.path="$ModelPath" `
    --policy.input_features=null `
    --policy.output_features=null `
    --policy.device=cuda `
    --policy.push_to_hub=false `
    --policy.chunk_size=8 `
    --policy.n_action_steps=8 `
    --policy.freeze_vision_encoder=true `
    --policy.train_expert_only=true `
    --policy.train_state_proj=true `
    --policy.use_amp=true `
    --dataset.repo_id=local/isaac-pick-lift-vla-images-baseline-v1 `
    --dataset.root="$DatasetRoot" `
    --output_dir="$OutputDir" `
    --job_name="$RunName" `
    --batch_size=1 `
    --num_workers=0 `
    --steps=$Steps `
    --log_freq=10 `
    --eval_freq=0 `
    --save_checkpoint=true `
    --save_freq=$SaveFreq `
    --wandb.enable=false `
    --peft.method_type=LORA `
    --peft.r=8

if ($LASTEXITCODE -ne 0) {
    throw "SmolVLA training failed with exit code $LASTEXITCODE"
}

Write-Host "Training completed. Checkpoints: $OutputDir\checkpoints"
