param(
    [int]$Steps = 6000,
    [int]$SaveFreq = 500,
    [string]$RunName = "smolvla_visual_fsm_delta_tight_dagger15mm_lora16_v7",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AuditPath = Join-Path $ProjectRoot "results\vla\dagger_recovery_v7\data_audit.json"
if (-not (Test-Path -LiteralPath $AuditPath)) {
    throw "Run the strict recovery audit before training: $AuditPath"
}
$Audit = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
if ($Audit.status -ne "VLA_DAGGER_RECOVERY_AUDIT_OK" -or $Audit.frames -ne 1440 -or $Audit.learner_visited_frames -ne 360) {
    throw "Expected the audited 1440-frame / 360-learner-state tight recovery round."
}
$DatasetRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_visual_fsm_delta_tight_dagger15mm_v7\train"
$SamplingPlan = Join-Path $ProjectRoot "results\vla\sampling_plans\v7_train_tight_dagger15mm_sampling_plan.json"
$AdapterCheckpoint = Join-Path $ProjectRoot "results\vla\training\smolvla_visual_fsm_delta_dagger15mm_lora16_v6\checkpoints\006000\pretrained_model"
foreach ($RequiredPath in @($DatasetRoot, $SamplingPlan, $AdapterCheckpoint)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) { throw "Missing artifact: $RequiredPath" }
}
$ManifestPath = Join-Path $DatasetRoot "balanced_split_manifest.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$Plan = Get-Content -LiteralPath $SamplingPlan -Raw | ConvertFrom-Json
if ($Manifest.split -ne "train" -or $Manifest.frames -ne 11706 -or $Manifest.episodes -ne 852) {
    throw "Expected 11706 train frames / 852 episodes; validation must remain separate."
}
if ($Plan.num_frames -ne $Manifest.frames -or $Plan.dataset_root -ne $DatasetRoot -or
    $Plan.dataset_manifest_sha256 -ne (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()) {
    throw "Sampling plan does not match the immutable training manifest."
}
if ($Steps -lt 1 -or $SaveFreq -lt 1) { throw "Steps and SaveFreq must be positive." }
if ($ValidateOnly) {
    & "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe" (Join-Path $PSScriptRoot "lerobot_train_windows.py") "--policy.path=$AdapterCheckpoint" --checkpoint-preflight-only
    if ($LASTEXITCODE -ne 0) { throw "Input checkpoint preflight failed: $AdapterCheckpoint" }
    Write-Host "TIGHT_RECOVERY_TRAINING_PREFLIGHT_OK: 11706 frames / 852 episodes; v6 checkpoint 006000; $Steps steps, save every $SaveFreq."
    return
}

# Continue the existing v6 best adapter with a fresh optimizer. The original
# replay and broad v6 recovery are retained in the merged train-only dataset.
# Learner states already carry x2 extra loss weight, so the sampler adds no
# additional learner-only multiplier.
& (Join-Path $PSScriptRoot "train_smolvla_dagger_recovery_8gb_v7.ps1") `
    -Steps $Steps -SaveFreq $SaveFreq -RunName $RunName `
    -SchedulerWarmupSteps 300 -SchedulerDecaySteps $Steps `
    -PeakLearningRate 0.000005 -DecayLearningRate 0.0000005 `
    -DatasetRoot $DatasetRoot `
    -DatasetRepoId "local/isaac-pick-lift-vla-visual-fsm-delta-tight-dagger15mm-v7-train" `
    -SamplingPlan $SamplingPlan `
    -AdapterCheckpoint $AdapterCheckpoint
