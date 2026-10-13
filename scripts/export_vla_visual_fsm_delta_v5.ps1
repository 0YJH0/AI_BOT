param(
    [string]$OutputRoot = "",
    [string]$SamplingPlan = "",
    [double]$LiftLossMultiplier = 1.5,
    [double]$HardLossMultiplier = 1.25,
    [int]$Seed = 20261008,
    [int]$SamplingSeed = 20261009
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Exporter = Join-Path $PSScriptRoot "export_balanced_lerobot_vla_images_v044.py"
$PlanBuilder = Join-Path $PSScriptRoot "build_vla_sampling_plan.py"
$VisualCache = Join-Path $ProjectRoot "results\vla\visual_state_cache_balanced504_causal_clean_v3"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "exports\lerobot\pick_lift_vla_visual_fsm_delta_weighted_causal_clean_504_v5"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot $OutputRoot
}
if ([string]::IsNullOrWhiteSpace($SamplingPlan)) {
    $SamplingPlan = Join-Path $ProjectRoot "results\vla\sampling_plans\smolvla_visual_fsm_delta_weighted_v5_train.json"
} elseif (-not [System.IO.Path]::IsPathRooted($SamplingPlan)) {
    $SamplingPlan = Join-Path $ProjectRoot $SamplingPlan
}

foreach ($RequiredPath in @($PythonExe, $Exporter, $PlanBuilder, $VisualCache)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required VLA artifact is missing: $RequiredPath"
    }
}
if ($LiftLossMultiplier -le 0.0 -or $HardLossMultiplier -le 0.0) {
    throw "Loss multipliers must be positive."
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to overwrite existing v5 dataset: $OutputRoot"
}
if (Test-Path -LiteralPath $SamplingPlan) {
    throw "Refusing to overwrite existing sampling plan: $SamplingPlan"
}

$env:PYTHONUTF8 = "1"

Write-Host "Exporting independent v5 delta-XYZ VLA dataset"
Write-Host "  output:             $OutputRoot"
Write-Host "  delta bound:        0.04 m"
Write-Host "  Hard loss weight:   $HardLossMultiplier"
Write-Host "  LIFT loss weight:   $LiftLossMultiplier"
Write-Host "  policy state:       30-D (difficulty stays training-only)"

$ExportArgs = @(
    "--easy-source", (Join-Path $ProjectRoot "dataset\vla_camera_fix_clean\easy"),
    "--easy-source", (Join-Path $ProjectRoot "dataset\vla_camera_fix_pilot\easy"),
    "--medium-source", (Join-Path $ProjectRoot "dataset\vla_balanced_504\medium"),
    "--medium-source", (Join-Path $ProjectRoot "dataset\vla_camera_fix_clean\medium"),
    "--hard-source", (Join-Path $ProjectRoot "dataset\vla_camera_fix_clean\hard"),
    "--output-root", $OutputRoot,
    "--repo-prefix", "local/isaac-pick-lift-vla-visual-fsm-delta-weighted-causal-clean-504-v5",
    "--episodes-per-level", "168",
    "--train-per-level", "134",
    "--validation-per-level", "17",
    "--test-per-level", "17",
    "--seed", "$Seed",
    "--fps", "10",
    "--action-offset-steps", "1",
    "--action-target-mode", "visual_fsm_bounded_delta",
    "--maximum-position-step-m", "0.04",
    "--lift-loss-multiplier", "$LiftLossMultiplier",
    "--hard-loss-multiplier", "$HardLossMultiplier",
    "--minimum-rgb-std", "30",
    "--minimum-rgb-range", "128",
    "--rgb-quality-frames", "1000",
    "--visual-state-cache", $VisualCache
)
& $PythonExe $Exporter @ExportArgs
if ($LASTEXITCODE -ne 0) {
    throw "v5 delta-XYZ dataset export failed with exit code $LASTEXITCODE"
}

Write-Host "Building train-only Hard/LIFT sampling plan"
$PlanArgs = @(
    "--dataset-root", (Join-Path $OutputRoot "train"),
    "--output", $SamplingPlan,
    "--seed", "$SamplingSeed",
    "--easy-weight", "1.0",
    "--medium-weight", "1.0",
    "--hard-weight", "2.0",
    "--phase-0-weight", "1.0",
    "--phase-1-weight", "1.0",
    "--phase-2-weight", "1.25",
    "--phase-3-weight", "1.5",
    "--phase-4-weight", "3.0"
)
& $PythonExe $PlanBuilder @PlanArgs
if ($LASTEXITCODE -ne 0) {
    throw "v5 sampling-plan build failed with exit code $LASTEXITCODE"
}

Write-Host "v5 dataset and sampling plan completed."
Write-Host "  dataset: $OutputRoot"
Write-Host "  plan:    $SamplingPlan"
