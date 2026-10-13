param(
    [int]$TotalSteps = 30000,
    [int]$Port = 8775
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

if ($TotalSteps -lt 30000) {
    throw "TotalSteps must be at least 30000 for the one-equivalent-epoch experiment."
}

$RunTag = "20261011_v13"
$TrainingRun = "smolvla_relative4d_fresh_lora16_$RunTag"
$Results = Join-Path $Root "results\vla\relative4d_$RunTag"
$TrainingRoot = Join-Path $Root "results\vla\training\$TrainingRun"
$CheckpointRoot = Join-Path $TrainingRoot "checkpoints"
$Export = Join-Path $Root "exports\lerobot\pick_lift_relative4d_$RunTag"
$ValidationRoot = Join-Path $Export "validation"
$Plan = Join-Path $Results "train_sampling_plan.json"
$Contract = Join-Path $Export "train\interface_contract.json"
$VlaPython = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Launcher = Join-Path $PSScriptRoot "lerobot_train_windows.py"
$TrainingLog = Join-Path $Root "results\vla\training_logs\${TrainingRun}_resume_to_${TotalSteps}.log"
$ValidationReport = Join-Path $Results "validation_scan_${TotalSteps}.json"
$ValidationLog = Join-Path $Results "validation_scan_${TotalSteps}.log"

function Set-Stage([string]$Stage, [string]$Message = "", [int]$ResumeFrom = 0) {
    $Payload = @{
        stage = $Stage
        message = $Message
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        results = $Results
        training_run = $TrainingRun
        resume_from_step = $ResumeFrom
        total_steps = $TotalSteps
        train_frames = 27338
        nominal_equivalent_epochs = $TotalSteps / 27338.0
        state_dim = 29
        action_dim = 4
        device = "cuda"
        use_amp = $true
        checkpoint_scan_steps = @(12000, 18000, 24000, 30000)
        validation_report = $ValidationReport
        closed_loop_started = $false
        scheduler_note = "Optimizer/RNG state restored; cosine horizon extended to TotalSteps. The first 6000-step run had already reached its old LR floor."
    } | ConvertTo-Json -Depth 5
    $Payload | Set-Content -LiteralPath (Join-Path $Results "pipeline_status_30k.json") -Encoding UTF8
    $Payload | Set-Content -LiteralPath (Join-Path $Results "pipeline_status.json") -Encoding UTF8
    Write-Output "RELATIVE4D_30K_STAGE=$Stage $Message"
}

function Invoke-Logged([string]$Exe, [string[]]$ArgsList, [string]$Log) {
    $OldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Exe @ArgsList 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message }
        else { $_.ToString() }
    } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Tee-Object -FilePath $Log
    $Code = $LASTEXITCODE
    $ErrorActionPreference = $OldPreference
    if ($Code -ne 0) { throw "Stage failed with code $Code; see $Log" }
}

try {
    foreach ($Required in @($VlaPython, $Launcher, $Results, $TrainingRoot, $CheckpointRoot, $ValidationRoot, $Plan, $Contract)) {
        if (-not (Test-Path -LiteralPath $Required)) { throw "Required artifact is missing: $Required" }
    }

    $NumericCheckpoints = Get-ChildItem -LiteralPath $CheckpointRoot -Directory |
        Where-Object { $_.Name -match '^\d{6,}$' } |
        Sort-Object { [int]$_.Name }
    if (-not $NumericCheckpoints) { throw "No numeric checkpoints found in $CheckpointRoot" }
    $Latest = $NumericCheckpoints[-1]
    $ResumeFrom = [int]$Latest.Name
    if ($ResumeFrom -ge $TotalSteps) { throw "Latest checkpoint $ResumeFrom is already at or beyond TotalSteps=$TotalSteps" }

    $Pretrained = Join-Path $Latest.FullName "pretrained_model"
    $ConfigPath = Join-Path $Pretrained "train_config.json"
    $TrainingState = Join-Path $Latest.FullName "training_state"
    foreach ($Required in @(
        $ConfigPath,
        (Join-Path $Pretrained "adapter_model.safetensors"),
        (Join-Path $Pretrained "interface_contract.json"),
        (Join-Path $TrainingState "optimizer_state.safetensors"),
        (Join-Path $TrainingState "scheduler_state.json"),
        (Join-Path $TrainingState "rng_state.safetensors"),
        (Join-Path $TrainingState "training_step.json")
    )) {
        if (-not (Test-Path -LiteralPath $Required)) { throw "Resume checkpoint is incomplete: $Required" }
    }

    $StepState = Get-Content -LiteralPath (Join-Path $TrainingState "training_step.json") -Raw | ConvertFrom-Json
    if ([int]$StepState.step -ne $ResumeFrom) { throw "Checkpoint directory and training step disagree" }
    $PolicyConfig = Get-Content -LiteralPath (Join-Path $Pretrained "config.json") -Raw | ConvertFrom-Json
    if ($PolicyConfig.input_features.'observation.state'.shape[0] -ne 29 -or
        $PolicyConfig.output_features.action.shape[0] -ne 4 -or
        $PolicyConfig.chunk_size -ne 1) {
        throw "Resume checkpoint is not the expected relative4d chunk-1 policy"
    }

    if (-not (Test-Path -LiteralPath (Join-Path $Results "pipeline_status_before_30k.json"))) {
        Copy-Item -LiteralPath (Join-Path $Results "pipeline_status.json") -Destination (Join-Path $Results "pipeline_status_before_30k.json")
    }
    if ((Test-Path -LiteralPath (Join-Path $Results "validation_scan.json")) -and
        -not (Test-Path -LiteralPath (Join-Path $Results "validation_scan_6000.json"))) {
        Copy-Item -LiteralPath (Join-Path $Results "validation_scan.json") -Destination (Join-Path $Results "validation_scan_6000.json")
    }
    if (Test-Path -LiteralPath $TrainingLog) { throw "Training log already exists: $TrainingLog" }
    if (Test-Path -LiteralPath $ValidationReport) { throw "Validation report already exists: $ValidationReport" }

    $env:PYTHONPATH = Join-Path $Root "src"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONUNBUFFERED = "1"
    $env:HF_HUB_OFFLINE = "1"
    $env:LEROBOT_INTERFACE_CONTRACT = $Contract
    $env:LEROBOT_VLA_SAMPLING_PLAN = $Plan
    $env:LEROBOT_TRAIN_EXISTING_PEFT = "1"
    $env:LEROBOT_EXPECTED_PEFT_TRAINABLE_PARAMS = "742656"
    Remove-Item Env:LEROBOT_FRESH_LORA_AUDIT_PATH -ErrorAction SilentlyContinue
    $env:LEROBOT_SMOLVLA_WEIGHTED_LOSS_JSON = @{
        z_axis_index = 2
        z_axis_weight = 3.0
        sample_weight_key = "training.loss_weight"
        sample_weight_reduction = "per_sample"
    } | ConvertTo-Json -Compress
    $env:LEROBOT_IMAGE_TRANSFORMS_TFS_JSON = '{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'

    Set-Stage "RESUMING_GPU_TRAINING" "Restore optimizer/RNG at step $ResumeFrom and train to $TotalSteps." $ResumeFrom
    Invoke-Logged $VlaPython @(
        $Launcher,
        "--config_path=$ConfigPath",
        "--resume=true",
        "--steps=$TotalSteps",
        "--peft=null",
        "--scheduler.num_decay_steps=$TotalSteps"
    ) $TrainingLog

    foreach ($Step in @(12000, 18000, 24000, 30000)) {
        $Checkpoint = Join-Path $CheckpointRoot ("{0:D6}" -f $Step)
        if (-not (Test-Path -LiteralPath (Join-Path $Checkpoint "pretrained_model\interface_contract.json"))) {
            throw "Required scan checkpoint is incomplete: $Checkpoint"
        }
    }

    Set-Stage "SCANNING_FIXED_VALIDATION" "Scan 12k/18k/24k/30k on the same 240 held-out frames." $ResumeFrom
    Invoke-Logged $VlaPython @(
        "scripts\scan_smolvla_checkpoints.py",
        "--checkpoint-root", $CheckpointRoot,
        "--checkpoint-steps", "12000", "18000", "24000", "30000",
        "--dataset-root", $ValidationRoot,
        "--repo-id", "local/isaac-relative4d-v13-validation",
        "--samples", "240",
        "--seed", "20261011",
        "--device", "cuda",
        "--batch-size", "2",
        "--position-target-mode", "delta",
        "--threshold-position-mae-cm", "2",
        "--output", $ValidationReport
    ) $ValidationLog

    $Scan = Get-Content -LiteralPath $ValidationReport -Raw | ConvertFrom-Json
    if ($Scan.threshold_passed) {
        Set-Stage "EXTENDED_VALIDATION_GATE_PASSED" "At least one extended checkpoint is below 2 cm; closed loop intentionally awaits review." $ResumeFrom
    } else {
        Set-Stage "EXTENDED_VALIDATION_GATE_NOT_PASSED" "All extended checkpoints remain at or above 2 cm; no closed-loop run started." $ResumeFrom
    }
} catch {
    Set-Stage "EXTENDED_RUN_FAILED" $_.Exception.Message $ResumeFrom
    throw
}
