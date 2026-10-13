param(
    [string]$RunTag = "",
    [int]$Steps = 30000,
    [int]$Port = 8798,
    [switch]$SmokeOnly
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $RunTag) { $RunTag = (Get-Date -Format "yyyyMMdd_HHmmss_fff") + "_v22" }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $Steps -lt 1) { throw "Unsafe settings" }
$Python = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Source = Join-Path $Root "exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9"
$Export = Join-Path $Root "exports\lerobot\pick_lift_rgb_language_proprio_v22"
$Base = Join-Path $Root "results\vla\models\smolvla_base"
$Results = Join-Path $Root "results\vla\proprio_fresh_$RunTag"
$Run = "smolvla_rgb_language_proprio_fresh_lora16_$RunTag"
$Checkpoints = Join-Path $Root "results\vla\training\$Run\checkpoints"
if (Test-Path -LiteralPath $Results) { throw "Refusing to overwrite $Results" }
New-Item -ItemType Directory -Path $Results | Out-Null
$Status = Join-Path $Results "pipeline_status.json"
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
foreach ($Name in @("LEROBOT_PLATEAU_CONFIG_JSON", "LEROBOT_LOSS_PLATEAU_JSON", "LEROBOT_INTERFACE_CONTRACT",
    "LEROBOT_CHUNK_INITIAL_AUDIT_PATH", "LEROBOT_CHUNK_INITIAL_AUDIT_EXPECTED_CHUNK_SIZE",
    "LEROBOT_CHUNK_ABLATION_AUDIT", "LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK",
    "LEROBOT_LIFT_PHASE_METADATA")) { Remove-Item "Env:$Name" -ErrorAction SilentlyContinue }
function Set-Stage([string]$Stage, [string]$Message) {
    @{stage=$Stage; message=$Message; updated_at_utc=[DateTime]::UtcNow.ToString("o")
      results=$Results; training_run=$Run; dataset=$Export; source_dataset=$Source; original_base=$Base
      steps=$Steps; smoke_only=[bool]$SmokeOnly; planned_hard_episodes=$(if ($SmokeOnly) { 0 } else { 10 })
      policy_state_dim=25; action_dim=8; visual_target_in_input=$false; phase_in_input=$false
      external_target_used_for_control=$false; previous_adapter_loaded=$false
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Status -Encoding UTF8
    Write-Host "PROPRIO_PIPELINE_STAGE=$Stage $Message"
}
function Invoke-Logged([string[]]$Arguments, [string]$Log) {
    $Previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_.ToString() }
        } | Tee-Object -FilePath $Log
        $Code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Previous }
    if ($Code -ne 0) { throw "Python failed ($Code); see $Log" }
}
try {
    Set-Stage "PROJECTING_DATA" "Keep RGB/language/executed actions; remove state columns 25..29; preserve scene splits."
    $SourceAudit = Get-Content (Join-Path $Source "parquet_audit.json") -Raw | ConvertFrom-Json
    if ($SourceAudit.status -ne "EXECUTION_ALIGNED_PARQUET_AUDIT_OK") { throw "Source audit did not pass" }
    $BaseHash = (Get-FileHash (Join-Path $Base "model.safetensors") -Algorithm SHA256).Hash
    if (-not (Test-Path -LiteralPath $Export)) {
        Invoke-Logged @("scripts\export_vla_proprio_v22.py", "--source", $Source, "--output", $Export) (Join-Path $Results "export.log")
    }
    $Audit = Get-Content (Join-Path $Export "projection_audit.json") -Raw | ConvertFrom-Json
    if ($Audit.status -ne "PROPRIO25_PROJECTION_VERIFIED" -or $Audit.interface_contract.state_dim -ne 25 -or
        $Audit.interface_contract.action_dim -ne 8 -or $Audit.source_export -ne $Source) { throw "Wrong projected dataset" }
    $Plan = Join-Path $Results "train_sampling_plan.json"
    Invoke-Logged @("scripts\build_vla_sampling_plan.py", "--dataset-root", (Join-Path $Export "train"),
        "--output", $Plan, "--seed", "20261009", "--balance-levels", "--easy-weight", "1",
        "--medium-weight", "1", "--hard-weight", "1", "--phase-0-weight", "0.5",
        "--phase-1-weight", "1", "--phase-2-weight", "1.5", "--phase-3-weight", "1.5",
        "--phase-4-weight", "2", "--dagger-recovery-weight", "1") (Join-Path $Results "sampling.log")
    @{base_model_sha256=$BaseHash; input_contract=$Audit.interface_contract; training_seed=1000
      sampling_seed=20261009; evaluation_seed=20261011; train_episodes=510; train_frames=27338
      validation_episodes=45; test_episodes=45; test_used_for_training=$false
      steps=$Steps; batch_size=1; chunk_size=4; action_steps=1
      approximate_uniform_equivalent_epochs=($Steps / 27338.0)
      epoch_caveat="Weighted replacement sampling; not guaranteed traversal of every frame."
      initialization="Original SmolVLA + newly initialized rank-16 LoRA; no old adapter."
      rgb_augmentation="photometric only"; language="Pick up the red cube and lift it 10 centimeters."
      evaluation="10 fixed Hard scenes, no coordinates/phase/expert fallback; true 10 cm sustained success"
      gripper_safety="Original close latch retained; target-distance gate removed. Not a single-factor v9 comparison."
    } | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $Results "experiment_context.json") -Encoding UTF8
    Set-Stage "FRESH_FINETUNING" "Fresh original-base LoRA; robot-only 25-D state, unchanged executed 8-D delta actions."
    $Save = if ($SmokeOnly) { $Steps } else { 6000 }
    $Warmup = if ($SmokeOnly) { [Math]::Min(3, $Steps-1) } else { 500 }
    & (Join-Path $PSScriptRoot "train_smolvla_dagger_recovery_8gb_v7.ps1") -FreshLora -AdapterCheckpoint $Base `
        -DatasetRoot (Join-Path $Export "train") -DatasetRepoId "local/isaac-proprio-v22-train" -SamplingPlan $Plan `
        -RunName $Run -Steps $Steps -SaveFreq $Save -BatchSize 1 -Seed 1000 -ChunkSize 4 -ActionSteps 1 `
        -SchedulerWarmupSteps $Warmup -SchedulerDecaySteps $Steps -PeakLearningRate 0.00001 -DecayLearningRate 0.000001 `
        -ZAxisWeight 3 -GripperIndex 7 -GripperWeight 1
    $Init = Get-Content (Join-Path $Root "results\vla\training_logs\$Run.initialization.json") -Raw | ConvertFrom-Json
    if ($Init.status -ne "FRESH_LORA_INITIALIZATION_VERIFIED" -or $Init.old_adapter_loaded) { throw "Fresh initialization audit failed" }
    $Final = Join-Path $Checkpoints (("{0:D6}" -f $Steps) + "\pretrained_model")
    Invoke-Logged @("scripts\review_vla_proprio_v22.py", "--checkpoint", $Final) (Join-Path $Results "checkpoint_contract.log")
    Set-Stage "FIXED_VALIDATION" "Same fixed independent validation frames for all checkpoints; Hard scenes never select weights."
    $ScanSteps = if ($SmokeOnly) { @($Steps) } else { @(6000,12000,18000,24000,30000) }
    $Samples = if ($SmokeOnly) { 4 } else { 240 }
    $Validation = Join-Path $Results "validation_scan.json"
    $ScanArgs = @("scripts\scan_smolvla_checkpoints.py", "--checkpoint-root", $Checkpoints, "--checkpoint-steps")
    $ScanArgs += @($ScanSteps | ForEach-Object { [string]$_ })
    $ScanArgs += @("--dataset-root", (Join-Path $Export "validation"), "--repo-id", "local/isaac-proprio-v22-validation",
        "--samples", [string]$Samples, "--seed", "20261011", "--device", "cuda", "--batch-size", "1",
        "--position-target-mode", "delta", "--threshold-position-mae-cm", "2", "--output", $Validation)
    Invoke-Logged $ScanArgs (Join-Path $Results "validation.log")
    if ($SmokeOnly) { Set-Stage "SMOKE_COMPLETE" "Export, fresh initialization, 25/8 checkpoint and real RGB+language+state inference passed."; return }
    $Scan = Get-Content $Validation -Raw | ConvertFrom-Json
    $SelectedStep = [int]$Scan.best_checkpoint.step
    Set-Stage "HARD_CLOSED_LOOP" "Ten Hard diagnostic episodes with validation-selected weights; no target/phase auxiliary control."
    $BenchRoot = Join-Path $Results "hard_benchmark"
    $Bench = @{RunTag="20261011_truth_v9"; TrainingRun=$Run; DatasetExportRoot=$Export; ValidationReport=$Validation
      OutputRoot=$BenchRoot; CheckpointStep=$SelectedStep; Levels=@("hard"); EpisodesPerLevel=10
      MaxSteps=600; Port=$Port; ProprioOnly=$true; RecordPhaseDiagnostics=$true
      AllowValidationGateOverride=$true
      ValidationGateOverrideReason="User requested ten Hard tests after robot-only fresh fine-tuning; diagnostic if offline 2 cm gate fails."
    }
    & (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") @Bench
    $Selected = Join-Path $Checkpoints (("{0:D6}" -f $SelectedStep) + "\pretrained_model")
    Invoke-Logged @("scripts\review_vla_proprio_v22.py", "--checkpoint", $Selected, "--benchmark-root", $BenchRoot,
        "--output", (Join-Path $Results "hard10_summary.json")) (Join-Path $Results "hard10_summary.log")
    if ((Get-FileHash (Join-Path $Base "model.safetensors") -Algorithm SHA256).Hash -ne $BaseHash) { throw "Original base weights changed" }
    Set-Stage "COMPLETE" "Read validation_scan.json and hard10_summary.json; no previous weights or datasets overwritten."
} catch {
    Set-Stage "FAILED" $_.Exception.Message
    throw
}
