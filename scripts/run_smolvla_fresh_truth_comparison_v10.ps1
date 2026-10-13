param([string]$RunTag = "20261011_v10", [int]$EpisodesPerLevel = 3, [int]$Port = 8768)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $EpisodesPerLevel -lt 1) { throw "Unsafe settings" }
$Python = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$DataTag = "20261011_truth_v9"
$Export = Join-Path $Root "exports\lerobot\pick_lift_execution_aligned_$DataTag"
$Plan = Join-Path $Root "results\vla\execution_aligned_$DataTag\train_sampling_plan.json"
$Base = Join-Path $Root "results\vla\models\smolvla_base"
$ContinuedRun = "smolvla_execution_aligned_full_lora16_$DataTag"
$FreshRun = "smolvla_execution_aligned_fresh_lora16_$RunTag"
$Results = Join-Path $Root "results\vla\fresh_lora_comparison_$RunTag"
$ContinuedFinal = Join-Path $Root "results\vla\training\$ContinuedRun\checkpoints\006000\pretrained_model"
$FreshFinal = Join-Path $Root "results\vla\training\$FreshRun\checkpoints\006000\pretrained_model"
foreach ($Existing in @($Results,(Join-Path $Root "results\vla\training\$FreshRun"))) {
    if (Test-Path -LiteralPath $Existing) { throw "Refusing to overwrite $Existing" }
}
foreach ($Required in @($Python,$Plan,(Join-Path $Base "model.safetensors"),(Join-Path $ContinuedFinal "adapter_model.safetensors"))) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing $Required" }
}
$Audit = Get-Content -LiteralPath (Join-Path $Export "parquet_audit.json") -Raw | ConvertFrom-Json
if ($Audit.status -ne "EXECUTION_ALIGNED_PARQUET_AUDIT_OK" -or -not $Audit.scene_groups_disjoint -or
    -not $Audit.no_posthoc_action_relabeling -or -not $Audit.teacher_coordinates_excluded_from_student_input -or
    $Audit.splits.train.episodes -ne 510 -or $Audit.splits.train.frames -ne 27338) { throw "Validated truth-v9 dataset is required" }
foreach ($Split in @("train","validation","test")) {
    if ($Audit.splits.$Split.maximum_state_error -ne 0 -or $Audit.splits.$Split.maximum_action_error -ne 0) { throw "Label audit failed: $Split" }
}
New-Item -ItemType Directory -Path $Results | Out-Null
$StatusPath = Join-Path $Results "pipeline_status.json"
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
function Set-Stage([string]$Stage, [string]$Message = "") {
    @{
        stage=$Stage; message=$Message; updated_at_utc=[DateTime]::UtcNow.ToString("o")
        fresh_training_run=$FreshRun; continued_training_run=$ContinuedRun
        dataset=$Export; results=$Results; steps=6000; episodes_per_level=$EpisodesPerLevel
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    Write-Host "COMPARISON_STAGE=$Stage $Message"
}
function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$Log) {
    $Previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Exe @Arguments 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_.ToString() }
        } | Tee-Object -FilePath $Log
        $Code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Previous }
    if ($Code -ne 0) { throw "$Exe failed ($Code); see $Log" }
}
$Protected = @((Join-Path $Base "model.safetensors"),(Join-Path $ContinuedFinal "adapter_model.safetensors"),$Plan,(Join-Path $Export "parquet_audit.json"))
foreach ($Split in @("train","validation","test")) { $Protected += Join-Path $Export "$Split\balanced_split_manifest.json" }
$Hashes = @($Protected | ForEach-Object { @{path=$_; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash} })
@{
    protected_artifacts=$Hashes; dataset_audit=$Audit; training_seed=1000; evaluation_seed=20261011
    checkpoint_steps=@(500,3000,6000); primary_checkpoint_step=6000; test_split_used=$false; validation_frames=240
    fresh_base=$Base; old_adapter_initialized_from="v7/checkpoints/001500"
    note="Same seeds/configuration, not a promise of bitwise-identical training or rendered RGB."
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $Results "experiment_context.json") -Encoding UTF8
try {
    Set-Stage "FRESH_FINETUNING" "Original open-source SmolVLA + zero-output new LoRA; only 510 audited train trajectories."
    $Train = @{
        FreshLora=$true; AdapterCheckpoint=$Base; DatasetRoot=(Join-Path $Export "train")
        DatasetRepoId="local/isaac-pick-lift-execution-aligned-$DataTag-train"; SamplingPlan=$Plan
        RunName=$FreshRun; Steps=6000; SaveFreq=500; BatchSize=1; Seed=1000
        SchedulerWarmupSteps=300; SchedulerDecaySteps=6000; PeakLearningRate=0.00001; DecayLearningRate=0.000001; ZAxisWeight=3
    }
    & (Join-Path $PSScriptRoot "train_smolvla_dagger_recovery_8gb_v7.ps1") @Train
    $FreshAudit = Join-Path $Root "results\vla\training_logs\$FreshRun.initialization.json"
    Set-Stage "FIXED_VALIDATION_COMPARISON" "Both arms: same 240 validation indices and 500/3000/6000 checkpoints; primary comparison at 6000."
    foreach ($Arm in @("continued","fresh")) {
        $TrainingRun = if ($Arm -eq "fresh") { $FreshRun } else { $ContinuedRun }
        Invoke-Logged $Python @(
            (Join-Path $PSScriptRoot "scan_smolvla_checkpoints.py"),
            "--checkpoint-root",(Join-Path $Root "results\vla\training\$TrainingRun\checkpoints"),
            "--checkpoint-steps","500","3000","6000","--dataset-root",(Join-Path $Export "validation"),
            "--repo-id","local/isaac-pick-lift-execution-aligned-$DataTag-validation",
            "--samples","240","--seed","20261011","--device","cuda","--batch-size","2",
            "--position-target-mode","delta","--threshold-position-mae-cm","2",
            "--output",(Join-Path $Results ($Arm + "_validation240.json"))
        ) (Join-Path $Results ($Arm + "_validation240.log"))
    }
    $CompareArgs = @(
        (Join-Path $PSScriptRoot "compare_smolvla_initializations.py"),"--experiment-root",$Results,
        "--continued-checkpoint",$ContinuedFinal,"--fresh-checkpoint",$FreshFinal,"--fresh-initialization",$FreshAudit
    )
    Invoke-Logged $Python $CompareArgs (Join-Path $Results "comparison_offline.log")
    $Summary = Get-Content -LiteralPath (Join-Path $Results "comparison.json") -Raw | ConvertFrom-Json
    if (-not $Summary.both_final_checkpoints_pass_2cm_gate) {
        Set-Stage "VALIDATION_GATE_NOT_PASSED" "Offline comparison saved. At least one final checkpoint exceeds 2 cm; no closed-loop comparison."
        return
    }
    foreach ($Arm in @("continued","fresh")) {
        Set-Stage "CLOSED_LOOP_$($Arm.ToUpper())" "Same scene seeds, 600 control steps, actual 10 cm lift, no truth control or scripted fallback."
        $TrainingRun = if ($Arm -eq "fresh") { $FreshRun } else { $ContinuedRun }
        $Benchmark = @{
            RunTag=$DataTag; TrainingRun=$TrainingRun; EpisodesPerLevel=$EpisodesPerLevel; MaxSteps=600
            Port=$Port; ValidationReport=(Join-Path $Results ($Arm + "_validation240.json"))
            OutputRoot=(Join-Path $Results ($Arm + "_benchmark")); CheckpointStep=6000
        }
        & (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") @Benchmark
    }
    Invoke-Logged $Python ($CompareArgs + @("--require-benchmarks")) (Join-Path $Results "comparison_closed_loop.log")
    foreach ($Snapshot in $Hashes) {
        if ((Get-FileHash -LiteralPath $Snapshot.path -Algorithm SHA256).Hash -ne $Snapshot.sha256) { throw "Protected artifact changed: $($Snapshot.path)" }
    }
    Set-Stage "COMPARISON_COMPLETE" "Read comparison.json; small paired diagnostic, not a statistically stable success rate."
} catch {
    Set-Stage "FAILED" $_.Exception.Message
    throw
}
