param(
    [string]$RunTag = "20261011_truth_v9",
    [int]$TargetPerLevel = 200,
    [int]$TrainPerLevel = 170,
    [int]$ValidationPerLevel = 15,
    [int]$TestPerLevel = 15,
    [int]$MaxAttemptsPerLevel = 1000,
    [int]$SeedOffset = 20262011,
    [int]$Steps = 6000,
    [ValidateSet("visual","truth")][string]$Teacher = "truth",
    [switch]$SkipCollection,
    [switch]$SkipTraining
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$') { throw "Unsafe RunTag" }
if ($TrainPerLevel + $ValidationPerLevel + $TestPerLevel -ne $TargetPerLevel) { throw "Split counts must sum to collection target" }
$IsaacPython = "C:\Users\yjh\anaconda3\envs\isaaclab23\python.exe"
$VlaPython = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Source = Join-Path $Root "dataset\vla_execution_aligned_full_$RunTag"
$Export = Join-Path $Root "exports\lerobot\pick_lift_execution_aligned_$RunTag"
$Results = Join-Path $Root "results\vla\execution_aligned_$RunTag"
$Plan = Join-Path $Results "train_sampling_plan.json"
$StatusPath = Join-Path $Results "pipeline_status.json"
$RunName = "smolvla_execution_aligned_full_lora16_$RunTag"
$Adapter = Join-Path $Root "results\vla\training\smolvla_visual_fsm_delta_tight_dagger15mm_lora16_v7\checkpoints\001500\pretrained_model"
if (Test-Path -LiteralPath $Results) { throw "Refusing to overwrite pipeline results: $Results" }
if (Test-Path -LiteralPath $Export) { throw "Refusing to overwrite export: $Export" }
if (-not $SkipCollection -and (Test-Path -LiteralPath $Source)) { throw "Refusing to overwrite source: $Source" }
foreach ($Path in @($IsaacPython,$VlaPython,$Adapter)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing $Path" }
}
New-Item -ItemType Directory -Path $Results | Out-Null
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
function Set-Stage([string]$Stage, [string]$Message = "") {
    @{
        stage=$Stage; message=$Message; run_tag=$RunTag; source=$Source; export=$Export
        training_run=$RunName; updated_at_utc=[DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    Write-Host "PIPELINE_STAGE=$Stage $Message"
}
function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$Log) {
    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message }
        else { $_.ToString() }
    } | Tee-Object -FilePath $Log
    $Code = $LASTEXITCODE
    $ErrorActionPreference = $PreviousPreference
    if ($Code -ne 0) { throw "$Exe failed with code $Code; see $Log" }
}
try {
    if (-not $SkipCollection) {
        Set-Stage "COLLECTING" "Teacher=$Teacher; student RGB/proprioception/estimated pose only; no VLA or fallback."
        $CollectArgs = @(
            (Join-Path $PSScriptRoot "run_vla_benchmark.py"), "--headless", "--levels","easy","medium","hard",
            "--max-steps","600", "--collect-full-trajectories",$Source,
            "--collection-target-per-level","$TargetPerLevel", "--collection-max-attempts-per-level","$MaxAttemptsPerLevel",
            "--collection-seed-offset","$SeedOffset",
            "--collection-teacher",$Teacher,
            "--condition-vla-on-visual-pose", "--condition-vla-on-controller-phase",
            "--vla-position-target-mode","delta", "--execution-horizon","1", "--action-repeat","5",
            "--visual-init-attempts","1", "--camera-ready-timeout-s","45",
            "--visual-invalid-pose-timeout-s","8", "--device","cuda:0",
            "--rendering_mode","performance", "--output-dir",(Join-Path $Results "collection_reports")
        )
        Invoke-Logged $IsaacPython $CollectArgs (Join-Path $Results "collection.log")
    }
    Set-Stage "AUDITING_AND_EXPORTING"
    $ExportArgs = @(
        (Join-Path $PSScriptRoot "export_vla_execution_aligned.py"),
        "--source-root",$Source, "--output-root",$Export,
        "--repo-prefix","local/isaac-pick-lift-execution-aligned-$RunTag",
        "--train-per-level","$TrainPerLevel","--validation-per-level","$ValidationPerLevel",
        "--test-per-level","$TestPerLevel"
    )
    Invoke-Logged $VlaPython $ExportArgs (Join-Path $Results "export.log")
    Set-Stage "AUDITING_EXPORTED_LABELS"
    Invoke-Logged $VlaPython @(
        (Join-Path $PSScriptRoot "audit_vla_execution_aligned_export.py"), "--root",$Export
    ) (Join-Path $Results "parquet_audit.log")
    Set-Stage "BUILDING_SAMPLER"
    Invoke-Logged $VlaPython @(
        (Join-Path $PSScriptRoot "build_vla_sampling_plan.py"), "--dataset-root",(Join-Path $Export "train"),
        "--output",$Plan, "--balance-levels", "--easy-weight","1", "--medium-weight","1", "--hard-weight","1",
        "--phase-0-weight","0.5", "--phase-1-weight","1", "--phase-2-weight","1.5",
        "--phase-3-weight","1.5", "--phase-4-weight","2", "--dagger-recovery-weight","1"
    ) (Join-Path $Results "sampling.log")
    if ($SkipTraining) { Set-Stage "DATA_READY"; return }
    Set-Stage "FINETUNING" "Fresh optimizer, v7 best adapter, only newly collected train data."
    $FineTuneArgs = @{
        DatasetRoot = (Join-Path $Export "train")
        DatasetRepoId = "local/isaac-pick-lift-execution-aligned-$RunTag-train"
        SamplingPlan = $Plan; AdapterCheckpoint = $Adapter; RunName = $RunName
        Steps = $Steps; SaveFreq = 500; SchedulerWarmupSteps = 300; SchedulerDecaySteps = $Steps
        PeakLearningRate = 0.00001; DecayLearningRate = 0.000001; ZAxisWeight = 3
    }
    & (Join-Path $PSScriptRoot "train_smolvla_dagger_recovery_8gb_v7.ps1") @FineTuneArgs
    Set-Stage "SCANNING_VALIDATION" "Fixed 240-frame validation set; no test data used to select checkpoints."
    Invoke-Logged $VlaPython @(
        (Join-Path $PSScriptRoot "scan_smolvla_checkpoints.py"),
        "--checkpoint-root",(Join-Path $Root "results\vla\training\$RunName\checkpoints"),
        "--dataset-root",(Join-Path $Export "validation"),
        "--repo-id","local/isaac-pick-lift-execution-aligned-$RunTag-validation",
        "--samples","240", "--seed","20261011", "--device","cuda",
        "--batch-size","2", "--position-target-mode","delta",
        "--threshold-position-mae-cm","2", "--output",(Join-Path $Results "validation_scan.json")
    ) (Join-Path $Results "validation_scan.log")
    $Scan = Get-Content -LiteralPath (Join-Path $Results "validation_scan.json") -Raw | ConvertFrom-Json
    if (-not $Scan.threshold_passed) {
        Set-Stage "VALIDATION_GATE_NOT_PASSED" "Position error exceeds 2 cm; no closed-loop benchmark started."
        return
    }
    Set-Stage "VISUAL_ONLY_STUDENT_BENCHMARK" "Unseen benchmark scenes; no truth teacher/phase/guard targets; require actual 10 cm lift."
    & (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") -RunTag $RunTag
    Set-Stage "COMPLETE_TRAIN_VALIDATION_AND_VISUAL_BENCHMARK" "Consult the actual VLA report, never substitute teacher collection success for student success."
} catch {
    Set-Stage "FAILED" $_.Exception.Message
    throw
}
