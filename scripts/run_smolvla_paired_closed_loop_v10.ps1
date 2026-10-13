param(
    [string]$RunTag = "20261011_v10",
    [int]$EpisodesPerLevel = 3,
    [int]$Port = 8768,
    [int]$FreshPort = 0,
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $EpisodesPerLevel -lt 1) { throw "Unsafe run settings" }
if ($FreshPort -eq 0) { $FreshPort = $Port + 1 }
if ($Port -lt 1024 -or $Port -gt 65535 -or $FreshPort -lt 1024 -or $FreshPort -gt 65535 -or $Port -eq $FreshPort) {
    throw "Arms require distinct valid service ports"
}
$PreviousResults = Join-Path $Root "results\vla\fresh_lora_comparison_20261011_v10"
$Results = Join-Path $Root "results\vla\fresh_lora_closed_loop_$RunTag"
$DataTag = "20261011_truth_v9"
$ContinuedRun = "smolvla_execution_aligned_full_lora16_20261011_truth_v9"
$FreshRun = "smolvla_execution_aligned_fresh_lora16_20261011_v10"
$ContinuedFinal = Join-Path $Root "results\vla\training\$ContinuedRun\checkpoints\006000\pretrained_model"
$FreshFinal = Join-Path $Root "results\vla\training\$FreshRun\checkpoints\006000\pretrained_model"
$FreshAudit = Join-Path $Root "results\vla\training_logs\$FreshRun.initialization.json"
$Python = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Reason = "User requested paired closed-loop evaluation despite fresh LoRA failing the offline 2 cm gate on 2026-10-11. Diagnostic only; do not claim the gate passed."
$ResumeSuffix = if ($Resume) { "_resume_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff") } else { "" }
if ((Test-Path -LiteralPath $Results) -and -not $Resume) { throw "Refusing to overwrite $Results; use -Resume to reuse completed arms" }
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $Results "pipeline_status.json"))) { throw "Resume requires an existing run status" }
foreach ($Required in @($Python,$FreshAudit,(Join-Path $ContinuedFinal "adapter_model.safetensors"),(Join-Path $FreshFinal "adapter_model.safetensors"))) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing $Required" }
}
if ($Resume) {
    $PreviousStatus = Get-Content -LiteralPath (Join-Path $Results "pipeline_status.json") -Raw | ConvertFrom-Json
    if ($PreviousStatus.stage -notin @("FAILED","PAIRED_CLOSED_LOOP_COMPLETE") -or
        $PreviousStatus.episodes_per_level -ne $EpisodesPerLevel -or
        $PreviousStatus.continued_training_run -ne $ContinuedRun -or $PreviousStatus.fresh_training_run -ne $FreshRun) {
        throw "Resume requires a stopped run with exactly the same weights and episode counts"
    }
    Copy-Item -LiteralPath (Join-Path $Results "pipeline_status.json") -Destination (Join-Path $Results ("pipeline_status_before_resume_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff") + ".json"))
    if (Test-Path -LiteralPath (Join-Path $Results "comparison.json")) {
        Copy-Item -LiteralPath (Join-Path $Results "comparison.json") -Destination (Join-Path $Results ("comparison_before" + $ResumeSuffix + ".json"))
    }
} else {
    New-Item -ItemType Directory -Path $Results | Out-Null
    foreach ($File in @("continued_validation240.json","fresh_validation240.json","experiment_context.json")) {
        Copy-Item -LiteralPath (Join-Path $PreviousResults $File) -Destination (Join-Path $Results $File)
    }
    Copy-Item -LiteralPath (Join-Path $PreviousResults "comparison.json") -Destination (Join-Path $Results "prior_offline_comparison.json")
}
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
$StatusPath = Join-Path $Results "pipeline_status.json"
function Set-Stage([string]$Stage, [string]$Message = "") {
    @{
        stage=$Stage; message=$Message; updated_at_utc=[DateTime]::UtcNow.ToString("o")
        previous_results=$PreviousResults; results=$Results; checkpoint_step=6000
        continued_training_run=$ContinuedRun; fresh_training_run=$FreshRun
        episodes_per_level=$EpisodesPerLevel; planned_episodes_per_arm=(3*$EpisodesPerLevel)
        planned_total_episodes=(6*$EpisodesPerLevel); validation_gate_override_requested=$true
        validation_gate_override_reason=$Reason; retraining=$false
        resumed=[bool]$Resume; continued_port=$Port; fresh_port=$FreshPort
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    Write-Host "PAIRED_CLOSED_LOOP_STAGE=$Stage $Message"
}
function Invoke-Comparison([switch]$WithBenchmarks) {
    $CompareArgs = @(
        (Join-Path $PSScriptRoot "compare_smolvla_initializations.py"),
        "--experiment-root",$Results,"--continued-checkpoint",$ContinuedFinal,
        "--fresh-checkpoint",$FreshFinal,"--fresh-initialization",$FreshAudit
    )
    if ($WithBenchmarks) {
        $CompareArgs += @("--require-benchmarks","--allow-validation-gate-override","--validation-gate-override-reason",$Reason)
    }
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Python @CompareArgs 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_.ToString() }
        } | Tee-Object -FilePath (Join-Path $Results ("comparison_" + [bool]$WithBenchmarks + $ResumeSuffix + ".log"))
        $Code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $PreviousPreference }
    if ($Code -ne 0) { throw "Comparison verification failed ($Code)" }
}
try {
    Set-Stage "VERIFYING_PAIRED_PROTOCOL" "Existing 6000-step weights unchanged; explicit diagnostic override of position-error gate."
    Invoke-Comparison
    foreach ($Arm in @("continued","fresh")) {
        $TrainingRun = if ($Arm -eq "fresh") { $FreshRun } else { $ContinuedRun }
        $ArmRoot = Join-Path $Results ($Arm + "_benchmark")
        $ArmCheckpoint = if ($Arm -eq "fresh") { $FreshFinal } else { $ContinuedFinal }
        if (Test-Path -LiteralPath $ArmRoot) {
            $Completed = @(Get-ChildItem -LiteralPath $ArmRoot -Filter "vla_benchmark_report.json" -Recurse)
            if (-not $Resume -or $Completed.Count -ne 1) { throw "Existing partial/ambiguous arm: $ArmRoot; preserve it and use a new RunTag" }
            $Saved = Get-Content -LiteralPath $Completed[0].FullName -Raw | ConvertFrom-Json
            $Context = Get-Content -LiteralPath (Join-Path $Completed[0].DirectoryName "test_context.json") -Raw | ConvertFrom-Json
            if ($Saved.status -ne "VLA_BENCHMARK_OK" -or $Saved.overall.num_episodes -ne (3*$EpisodesPerLevel) -or
                $Context.training_run -ne $TrainingRun -or $Context.checkpoint_step -ne 6000 -or
                $Context.maximum_control_steps -ne 600 -or $Context.inference_seed -ne 20261011 -or
                $Context.checkpoint_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $ArmCheckpoint "adapter_model.safetensors") -Algorithm SHA256).Hash) {
                throw "Saved benchmark context/weight/count mismatch: $Arm"
            }
            foreach ($Level in @("easy","medium","hard")) {
                if ($Saved.config.episode_counts.$Level -ne $EpisodesPerLevel) { throw "Saved level count mismatch" }
            }
            Write-Host "REUSING_COMPLETED_ARM=$Arm $($Completed[0].FullName)"
            continue
        }
        Set-Stage ("CLOSED_LOOP_" + $Arm.ToUpper()) "Same Easy/Medium/Hard scene seeds; no truth control or scripted fallback; actual 10 cm lift."
        $Benchmark = @{
            RunTag=$DataTag; TrainingRun=$TrainingRun; EpisodesPerLevel=$EpisodesPerLevel
            MaxSteps=600; Port=$(if ($Arm -eq "fresh") { $FreshPort } else { $Port }); CheckpointStep=6000
            ValidationReport=(Join-Path $Results ($Arm + "_validation240.json"))
            OutputRoot=(Join-Path $Results ($Arm + "_benchmark"))
            AllowValidationGateOverride=$true; ValidationGateOverrideReason=$Reason
        }
        & (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") @Benchmark
    }
    Set-Stage "VERIFYING_SCENES_AND_SUMMARIZING" "Verify matching sample keys/scene parameters/weight hashes, then report measured successes."
    Invoke-Comparison -WithBenchmarks
    Set-Stage "PAIRED_CLOSED_LOOP_COMPLETE" "Read comparison.json. Small paired diagnostic, not a statistically stable success-rate estimate."
} catch {
    Set-Stage "FAILED" $_.Exception.Message
    throw
}
