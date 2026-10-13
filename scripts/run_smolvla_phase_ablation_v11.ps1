param([string]$RunTag="20261011_v11", [int]$EpisodesPerLevel=3, [int]$BasePort=8770)
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $EpisodesPerLevel -lt 1 -or $BasePort -lt 1024 -or $BasePort -gt 65532) {throw "Unsafe settings"}
$Results=Join-Path $Root "results\vla\phase_ablation_$RunTag"
if (Test-Path -LiteralPath $Results) {throw "Refusing to overwrite $Results"}
$TrainingRun="smolvla_execution_aligned_full_lora16_20261011_truth_v9"
$Checkpoint=Join-Path $Root "results\vla\training\$TrainingRun\checkpoints\006000\pretrained_model"
$Validation=Join-Path $Root "results\vla\fresh_lora_comparison_20261011_v10\continued_validation240.json"
$Python="C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$ModelHash=(Get-FileHash -LiteralPath (Join-Path $Checkpoint "adapter_model.safetensors") -Algorithm SHA256).Hash
$Variants=@(
    @{name="legacy_model"; phase="legacy"; orientation="model"},
    @{name="feedback_model"; phase="feedback_hysteresis"; orientation="model"},
    @{name="legacy_fixed"; phase="legacy"; orientation="fixed_top_down"},
    @{name="feedback_fixed"; phase="feedback_hysteresis"; orientation="fixed_top_down"}
)
New-Item -ItemType Directory -Path $Results | Out-Null
@{
    training_run=$TrainingRun; checkpoint=$Checkpoint; checkpoint_sha256=$ModelHash
    episodes_per_level=$EpisodesPerLevel; inference_seed=20261011; variants=$Variants
    purpose="2x2 diagnostic: composite phase mechanism x orientation override; no new weights/data/interface."
    fixed_orientation="Replace quaternion only for phases 1..4; preserve each prediction xyz/gripper. Different closed-loop observations may lead to different subsequent xyz predictions."
    phase_intervention="2.5 cm entry/3.5 cm exit, dwell and nonempty-grip proprioceptive proxy; no object truth or timer-only progression."
    next_stage="No automatic interface redesign/fine-tuning from a small-sample result."
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Results "experiment_context.json") -Encoding UTF8
function Set-Stage([string]$Stage,[string]$Message="") {
    @{stage=$Stage; message=$Message; results=$Results; training_run=$TrainingRun; checkpoint_step=6000;
      checkpoint_sha256=$ModelHash; episodes_per_level=$EpisodesPerLevel; planned_total_episodes=(12*$EpisodesPerLevel);
      updated_at_utc=[DateTime]::UtcNow.ToString("o"); retraining=$false; interface_changed=$false
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Results "pipeline_status.json") -Encoding UTF8
    Write-Host "PHASE_ABLATION_STAGE=$Stage $Message"
}
try {
    for ($Index=0; $Index -lt $Variants.Count; $Index++) {
        $Variant=$Variants[$Index]
        Set-Stage $Variant.name "Same weights and scenes; record per-control-step phase/action/guard diagnostics."
        $Options=@{
            RunTag="20261011_truth_v9"; TrainingRun=$TrainingRun; CheckpointStep=6000
            ValidationReport=$Validation; EpisodesPerLevel=$EpisodesPerLevel; MaxSteps=600
            Port=($BasePort+$Index); OutputRoot=(Join-Path $Results $Variant.name)
            PhaseMechanism=$Variant.phase; OrientationMode=$Variant.orientation; RecordPhaseDiagnostics=$true
        }
        & (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") @Options
        if ((Get-FileHash -LiteralPath (Join-Path $Checkpoint "adapter_model.safetensors") -Algorithm SHA256).Hash -ne $ModelHash) {throw "Model changed"}
    }
    Set-Stage "SUMMARIZING" "Verify scenes and calculate phase/pregrasp/guard metrics."
    $env:PYTHONPATH=Join-Path $Root "src"
    $env:PYTHONUTF8="1"
    & $Python (Join-Path $PSScriptRoot "summarize_vla_phase_ablation.py") --root $Results --checkpoint $Checkpoint
    if ($LASTEXITCODE -ne 0) {throw "Ablation verification failed"}
    Set-Stage "PHASE_ABLATION_COMPLETE" "Read phase_ablation_report.json; inspect evidence before interface redesign."
} catch {
    Set-Stage "FAILED" $_.Exception.Message
    throw
}
