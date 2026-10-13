param([string]$RunTag = '', [switch]$PreflightOnly)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $RunTag) { $RunTag = Get-Date -Format 'yyyyMMdd_HHmmss_fff' }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid run tag' }
$Output = Join-Path $Root "results\vla\medium32_four_models_$RunTag"
$Arms = @(
    @{name='v9_r16_chunk4'; run='smolvla_execution_aligned_full_lora16_20261011_truth_v9'; rank=16; chunk=4;
      validation='results\vla\fresh_lora_comparison_20261011_v10\continued_validation240.json'},
    @{name='v17_r16_chunk1'; run='smolvla_v9_chunk1_20261012_195615_610_v17'; rank=16; chunk=1;
      validation='results\vla\v9_chunk_ablation_20261012_195615_610_v17\chunk1_validation.json'},
    @{name='v17_r16_chunk4'; run='smolvla_v9_chunk4_20261012_195615_610_v17'; rank=16; chunk=4;
      validation='results\vla\v9_chunk_ablation_20261012_195615_610_v17\chunk4_validation.json'},
    @{name='v18_r32_chunk4'; run='smolvla_chunk4_rank32_20261012_212747_052_v18'; rank=32; chunk=4;
      validation='results\vla\rank32_plateau_20261012_212747_052_v18\validation.json'}
)
foreach ($Arm in $Arms) {
    $Arm.checkpoint = Join-Path $Root "results\vla\training\$($Arm.run)\checkpoints\006000\pretrained_model"
    $Config = Get-Content (Join-Path $Arm.checkpoint 'config.json') -Raw | ConvertFrom-Json
    $Adapter = Get-Content (Join-Path $Arm.checkpoint 'adapter_config.json') -Raw | ConvertFrom-Json
    if ($Config.chunk_size -ne $Arm.chunk -or $Adapter.r -ne $Arm.rank -or
        $Config.input_features.'observation.state'.shape[0] -ne 30 -or
        $Config.output_features.action.shape[0] -ne 8) { throw "Interface mismatch: $($Arm.name)" }
    $Scan = Get-Content (Join-Path $Root $Arm.validation) -Raw | ConvertFrom-Json
    $Selected = @($Scan.results | Where-Object { $_.step -eq 6000 })
    if ($Scan.status -ne 'VLA_CHECKPOINT_SCAN_COMPLETE' -or $Scan.dataset_split -ne 'validation' -or
        -not $Scan.threshold_passed -or $Selected.Count -ne 1 -or
        $Selected[0].position_mae_cm -gt $Scan.threshold_position_mae_cm) { throw "Validation failed: $($Arm.name)" }
    $Arm.checkpoint_sha256 = (Get-FileHash (Join-Path $Arm.checkpoint 'adapter_model.safetensors')).Hash
    $Arm.validation_position_mae_cm = $Selected[0].position_mae_cm
}
if ($PreflightOnly) { $Arms | ConvertTo-Json -Depth 6; exit 0 }
if (Test-Path -LiteralPath $Output) { throw "Refusing to overwrite $Output" }
New-Item -ItemType Directory -Path $Output | Out-Null
$Completed = @()
$CurrentArm = $null
$Protocol = @{
    results=$Output; episodes_per_arm=32; planned_total_episodes=128; levels=@('medium'); scene_seed=7102;
    checkpoint_step=6000; max_steps=600; execution_horizon=1; action_repeat=5;
    minimum_actual_lift_m=0.10; phase_mechanism='legacy'; orientation_mode='model';
    input='RGB_language_proprio_visual_estimate_phase'; grasp_enable_distance_m=0.075;
    retraining=$false; scripted_fallback=$false; teacher_truth_in_controller=$false;
    headless=$true; sequential_models=$true; inference_seed=20261011;
    caveat='Matched scene seeds and protocol; not pixel-identical images or common per-request diffusion noise. Different training histories: not a pure rank ablation.'
}
function Stage([string]$Name, [string]$Message) {
    $State = @{}
    foreach ($Key in $Protocol.Keys) { $State[$Key] = $Protocol[$Key] }
    $State.stage=$Name; $State.message=$Message; $State.current_arm=$CurrentArm
    $State.completed_arms=@($Completed); $State.completed_episodes=32*$Completed.Count
    $State.driver_pid=$PID; $State.updated_at_utc=[DateTime]::UtcNow.ToString('o')
    $State | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "MEDIUM32_STAGE=$Name ARM=$CurrentArm RESULTS=$Output"
}
$SourceHashes = @{}
foreach ($File in @('scripts\benchmark_vla_truth_student_v9.ps1','scripts\run_vla_benchmark.py',
    'scripts\serve_smolvla.py','src\isaac_lab_data_engine\vla\hybrid_controller.py',
    'configs\benchmark.yaml','configs\randomization_medium.yaml',
    'results\vision\rgbd_keypoint_supplement_160x120_v3\best_model.pt')) {
    $SourceHashes[$File]=(Get-FileHash (Join-Path $Root $File)).Hash
}
@{protocol=$Protocol; arms=$Arms; source_sha256=$SourceHashes} | ConvertTo-Json -Depth 10 |
    Set-Content (Join-Path $Output 'experiment_plan.json') -Encoding UTF8
try {
    foreach ($Arm in $Arms) {
        $CurrentArm=$Arm.name
        foreach ($File in $SourceHashes.Keys) {
            if ((Get-FileHash (Join-Path $Root $File)).Hash -ne $SourceHashes[$File]) { throw "Source changed: $File" }
        }
        $Probe=New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback,0)
        try { $Probe.Start(); $Port=$Probe.LocalEndpoint.Port } finally { $Probe.Stop() }
        $ArmOutput=Join-Path $Output $Arm.name
        Stage 'RUNNING' "Running 32 Medium scenes for $CurrentArm; live episode progress is in this arm's isaac_stdout.log."
        & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
            -TrainingRun $Arm.run -CheckpointStep 6000 -ValidationReport $Arm.validation `
            -OutputRoot $ArmOutput -Port $Port -Levels medium -EpisodesPerLevel 32 -MaxSteps 600 `
            -ExecutionHorizon 1 -InputInterface legacy -PhaseMechanism legacy -StudentPhaseMode legacy `
            -OrientationMode model -GraspEnableDistanceM 0.075 -RecordPhaseDiagnostics
        if ((Get-FileHash (Join-Path $Arm.checkpoint 'adapter_model.safetensors')).Hash -ne $Arm.checkpoint_sha256) {
            throw "Weights changed: $CurrentArm"
        }
        $Reports=@(Get-ChildItem -LiteralPath $ArmOutput -Recurse -Filter 'vla_benchmark_report.json' -File)
        if ($Reports.Count -ne 1) { throw 'Expected one report per arm' }
        $Report=Get-Content $Reports[0].FullName -Raw | ConvertFrom-Json
        if ($Report.status -ne 'VLA_BENCHMARK_OK' -or $Report.overall.num_episodes -ne 32 -or
            $Report.levels.Count -ne 1 -or $Report.levels[0].difficulty -ne 'medium' -or
            $Report.overall.scripted_ik_action_count -ne 0 -or $Report.overall.training_label_action_count -ne 0) {
            throw "Incomplete or contaminated report: $CurrentArm"
        }
        # Preserve initialization/RPC failures in the denominator; never silently replace scenes.
        $Completed += @{
            model=$CurrentArm; checkpoint=$Arm.checkpoint; checkpoint_sha256=$Arm.checkpoint_sha256;
            report=$Reports[0].FullName; episodes=$Report.overall.num_episodes;
            successes=$Report.overall.num_success; success_rate=$Report.overall.success_rate;
            failure_counts=$Report.overall.failure_counts; vla_error_count=$Report.overall.vla_error_count;
            mean_vla_inference_ms=$Report.overall.mean_vla_inference_ms; wall_time_s=$Report.overall.wall_time_s
        }
        @{status='PARTIAL'; protocol=$Protocol; results=@($Completed)} | ConvertTo-Json -Depth 10 |
            Set-Content (Join-Path $Output 'comparison_summary.json') -Encoding UTF8
        Stage 'ARM_COMPLETE' "Finished $CurrentArm; starting next model automatically."
    }
    @{status='COMPLETE'; protocol=$Protocol; results=@($Completed)} | ConvertTo-Json -Depth 10 |
        Set-Content (Join-Path $Output 'comparison_summary.json') -Encoding UTF8
    $CurrentArm=$null
    Stage 'COMPLETE' 'All four models completed 32 Medium episodes each; original weights unchanged.'
} catch { Stage 'FAILED' $_.Exception.Message; throw }
