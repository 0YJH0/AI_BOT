param(
    [string]$RunTag = "20261011_truth_v9",
    [int]$EpisodesPerLevel = 10,
    [int]$MaxSteps = 600,
    [int]$Port = 8768,
    [string]$ValidationReport = "",
    [string]$TrainingRun = "",
    [string]$OutputRoot = "",
    [int]$CheckpointStep = 0,
    [switch]$AllowValidationGateOverride,
    [string]$ValidationGateOverrideReason = "",
    [ValidateSet("legacy","feedback_hysteresis")][string]$PhaseMechanism = "legacy",
    [ValidateSet("model","fixed_top_down")][string]$OrientationMode = "model",
    [switch]$RecordPhaseDiagnostics,
    [ValidateSet("easy","medium","hard")][string[]]$Levels = @("easy","medium","hard"),
    [string]$DatasetExportRoot = "",
    [ValidateSet("legacy","relative4d")][string]$InputInterface = "legacy",
    [ValidateRange(0.000001,0.075)][double]$GraspEnableDistanceM = 0.075,
    [switch]$ShowWindow,
    [switch]$Realtime,
    [ValidateSet("legacy","grip_feedback")][string]$StudentPhaseMode = "legacy",
    [switch]$ProprioOnly,
    [ValidateSet(1,2,4)][int]$ExecutionHorizon = 1
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $EpisodesPerLevel -lt 1 -or $MaxSteps -lt 1) { throw "Invalid run settings" }
$ReviewRoot = Join-Path $Root "results\vla\execution_aligned_$RunTag"
if ([string]::IsNullOrWhiteSpace($TrainingRun)) { $TrainingRun = "smolvla_execution_aligned_full_lora16_$RunTag" }
if ($TrainingRun -notmatch '^[a-zA-Z0-9_-]+$') { throw "Unsafe TrainingRun" }
if ([string]::IsNullOrWhiteSpace($ValidationReport)) { $ValidationReport = Join-Path $ReviewRoot "validation_scan.json" }
elseif (-not [System.IO.Path]::IsPathRooted($ValidationReport)) { $ValidationReport = Join-Path $Root $ValidationReport }
$Scan = Get-Content -LiteralPath $ValidationReport -Raw | ConvertFrom-Json
if ($Scan.status -ne "VLA_CHECKPOINT_SCAN_COMPLETE" -or $Scan.dataset_split -ne "validation") {
    throw "A completed independent validation scan is required first"
}
if ($AllowValidationGateOverride -and ([string]::IsNullOrWhiteSpace($ValidationGateOverrideReason) -or $CheckpointStep -le 0)) {
    throw "Diagnostic gate override requires an explicit reason and checkpoint step"
}
if (-not $Scan.threshold_passed -and -not $AllowValidationGateOverride) {
    throw "Validation gate not passed; an explicitly authorized diagnostic override is required"
}
$ExpectedCheckpoints = Join-Path $Root "results\vla\training\$TrainingRun\checkpoints"
$ExpectedValidation = Join-Path $Root "exports\lerobot\pick_lift_execution_aligned_$RunTag\validation"
if ($DatasetExportRoot) { $ExpectedValidation = Join-Path ([System.IO.Path]::GetFullPath($DatasetExportRoot)) "validation" }
if ([System.IO.Path]::GetFullPath($Scan.scan_config.checkpoint_root) -ne [System.IO.Path]::GetFullPath($ExpectedCheckpoints) -or
    [System.IO.Path]::GetFullPath($Scan.scan_config.dataset_root) -ne [System.IO.Path]::GetFullPath($ExpectedValidation)) {
    throw "Validation report belongs to a different model or dataset"
}
$Selected = $Scan.best_checkpoint
if ($CheckpointStep -gt 0) {
    $Selected = @($Scan.results | Where-Object { [int]$_.step -eq $CheckpointStep })
    if ($Selected.Count -ne 1) { throw "Requested checkpoint was not validated" }
    $Selected = $Selected[0]
}
$SelectedGatePassed = [double]$Selected.position_mae_cm -le [double]$Scan.threshold_position_mae_cm
if (-not $SelectedGatePassed -and -not $AllowValidationGateOverride) { throw "Requested checkpoint did not pass the validation gate" }
$Step = "{0:D6}" -f [int]$Selected.step
$Checkpoint = Join-Path $Root "results\vla\training\$TrainingRun\checkpoints\$Step\pretrained_model"
$HorizonConfig = Get-Content (Join-Path $Checkpoint 'config.json') -Raw | ConvertFrom-Json
if ($ExecutionHorizon -gt [int]$HorizonConfig.chunk_size) { throw 'Execution horizon exceeds model prediction chunk' }
if ($ProprioOnly) {
    $PolicyConfig = Get-Content (Join-Path $Checkpoint "config.json") -Raw | ConvertFrom-Json
    if ($PolicyConfig.input_features.'observation.state'.shape[0] -ne 25 -or
        $PolicyConfig.output_features.action.shape[0] -ne 8 -or $InputInterface -ne "legacy" -or
        $PhaseMechanism -ne "legacy" -or $StudentPhaseMode -ne "legacy" -or $OrientationMode -ne "model") {
        throw "Robot-only evaluation requires a 25/8 checkpoint and no phase/orientation intervention"
    }
}
$VlaPython = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$IsaacPython = "C:\Users\yjh\anaconda3\envs\isaaclab23\python.exe"
if ($ShowWindow) {
    # pythonw creates the Isaac GUI without a second console whose close event
    # aborts native libraries and releases the camera while Python is running.
    $IsaacPython = "C:\Users\yjh\anaconda3\envs\isaaclab23\pythonw.exe"
}
foreach ($Required in @($VlaPython,$IsaacPython,(Join-Path $Checkpoint "config.json"),(Join-Path $Checkpoint "adapter_model.safetensors"))) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing $Required" }
}
$Probe = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
try { $Probe.Start() } finally { $Probe.Stop() }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = $ReviewRoot }
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $Root $OutputRoot }
$Output = Join-Path $OutputRoot ("visual_only_benchmark_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
@{
    run_tag=$RunTag; checkpoint=$Checkpoint; validation_report=$ValidationReport
    training_run=$TrainingRun; checkpoint_step=[int]$Selected.step
    validation_samples=@($Scan.indices).Count; selected_validation_position_mae_cm=$Selected.position_mae_cm
    validation_gate_passed=$SelectedGatePassed; validation_gate_threshold_cm=$Scan.threshold_position_mae_cm
    validation_gate_override_requested=[bool]$AllowValidationGateOverride
    validation_gate_override_used=(-not $SelectedGatePassed -and [bool]$AllowValidationGateOverride)
    validation_gate_override_reason=$ValidationGateOverrideReason; inference_seed=20261011
    phase_mechanism=$PhaseMechanism; orientation_mode=$OrientationMode
    phase_diagnostics_recorded=[bool]$RecordPhaseDiagnostics
    grasp_enable_distance_m=$(if ($ProprioOnly) { $null } else { $GraspEnableDistanceM }); gripper_latch_mode="hold_closed_until_episode_reset"
    levels=$Levels; visual_initialization_protocol="one_coherent_snapshot_per_real_control_step_v2"
    input_interface=$InputInterface; phase_in_model_input=($InputInterface -eq "legacy" -and -not $ProprioOnly)
    proprio_only=[bool]$ProprioOnly; external_target_used_for_control=(-not $ProprioOnly)
    checkpoint_sha256=(Get-FileHash -LiteralPath (Join-Path $Checkpoint "adapter_model.safetensors") -Algorithm SHA256).Hash
    episodes_per_level=$EpisodesPerLevel; maximum_control_steps=$MaxSteps; minimum_actual_lift_m=0.10
    scope=$(if ($EpisodesPerLevel -le 2) { "diagnostic_smoke_not_statistical_benchmark" } else { "fixed_scene_benchmark" })
    gui_visible=[bool]$ShowWindow; realtime_pacing=[bool]$Realtime
    student_phase_mode=$StudentPhaseMode
    execution_horizon=$ExecutionHorizon; chunk_size=[int]$HorizonConfig.chunk_size; action_repeat=5
    teacher_truth_in_controller=$false; scripted_fallback=$false
    student_phase_source=$(if ($ProprioOnly) { "none" } else { "visual_feedback_fsm" })
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output "test_context.json") -Encoding UTF8
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
$Server = $null
$Isaac = $null
try {
    $ServerArgs = @("scripts\serve_smolvla_windows_v044_v4.py","--checkpoint",('"' + $Checkpoint + '"'),
        "--port",$Port,"--execution-horizon",$ExecutionHorizon,"--device","cuda","--seed",20261011,
        "--gpu-memory-fraction",0.35,"--profile-memory")
    $ServerOptions = @{
        FilePath=$VlaPython; ArgumentList=$ServerArgs; WorkingDirectory=$Root; WindowStyle="Hidden"; PassThru=$true
        RedirectStandardOutput=(Join-Path $Output "server_stdout.log"); RedirectStandardError=(Join-Path $Output "server_stderr.log")
    }
    $Server = Start-Process @ServerOptions
    $Ready = $false
    $Deadline = [DateTime]::UtcNow.AddSeconds(180)
    while ([DateTime]::UtcNow -lt $Deadline) {
        $Server.Refresh()
        if ($Server.HasExited) { throw "VLA service exited; see server logs in $Output" }
        if ((Test-Path (Join-Path $Output "server_stdout.log")) -and (Select-String -LiteralPath (Join-Path $Output "server_stdout.log") -Pattern "SMOLVLA_SERVER_READY=" -Quiet)) { $Ready=$true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $Ready) { throw "VLA service readiness timeout" }
    # No collector, label replay, truth teacher, or Scripted IK fallback.
    # Both phase feedback and guard distance use the visual estimate only.
    # Simulator truth is used solely by the success/error metrics.
    $IsaacArgs = @("scripts\run_vla_benchmark.py","--levels") + $Levels + @(
        "--episodes-per-level",$EpisodesPerLevel,"--max-steps",$MaxSteps,"--minimum-true-lift-m",0.10,
        "--vla-position-target-mode","delta","--pure-vla","--require-vla",
        "--vla-port",$Port,"--vla-timeout",60,"--execution-horizon",$ExecutionHorizon,"--action-repeat",5,
        "--camera-ready-timeout-s",180,"--visual-init-attempts",1,"--progress-every",100,
        "--device","cuda:0","--rendering_mode","performance","--output-dir",('"' + $Output + '"'))
    $IsaacArgs += @("--phase-mechanism",$PhaseMechanism,"--orientation-mode",$OrientationMode)
    if ($ProprioOnly) { $IsaacArgs += "--proprio-only" }
    else { $IsaacArgs += @("--condition-vla-on-visual-pose","--condition-vla-on-controller-phase") }
    $IsaacArgs += @("--student-phase-mode",$StudentPhaseMode)
    $IsaacArgs += @("--vla-input-interface",$InputInterface)
    $IsaacArgs += @("--grasp-enable-distance-m",$GraspEnableDistanceM.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    if ($RecordPhaseDiagnostics) { $IsaacArgs += "--record-phase-diagnostics" }
    if (-not $ShowWindow) { $IsaacArgs += "--headless" }
    if ($Realtime) { $IsaacArgs += "--realtime" }
    $IsaacOptions = @{
        FilePath=$IsaacPython; ArgumentList=$IsaacArgs; WorkingDirectory=$Root;
        WindowStyle=$(if ($ShowWindow) { "Normal" } else { "Hidden" }); PassThru=$true
        RedirectStandardOutput=(Join-Path $Output "isaac_stdout.log"); RedirectStandardError=(Join-Path $Output "isaac_stderr.log")
    }
    $Isaac = Start-Process @IsaacOptions
    $null = $Isaac.Handle
    while (-not $Isaac.WaitForExit(1000)) { }
    if ($null -ne $Isaac.ExitCode -and $Isaac.ExitCode -ne 0) { throw "Isaac failed; see logs in $Output" }
    $Report = Get-Content -LiteralPath (Join-Path $Output "vla_benchmark_report.json") -Raw | ConvertFrom-Json
    if ($Report.status -ne "VLA_BENCHMARK_OK") { throw "Missing completed benchmark evidence" }
    Write-Output "VISUAL_ONLY_STUDENT_BENCHMARK_COMPLETED=$Output"
} finally {
    foreach ($Child in @($Isaac,$Server)) {
        if ($null -ne $Child) {
            $Child.Refresh()
            if (-not $Child.HasExited) {
                Stop-Process -InputObject $Child -ErrorAction SilentlyContinue
                # Termination is asynchronous on Windows. Do not start the next
                # arm while this process still owns its socket/CUDA resources.
                if (-not $Child.WaitForExit(10000)) { Write-Warning "Child process $($Child.Id) did not exit within 10 seconds" }
            }
        }
    }
}
