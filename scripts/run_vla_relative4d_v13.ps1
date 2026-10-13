param([string]$RunTag="20261011_v13", [int]$Steps=6000, [int]$Port=8775, [switch]$Resume)
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$' -or $Steps -lt 500) {throw "Unsafe settings"}
$Results=Join-Path $Root ("results\vla\relative4d_"+$RunTag)
$Export=Join-Path $Root ("exports\lerobot\pick_lift_relative4d_"+$RunTag)
$Pilot=Join-Path $Root ("dataset\vla_relative4d_teacher_pilot_"+$RunTag)
$TrainingRun="smolvla_relative4d_fresh_lora16_"+$RunTag
$VlaPython="C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$IsaacPython="C:\Users\yjh\anaconda3\envs\isaaclab23\python.exe"
$Source=Join-Path $Root "exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9"
foreach ($Path in @($Results,$Export,$Pilot)) {
    if ((Test-Path -LiteralPath $Path) -and -not $Resume) {throw "Refusing to overwrite $Path"}
}
if (Test-Path -LiteralPath (Join-Path $Root ("results\vla\training\"+$TrainingRun))) {throw "This resume supports pre-training stages only; never overwrite an existing training run"}
if ($Resume) {
    if (-not (Test-Path -LiteralPath $Results)) {throw "No experiment to resume"}
    Copy-Item -LiteralPath (Join-Path $Results "pipeline_status.json") -Destination (Join-Path $Results ("pipeline_status_before_resume_"+(Get-Date -Format "yyyyMMdd_HHmmss_fff")+".json"))
    if (Test-Path -LiteralPath (Join-Path $Results "sampling.log")) {
        Copy-Item -LiteralPath (Join-Path $Results "sampling.log") -Destination (Join-Path $Results ("sampling_before_resume_"+(Get-Date -Format "yyyyMMdd_HHmmss_fff")+".log"))
    }
} else { New-Item -ItemType Directory -Path $Results | Out-Null }
$env:PYTHONPATH=Join-Path $Root "src"
$env:PYTHONUTF8="1"; $env:PYTHONUNBUFFERED="1"; $env:HF_HUB_OFFLINE="1"; $env:OMNI_KIT_ACCEPT_EULA="YES"
function Set-Stage([string]$Stage,[string]$Message="") {
    @{stage=$Stage; message=$Message; updated_at_utc=[DateTime]::UtcNow.ToString("o");
      results=$Results; training_run=$TrainingRun; steps=$Steps; state_dim=29; action_dim=4;
      phase_in_model_input=$false; teacher_truth_in_student_input=$false;
      initialization="original open-source SmolVLA plus newly created rank16 LoRA";
      source_export=$Source; relative_axes="robot_base"; closed_loop_gate_cm=2
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Results "pipeline_status.json") -Encoding UTF8
    Write-Output "RELATIVE4D_STAGE=$Stage $Message"
}
function Invoke-Logged([string]$Exe,[string[]]$ArgsList,[string]$Log) {
    $OldPreference=$ErrorActionPreference
    $ErrorActionPreference="Continue"
    & $Exe @ArgsList 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $Log
    $Code=$LASTEXITCODE
    $ErrorActionPreference=$OldPreference
    if ($Code -ne 0) {throw "Stage failed with code $Code; see $Log"}
}
try {
    $Repair=Get-Content (Join-Path $Root "results\vla\phase_ablation_a_repair_20261011_v12\pipeline_status.json") -Raw | ConvertFrom-Json
    if ($Repair.stage -ne "RERUN_COMPLETE") {throw "A repair must finish before this serial GPU workflow"}
    Set-Stage "AUDITING_OLD_POLICY_PHASE_ERRORS" "Fixed validation frames, unchanged old checkpoint."
    if (-not ($Resume -and (Test-Path -LiteralPath (Join-Path $Results "old_phase_validation.json")))) {
    Invoke-Logged $VlaPython @("scripts\scan_smolvla_checkpoints.py",
        "--checkpoint-root","results\vla\training\smolvla_execution_aligned_full_lora16_20261011_truth_v9\checkpoints",
        "--checkpoint-steps","6000","--dataset-root",(Join-Path $Source "validation"),
        "--repo-id","local/isaac-pick-lift-execution-aligned-20261011_truth_v9-validation",
        "--samples","240","--seed","20261011","--device","cuda","--batch-size","2",
        "--position-target-mode","delta","--output",(Join-Path $Results "old_phase_validation.json")
    ) (Join-Path $Results "old_phase_validation.log")
    } else {
        $OldScan=Get-Content (Join-Path $Results "old_phase_validation.json") -Raw | ConvertFrom-Json
        if ($OldScan.status -ne "VLA_CHECKPOINT_SCAN_COMPLETE" -or $OldScan.results[0].step -ne 6000) {throw "Old scan incomplete"}
    }
    Set-Stage "VERIFYING_4D_TEACHER_EXECUTION" "Truth teacher pilot: 2 complete successes per difficulty, fixed quaternion, same 4 cm/5-step guard."
    if (-not ($Resume -and (Test-Path -LiteralPath (Join-Path $Pilot "collection_summary.json")))) {
    Invoke-Logged $IsaacPython @("scripts\run_vla_benchmark.py","--headless","--levels","easy","medium","hard",
        "--collect-full-trajectories",$Pilot,"--collection-teacher","truth",
        "--collection-target-per-level","2","--collection-max-attempts-per-level","10",
        "--collection-seed-offset","20262012","--max-steps","600","--minimum-true-lift-m","0.10",
        "--vla-input-interface","relative4d","--vla-position-target-mode","delta",
        "--execution-horizon","1","--action-repeat","5","--visual-init-attempts","1",
        "--camera-ready-timeout-s","45","--visual-invalid-pose-timeout-s","8",
        "--device","cuda:0","--rendering_mode","performance","--output-dir",(Join-Path $Results "teacher_pilot")
    ) (Join-Path $Results "teacher_pilot.log")
    }
    $Collection=Get-Content (Join-Path $Pilot "collection_summary.json") -Raw | ConvertFrom-Json
    if ($Collection.status -ne "FULL_TRAJECTORY_COLLECTION_OK") {throw "Incomplete teacher pilot"}
    foreach ($Level in @("easy","medium","hard")) {
        if ($Collection.successes.$Level -ne 2) {throw "Teacher execution gate failed for $Level"}
    }
    Set-Stage "EXPORTING_PROJECTED_INTERFACE" "Only original verified 600 episodes; pilot is not added to training; unchanged RGB and executed xyz/gripper."
    if (-not ($Resume -and (Test-Path -LiteralPath (Join-Path $Export "loader_audit.json")))) {
        Invoke-Logged $VlaPython @("scripts\export_vla_relative4d_v13.py","--source",$Source,"--output",$Export) (Join-Path $Results "export.log")
        Invoke-Logged $VlaPython @("scripts\verify_vla_relative4d_v13.py","--root",$Export) (Join-Path $Results "loader_audit.log")
    } else {
        $Audit=Get-Content (Join-Path $Export "loader_audit.json") -Raw | ConvertFrom-Json
        if ($Audit.status -ne "RELATIVE4D_LEROBOT_LOADER_VERIFIED") {throw "Dataset audit incomplete"}
        $Manifest=Get-Content (Join-Path $Export "train\balanced_split_manifest.json") -Raw | ConvertFrom-Json
        if ($Manifest.source_export -ne (Join-Path $Source "train") -or $Manifest.action_dim -ne 4 -or $Manifest.state_dim -ne 29) {throw "Different source or interface"}
    }
    $Plan=Join-Path $Results "train_sampling_plan.json"
    Invoke-Logged $VlaPython @("scripts\build_vla_sampling_plan.py","--dataset-root",(Join-Path $Export "train"),
        "--output",$Plan,"--balance-levels","--easy-weight","1","--medium-weight","1","--hard-weight","1",
        "--phase-0-weight","0.5","--phase-1-weight","1","--phase-2-weight","1.5",
        "--phase-3-weight","1.5","--phase-4-weight","2","--dagger-recovery-weight","1"
    ) (Join-Path $Results "sampling.log")
    Set-Stage "TRAINING_RELATIVE4D" "Fresh LoRA, new 29/4-D statistics, chunk1 matches execution; no teacher phase input."
    & (Join-Path $PSScriptRoot "train_smolvla_dagger_recovery_8gb_v7.ps1") -FreshLora `
        -RunName $TrainingRun -DatasetRoot (Join-Path $Export "train") -DatasetRepoId "local/isaac-relative4d-v13-train" `
        -SamplingPlan $Plan -InterfaceContract (Join-Path $Export "train\interface_contract.json") `
        -Steps $Steps -SaveFreq 500 -BatchSize 1 -ChunkSize 1 -SchedulerWarmupSteps 300 -SchedulerDecaySteps $Steps `
        -SampleWeightReduction per_sample -ZAxisWeight 3 -Seed 1000
    Set-Stage "SCANNING_VALIDATION" "Same fixed 240 held-out frames; test split never used for model selection."
    Invoke-Logged $VlaPython @("scripts\scan_smolvla_checkpoints.py",
        "--checkpoint-root",(Join-Path $Root ("results\vla\training\"+$TrainingRun+"\checkpoints")),
        "--checkpoint-steps","500","3000",[string]$Steps,
        "--dataset-root",(Join-Path $Export "validation"),"--repo-id","local/isaac-relative4d-v13-validation",
        "--samples","240","--seed","20261011","--device","cuda","--batch-size","2",
        "--position-target-mode","delta","--threshold-position-mae-cm","2",
        "--output",(Join-Path $Results "validation_scan.json")
    ) (Join-Path $Results "validation_scan.log")
    $Scan=Get-Content (Join-Path $Results "validation_scan.json") -Raw | ConvertFrom-Json
    if (-not $Scan.threshold_passed) {
        Set-Stage "VALIDATION_GATE_NOT_PASSED" "No automatic closed-loop test; position error still exceeds 2 cm."
        return
    }
    Set-Stage "RELATIVE4D_CLOSED_LOOP" "Pure VLA, 3 scenes per difficulty; no truth teacher or fallback."
    & (Join-Path $PSScriptRoot "benchmark_vla_truth_student_v9.ps1") -RunTag "20261011_truth_v9" `
        -TrainingRun $TrainingRun -ValidationReport (Join-Path $Results "validation_scan.json") `
        -DatasetExportRoot $Export -InputInterface relative4d -OutputRoot (Join-Path $Results "closed_loop") `
        -Port $Port -EpisodesPerLevel 3 -MaxSteps 600 -RecordPhaseDiagnostics
    Set-Stage "RELATIVE4D_COMPLETE" "Training, held-out scan and student benchmark complete; inspect measured successes."
} catch {
    Set-Stage "FAILED" $_.Exception.Message
    throw
}
