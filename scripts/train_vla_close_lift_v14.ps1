param([int]$Steps = 6000)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$Results = Join-Path $Root "results\vla\relative4d_close_lift_20261012_v14"
$Diagnostic = Join-Path $Root "results\vla\relative4d_30k_diagnostic_20261012_v14"
$Export = Join-Path $Root "exports\lerobot\pick_lift_relative4d_20261011_v13"
$Checkpoint = Join-Path $Root "results\vla\training\smolvla_relative4d_fresh_lora16_20261011_v13\checkpoints\030000\pretrained_model"
$Run = "smolvla_relative4d_close_lift_lora16_20261012_v14"
$Plan = Join-Path $Results "train_sampling_plan.json"
if ($Steps -ne 6000) {throw "This diagnostic training plan is fixed at 6000 additional optimizer steps"}
if (Test-Path -LiteralPath $Results) {throw "Refusing to overwrite $Results"}
$Reports = @(Get-ChildItem -LiteralPath $Diagnostic -Filter vla_benchmark_report.json -Recurse)
if ($Reports.Count -ne 1) {throw "Exactly one completed 30k diagnostic is required before training"}
$Report = Get-Content $Reports[0].FullName -Raw | ConvertFrom-Json
$Context = Get-Content (Join-Path $Reports[0].DirectoryName 'test_context.json') -Raw | ConvertFrom-Json
if ($Report.status -ne 'VLA_BENCHMARK_OK' -or $Report.overall.num_episodes -ne 6 -or
    $Report.overall.vla_request_count -le 0 -or $Report.overall.vla_error_count -ne 0 -or
    $Report.overall.scripted_ik_fallback_fraction -ne 0 -or $Context.checkpoint_step -ne 30000 -or
    $Context.input_interface -ne 'relative4d' -or
    $Context.checkpoint_sha256 -ne (Get-FileHash (Join-Path $Checkpoint 'adapter_model.safetensors')).Hash) {
    throw "Diagnostic is incomplete or does not match the intended pure-VLA 30k policy"
}
New-Item -ItemType Directory -Path $Results | Out-Null
$env:PYTHONPATH = Join-Path $Root 'src'
$env:PYTHONUTF8='1'; $env:PYTHONUNBUFFERED='1'; $env:HF_HUB_OFFLINE='1'
function Stage([string]$Name, [string]$Message) {
    @{stage=$Name; message=$Message; training_run=$Run; additional_optimizer_steps=$Steps;
      parent_checkpoint=$Checkpoint; parent_training_steps=30000;
      optimizer='fresh AdamW for changed objective'; peak_learning_rate=0.000003;
      z_weight=3; gripper_index=3; gripper_weight=2; phase3_sampling_weight=3;
      phase4_sampling_weight=4; balance_levels=$true; diagnostic_report=$Reports[0].FullName;
      updated_at_utc=[DateTime]::UtcNow.ToString('o'); closed_loop_started=$false
    } | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Results 'pipeline_status.json') -Encoding UTF8
    Write-Output "CLOSE_LIFT_STAGE=$Name $Message"
}
function Run-Python([string[]]$Arguments,[string]$Log) {
    $ErrorActionPreference='Continue'
    & $Python @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $Log
    $Code=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if ($Code -ne 0) {throw "Python stage failed ($Code): $Log"}
}
try {
    Stage 'BUILDING_SAMPLER' 'Double relative sampling mass for close and lift; preserve all approach phases and balance levels.'
    Run-Python @('scripts\build_vla_sampling_plan.py','--dataset-root',(Join-Path $Export 'train'),
        '--output',$Plan,'--balance-levels','--easy-weight','1','--medium-weight','1','--hard-weight','1',
        '--phase-0-weight','0.5','--phase-1-weight','1','--phase-2-weight','1.5',
        '--phase-3-weight','3','--phase-4-weight','4','--dagger-recovery-weight','1','--seed','20261012'
    ) (Join-Path $Results 'sampling.log')
    Stage 'TRAINING_CLOSE_LIFT' 'Load 30k adapter into a separate run; gripper flow-matching weight 2, Z weight 3.'
    & (Join-Path $PSScriptRoot 'train_smolvla_dagger_recovery_8gb_v7.ps1') `
        -RunName $Run -AdapterCheckpoint $Checkpoint -DatasetRoot (Join-Path $Export 'train') `
        -DatasetRepoId 'local/isaac-relative4d-v13-train' -SamplingPlan $Plan `
        -InterfaceContract (Join-Path $Export 'train\interface_contract.json') `
        -Steps $Steps -SaveFreq 2000 -BatchSize 1 -ChunkSize 1 -SchedulerWarmupSteps 200 `
        -SchedulerDecaySteps $Steps -PeakLearningRate 0.000003 -DecayLearningRate 0.000001 `
        -SampleWeightReduction per_sample -ZAxisWeight 3 -GripperIndex 3 -GripperWeight 2 -Seed 1000
    Stage 'SCANNING_FIXED_VALIDATION' 'Same 240 held-out validation indices and noise seed as 30k; compare grip and each phase.'
    Run-Python @('scripts\scan_smolvla_checkpoints.py',
        '--checkpoint-root',(Join-Path $Root "results\vla\training\$Run\checkpoints"),
        '--checkpoint-steps','2000','4000','6000','--dataset-root',(Join-Path $Export 'validation'),
        '--repo-id','local/isaac-relative4d-v13-validation','--samples','240','--seed','20261011',
        '--device','cuda','--batch-size','2','--position-target-mode','delta',
        '--threshold-position-mae-cm','2','--output',(Join-Path $Results 'validation_scan.json')
    ) (Join-Path $Results 'validation_scan.log')
    $Scan=Get-Content (Join-Path $Results 'validation_scan.json') -Raw | ConvertFrom-Json
    $Baseline=Get-Content (Join-Path $Root 'results\vla\relative4d_20261011_v13\validation_scan_30000.json') -Raw | ConvertFrom-Json
    if (($Scan.indices -join ',') -ne ($Baseline.indices -join ',') -or $Scan.seed -ne $Baseline.seed) {
        throw 'Fixed validation indices or diffusion seed changed'
    }
    @{
        status='FIXED_VALIDATION_COMPARISON_COMPLETE'; baseline_training_steps=30000;
        baseline=$Baseline.best_checkpoint; candidates=$Scan.results; best_candidate=$Scan.best_checkpoint;
        position_improvement_cm=($Baseline.best_checkpoint.position_mae_cm - $Scan.best_checkpoint.position_mae_cm);
        gripper_accuracy_gain_points=(100 * ($Scan.best_checkpoint.gripper_sign_accuracy - $Baseline.best_checkpoint.gripper_sign_accuracy));
        same_validation_indices=$true; phase_metrics_are_offline=$true;
        caveat='Sampling and gripper loss changed together, with fresh optimizer; no single-factor causal claim.'
    } | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $Results 'comparison_to_30k.json') -Encoding UTF8
    Stage 'TRAINING_AND_VALIDATION_COMPLETE' "Validation 2 cm gate passed=$($Scan.threshold_passed); review against 30k before further closed-loop experiments."
} catch {Stage 'FAILED' $_.Exception.Message; throw}
