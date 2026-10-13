param([ValidateRange(1024,65529)][int]$BasePort=8790)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Tag=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Output=Join-Path $Root "results\vla\lift_single_factor_${Tag}_v21"
$ParentRun='smolvla_chunk4_rank32_20261012_212747_052_v18'
$Source=Join-Path $Root "results\vla\training\$ParentRun\checkpoints\006000\pretrained_model"
$Export=Join-Path $Root 'exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9'
$Sampling=Join-Path $Root 'results\vla\execution_aligned_20261011_truth_v9\train_sampling_plan.json'
$Runs=@{control="smolvla_lift_control_${Tag}_v21";lift_sampling="smolvla_lift_sampling_${Tag}_v21";lift_z="smolvla_lift_z_${Tag}_v21"}
New-Item -ItemType Directory -Path $Output -ErrorAction Stop|Out-Null
$env:PYTHONPATH=Join-Path $Root 'src';$env:PYTHONUTF8='1';$env:PYTHONUNBUFFERED='1';$env:HF_HUB_OFFLINE='1'
Remove-Item Env:LEROBOT_LOSS_PLATEAU_JSON,Env:LEROBOT_CHUNK_ABLATION_AUDIT,Env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK -ErrorAction SilentlyContinue
@{project=$Root;parent_checkpoint=$Source;export=$Export;sampling_plan=$Sampling;runs=$Runs;
  steps_per_arm=6000;batch_size=1;seed=1000;rank=32;chunk=4;action_steps=1;
  save_steps=@(2000,4000,6000);levels=@('easy','medium','hard');episodes_per_level=3;
  loss_factor='Only current input-phase=4 has Z dimension weight 5 (baseline 3); entire predicted chunk conditioned on that current phase.';
  sampling_factor='Only phase4 sampling multiplier 2->3; retain per-level balancing.';
  plateau='Fixed initial diagnostic budget 6k for all arms; no automatic extension. Compare physical metrics, not weighted loss magnitudes.'
}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $Output 'experiment_plan.json') -Encoding UTF8
function Stage([string]$Name,[string]$Message){
    @{stage=$Name;message=$Message;results=$Output;runs=$Runs;updated_at_utc=[DateTime]::UtcNow.ToString('o')}|
      ConvertTo-Json -Depth 4|Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "LIFT_STAGE=$Name RESULTS=$Output"
}
function Audit([string]$Mode){
    & $Python scripts\analyze_vla_lift_v21.py --root $Output --stage $Mode
    if($LASTEXITCODE -ne 0){throw "Audit failed: $Mode"}
}
function Bench([string]$Run,[string]$Validation,[string]$Destination,[string]$Mode,[int]$Port){
    & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') -TrainingRun $Run -CheckpointStep 6000 `
      -ValidationReport $Validation -DatasetExportRoot $Export -OutputRoot $Destination -Port $Port `
      -Levels easy,medium,hard -EpisodesPerLevel 3 -MaxSteps 600 -RecordPhaseDiagnostics -StudentPhaseMode $Mode `
      -InputInterface legacy -PhaseMechanism legacy -OrientationMode model -GraspEnableDistanceM 0.075 `
      -AllowValidationGateOverride -ValidationGateOverrideReason 'User requested controlled lifting experiments; final 6k physical diagnostics, report offline gate honestly.'
}
try{
    Stage 'FREEZING' 'Freeze parent, train/validation data and experiment code.';Audit 'freeze'
    $ParentValidation=Join-Path $Root 'results\vla\rank32_plateau_20261012_212747_052_v18\validation.json'
    $i=0
    foreach($Mode in @('legacy','grip_feedback')){
      Stage "PHASE_$Mode" 'Same r32 weights; only model phase input differs. Nine paired scenes.'
      Bench $ParentRun $ParentValidation (Join-Path $Output "phase_$Mode") $Mode ($BasePort+$i)
      $i++
    }
    Stage 'ANALYZING_PHASE' 'Evaluate sustained upward actions, geometric hold and actual 10cm success.';Audit 'phase'
    $PhaseReport=Get-Content (Join-Path $Output 'phase_input_report.json') -Raw|ConvertFrom-Json
    $ChosenMode=$PhaseReport.selected_student_phase_mode
    Stage 'BUILDING_SAMPLER' "Shared evaluation phase mode: $ChosenMode. Build train-only lift sampling arm."
    $LiftPlan=Join-Path $Output 'lift_sampling_plan.json'
    & $Python scripts\build_vla_sampling_plan.py --dataset-root (Join-Path $Export 'train') --output $LiftPlan `
      --seed 20261009 --hard-weight 1 --easy-weight 1 --medium-weight 1 --balance-levels `
      --phase-0-weight 0.5 --phase-1-weight 1 --phase-2-weight 1.5 --phase-3-weight 1.5 --phase-4-weight 3 --dagger-recovery-weight 1
    if($LASTEXITCODE -ne 0){throw 'Sampler construction failed'}
    foreach($Arm in @('control','lift_sampling','lift_z')){
      Audit 'verify_inputs'
      Stage "TRAINING_$Arm" 'Same parent, fresh optimizer, 6k updates; one factor differs in each candidate.'
      $env:LEROBOT_CHUNK_ABLATION_AUDIT=Join-Path $Output "${Arm}_initialization.json"
      $env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK='4'
      $ArmPlan=if($Arm -eq 'lift_sampling'){$LiftPlan}else{$Sampling}
      $LiftWeight=if($Arm -eq 'lift_z'){5}else{0}
      & (Join-Path $PSScriptRoot 'train_smolvla_dagger_recovery_8gb_v7.ps1') `
        -AdapterCheckpoint $Source -ExpectedTrainableParams 1485312 -RunName $Runs[$Arm] `
        -DatasetRoot (Join-Path $Export 'train') -DatasetRepoId 'local/isaac-pick-lift-execution-aligned-20261011_truth_v9-train' `
        -SamplingPlan $ArmPlan -Steps 6000 -SaveFreq 2000 -BatchSize 1 -Seed 1000 -ChunkSize 4 -ActionSteps 1 `
        -SampleWeightReduction normalized -ZAxisWeight 3 -LiftZAxisWeight $LiftWeight `
        -SchedulerWarmupSteps 300 -SchedulerDecaySteps 6000 -PeakLearningRate 0.00001 -DecayLearningRate 0.000001
      Remove-Item Env:LEROBOT_CHUNK_ABLATION_AUDIT,Env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK -ErrorAction SilentlyContinue
      Stage "VALIDATING_$Arm" 'Identical held-out 240 frames; 2k/4k/6k first-action metrics.'
      & $Python scripts\scan_smolvla_checkpoints.py --checkpoint-root (Join-Path $Root ("results\vla\training\"+$Runs[$Arm]+'\checkpoints')) `
        --checkpoint-steps 2000 4000 6000 --dataset-root (Join-Path $Export 'validation') `
        --repo-id 'local/isaac-pick-lift-execution-aligned-20261011_truth_v9-validation' --samples 240 --seed 20261011 `
        --device cuda --batch-size 1 --position-target-mode delta --output (Join-Path $Output "${Arm}_validation.json")
      if($LASTEXITCODE -ne 0){throw "Validation failed: $Arm"}
    }
    $i=2
    foreach($Arm in @('control','lift_sampling','lift_z')){
      Audit 'verify_inputs'
      Stage "BENCHMARK_$Arm" "Nine paired scenes, chosen phase mode $ChosenMode; require actual 10cm."
      Bench $Runs[$Arm] (Join-Path $Output "${Arm}_validation.json") (Join-Path $Output "benchmark_$Arm") $ChosenMode ($BasePort+$i)
      $i++
    }
    Stage 'SUMMARIZING' 'Compare same-budget control with isolated sampling and Z-loss changes.';Audit 'final'
    Stage 'COMPLETE' 'Read phase_input_report.json and single_factor_report.json; no baseline overwritten.'
}catch{Stage 'FAILED' $_.Exception.Message;throw}
finally{Remove-Item Env:LEROBOT_CHUNK_ABLATION_AUDIT,Env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK,Env:LEROBOT_LIFT_PHASE_METADATA -ErrorAction SilentlyContinue}
