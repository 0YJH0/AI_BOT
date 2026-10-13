param([ValidateRange(1024,65534)][int]$BasePort=8784)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Tag=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Output=Join-Path $Root "results\vla\v9_chunk_ablation_${Tag}_v17"
$Export=Join-Path $Root 'exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9'
$Source=Join-Path $Root 'results\vla\training\smolvla_execution_aligned_full_lora16_20261011_truth_v9\checkpoints\006000\pretrained_model'
$Sampling=Join-Path $Root 'results\vla\execution_aligned_20261011_truth_v9\train_sampling_plan.json'
$Runs=@{4="smolvla_v9_chunk4_${Tag}_v17";1="smolvla_v9_chunk1_${Tag}_v17"}
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$env:PYTHONPATH=Join-Path $Root 'src';$env:PYTHONUTF8='1';$env:PYTHONUNBUFFERED='1';$env:HF_HUB_OFFLINE='1'
@{run_chunk4=$Runs[4];run_chunk1=$Runs[1];parent_checkpoint=$Source;
  steps_per_arm=6000;batch_size=1;seed=1000;sampling_plan=$Sampling;export=$Export;
  peak_lr=0.00001;floor_lr=0.000001;warmup_steps=300;fresh_optimizer_both=$true;
  save_steps=@(2000,4000,6000);validation_samples=240;validation_seed=20261011;
  benchmark_checkpoint=6000;execution_horizon=1;action_repeat=5;close_distance_m=0.075;
  only_factor='prediction chunk_size 4 vs 1';n_action_steps=1;state_dim=30;action_dim=8;
  closed_loop_diagnostic_override='User requested paired chunk training and closed-loop test. Offline gate status remains reported; failing it is not a pass.'
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Output 'experiment_plan.json') -Encoding UTF8
function Stage([string]$Name,[string]$Message) {
    @{stage=$Name;message=$Message;results=$Output;run_chunk4=$Runs[4];run_chunk1=$Runs[1];
      steps_per_arm=6000;planned_closed_loop_episodes=12;updated_at_utc=[DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "CHUNK_ABLATION_STAGE=$Name"
}
function Audit([string]$Mode) {
    & $Python scripts\audit_vla_chunk_ablation_v17.py --root $Output --stage $Mode
    if ($LASTEXITCODE -ne 0) {throw "Chunk audit failed: $Mode"}
}
try {
    Stage 'FREEZING_INPUTS' 'Hash the v9 checkpoint, base model, training/validation data, sampler and source.'
    Audit 'freeze'
    foreach ($Chunk in @(4,1)) {
        Stage "TRAINING_CHUNK_$Chunk" '6000 additional updates, identical v9 adapter, fresh optimizer, unchanged legacy interface/objective.'
        $env:LEROBOT_CHUNK_ABLATION_AUDIT=Join-Path $Output "chunk${Chunk}_initialization.json"
        $env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK=[string]$Chunk
        & (Join-Path $PSScriptRoot 'train_smolvla_dagger_recovery_8gb_v7.ps1') `
            -AdapterCheckpoint $Source -RunName $Runs[$Chunk] -DatasetRoot (Join-Path $Export 'train') `
            -DatasetRepoId 'local/isaac-pick-lift-execution-aligned-20261011_truth_v9-train' `
            -SamplingPlan $Sampling -Steps 6000 -SaveFreq 2000 -BatchSize 1 -Seed 1000 `
            -ChunkSize $Chunk -ActionSteps 1 -SampleWeightReduction normalized -ZAxisWeight 3 `
            -SchedulerWarmupSteps 300 -SchedulerDecaySteps 6000 -PeakLearningRate 0.00001 -DecayLearningRate 0.000001
        Remove-Item Env:LEROBOT_CHUNK_ABLATION_AUDIT,Env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK -ErrorAction SilentlyContinue
        Stage "VALIDATING_CHUNK_$Chunk" 'Same 240 frames; scan 2k/4k/6k, compare first predicted action rather than unequal horizon averages.'
        $ScanArgs=@('scripts\scan_smolvla_checkpoints.py','--checkpoint-root',
            (Join-Path $Root ("results\vla\training\"+$Runs[$Chunk]+'\checkpoints')),
            '--checkpoint-steps','2000','4000','6000','--dataset-root',(Join-Path $Export 'validation'),
            '--repo-id','local/isaac-pick-lift-execution-aligned-20261011_truth_v9-validation',
            '--samples','240','--seed','20261011','--device','cuda','--batch-size','1',
            '--position-target-mode','delta','--output',(Join-Path $Output "chunk${Chunk}_validation.json"))
        & $Python @ScanArgs
        if ($LASTEXITCODE -ne 0) {throw "Validation failed for chunk $Chunk"}
    }
    Stage 'VERIFYING_TRAINING' 'Require identical initial adapter tensors, normalization and training settings except chunk length.'
    Audit 'verify'
    $i=0
    foreach ($Chunk in @(4,1)) {
        Stage "BENCHMARK_CHUNK_$Chunk" 'Final 6k checkpoint; Easy/Medium 3 each, actual lift 10cm, no expert fallback.'
        & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
            -TrainingRun $Runs[$Chunk] -CheckpointStep 6000 -DatasetExportRoot $Export `
            -ValidationReport (Join-Path $Output "chunk${Chunk}_validation.json") `
            -OutputRoot (Join-Path $Output "chunk${Chunk}_closed_loop") -Port ($BasePort+$i) `
            -Levels easy,medium -EpisodesPerLevel 3 -MaxSteps 600 -InputInterface legacy `
            -PhaseMechanism legacy -OrientationMode model -GraspEnableDistanceM 0.075 -RecordPhaseDiagnostics `
            -AllowValidationGateOverride -ValidationGateOverrideReason 'User requested paired chunk ablation; final 6000 checkpoint precommitted. Diagnostic only if 2cm gate fails.'
        $i++
    }
    Stage 'SUMMARIZING' 'Verify paired scenes and unchanged execution settings.'
    Audit 'summarize'
    Stage 'CHUNK_ABLATION_COMPLETE' 'Read training_comparison.json and comparison.json; all original weights remain unchanged.'
} catch {Stage 'FAILED' $_.Exception.Message;throw}
finally {Remove-Item Env:LEROBOT_CHUNK_ABLATION_AUDIT,Env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK -ErrorAction SilentlyContinue}
