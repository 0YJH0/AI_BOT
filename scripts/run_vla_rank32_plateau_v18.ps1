param([int]$MaxSteps=30000)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Tag=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Output=Join-Path $Root "results\vla\rank32_plateau_${Tag}_v18"
$Run="smolvla_chunk4_rank32_${Tag}_v18"
$Source=Join-Path $Root 'results\vla\training\smolvla_v9_chunk4_20261012_195615_610_v17\checkpoints\006000\pretrained_model'
$Expanded=Join-Path $Output 'initial_rank32'
$Export=Join-Path $Root 'exports\lerobot\pick_lift_execution_aligned_20261011_truth_v9'
$Sampling=Join-Path $Root 'results\vla\execution_aligned_20261011_truth_v9\train_sampling_plan.json'
if ($MaxSteps -lt 6000 -or $MaxSteps % 1000 -ne 0) {throw 'MaxSteps must be a multiple of 1000, at least 6000'}
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$env:PYTHONPATH=Join-Path $Root 'src'
$env:PYTHONUTF8='1';$env:PYTHONUNBUFFERED='1';$env:HF_HUB_OFFLINE='1'
Remove-Item Env:LEROBOT_CHUNK_ABLATION_AUDIT,Env:LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK -ErrorAction SilentlyContinue
function Stage([string]$Name,[string]$Message) {
    @{stage=$Name;message=$Message;results=$Output;training_run=$Run;max_steps=$MaxSteps;
      updated_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json |
        Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "RANK32_STAGE=$Name RESULTS=$Output"
}
$StopConfig=@{output=$Output;window=1000;min_steps=6000;patience=3;relative=0.01}
@{source=$Source;rank=32;alpha=16;chunk_size=4;execution_horizon=1;batch_size=1;
  max_steps=$MaxSteps;fresh_optimizer=$true;peak_lr=0.00001;floor_lr=0.000001;
  warmup_steps=300;scheduler_decay_steps=$MaxSteps;sampling_plan=$Sampling;early_stop=$StopConfig;
  note='Single r32 continuation, not a paired rank ablation. Epoch counts are nominal with replacement sampling.'
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Output 'experiment_plan.json') -Encoding UTF8
try {
    Stage 'EXPANDING_RANK' 'Copy original checkpoint; verify all effective adapter matrices and projection outputs.'
    & $Python scripts\expand_smolvla_lora_rank.py --source $Source --output $Expanded
    if ($LASTEXITCODE -ne 0) {throw 'Rank expansion failed'}
    $env:LEROBOT_LOSS_PLATEAU_JSON=$StopConfig | ConvertTo-Json -Compress
    Stage 'TRAINING_R32' 'Up to 30k additional updates; save every 1k; plateau stops only after complete checkpoint.'
    & (Join-Path $PSScriptRoot 'train_smolvla_dagger_recovery_8gb_v7.ps1') `
        -AdapterCheckpoint $Expanded -ExpectedTrainableParams 1485312 -RunName $Run `
        -DatasetRoot (Join-Path $Export 'train') `
        -DatasetRepoId 'local/isaac-pick-lift-execution-aligned-20261011_truth_v9-train' `
        -SamplingPlan $Sampling -Steps $MaxSteps -SaveFreq 1000 -BatchSize 1 -Seed 1000 `
        -ChunkSize 4 -ActionSteps 1 -SampleWeightReduction normalized -ZAxisWeight 3 `
        -SchedulerWarmupSteps 300 -SchedulerDecaySteps $MaxSteps -PeakLearningRate 0.00001 -DecayLearningRate 0.000001
    Remove-Item Env:LEROBOT_LOSS_PLATEAU_JSON -ErrorAction SilentlyContinue
    $Progress=Get-Content (Join-Path $Output 'training_progress.json') -Raw | ConvertFrom-Json
    if ($Progress.status -notin @('EARLY_STOPPED','CHECKPOINT_SAVED')) {throw 'No safely saved terminal checkpoint'}
    $FinalStep=[int]$Progress.step
    $ScanSteps=@(6000,12000,18000,24000,30000,$FinalStep) | Where-Object {$_ -le $FinalStep} | Sort-Object -Unique
    Stage 'VALIDATING' "Training ended at $FinalStep ($($Progress.status)); fixed 240-frame validation."
    $ScanArgs=@('scripts\scan_smolvla_checkpoints.py','--checkpoint-root',
        (Join-Path $Root "results\vla\training\$Run\checkpoints"),'--checkpoint-steps')
    $ScanArgs+=@($ScanSteps | ForEach-Object {[string]$_})
    $ScanArgs+=@('--dataset-root',(Join-Path $Export 'validation'),
        '--repo-id','local/isaac-pick-lift-execution-aligned-20261011_truth_v9-validation',
        '--samples','240','--seed','20261011','--device','cuda','--batch-size','1',
        '--position-target-mode','delta','--output',(Join-Path $Output 'validation.json'))
    & $Python @ScanArgs
    if ($LASTEXITCODE -ne 0) {throw 'Fixed validation failed'}
    Stage 'TRAINING_AND_VALIDATION_COMPLETE' "Stopped at $FinalStep; review validation before any new closed-loop benchmark."
} catch {Stage 'FAILED' $_.Exception.Message;throw}
finally {Remove-Item Env:LEROBOT_LOSS_PLATEAU_JSON -ErrorAction SilentlyContinue}
