param([ValidateRange(1024,65534)][int]$BasePort=8782)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$Output=Join-Path $Root ('results\vla\v9_guard_ablation_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'_v16')
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$env:PYTHONUTF8='1'
$Arms=@(@{name='baseline_75mm';distance=0.075},@{name='strict_25mm';distance=0.025})
function Stage([string]$Name,[string]$Message) {
    @{stage=$Name;message=$Message;results=$Output;planned_total_episodes=12;retraining=$false;
      factor='grasp_enable_distance_m';updated_at_utc=[DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
}
try {
    Stage 'LOCKING_V9' 'One factor: first-close distance 7.5 cm vs 2.5 cm. Latch, weights and phase unchanged.'
    & $Python scripts\audit_vla_v9_regression.py --output $Output --freeze
    if ($LASTEXITCODE -ne 0) {throw 'Baseline fingerprint lock failed'}
    @{arms=$Arms;checkpoint='smolvla_execution_aligned_full_lora16_20261011_truth_v9/006000';
      levels=@('easy','medium');episodes_per_level=3;max_steps=600;inference_seed=20261011;
      prediction_chunk=4;execution_horizon=1;action_repeat=5;interface='legacy';
      phase='legacy';orientation='model';gripper_latch='unchanged, hold until reset';
      hypothesis='Tighter first-close gate may prevent empty grasps, but may also block valid grasps or fail to repair bad approach actions.';
      decision='Diagnostic only; no automatic replacement of defaults or training.'
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Output 'experiment_plan.json') -Encoding UTF8
    for ($i=0; $i -lt $Arms.Count; $i++) {
        $Arm=$Arms[$i]
        $ArmOutput=Join-Path $Output $Arm.name
        Stage $Arm.name 'Same v9 weights and six scenes; only first-close permission distance differs.'
        & (Join-Path $PSScriptRoot 'benchmark_vla_truth_student_v9.ps1') `
            -TrainingRun 'smolvla_execution_aligned_full_lora16_20261011_truth_v9' -CheckpointStep 6000 `
            -ValidationReport 'results\vla\fresh_lora_comparison_20261011_v10\continued_validation240.json' `
            -OutputRoot $ArmOutput -Port ($BasePort+$i) -Levels easy,medium -EpisodesPerLevel 3 -MaxSteps 600 `
            -PhaseMechanism legacy -OrientationMode model -InputInterface legacy -RecordPhaseDiagnostics `
            -GraspEnableDistanceM $Arm.distance
        & $Python scripts\compare_vla_v9_guard_ablation.py --validate-root $ArmOutput
        if ($LASTEXITCODE -ne 0) {throw 'Invalid diagnostic arm: preserve logs and rerun; do not count infrastructure failures'}
    }
    Stage 'COMPARING' 'Verify model/processors/source lock and all scene/control parameters.'
    & $Python scripts\compare_vla_v9_guard_ablation.py --root $Output
    if ($LASTEXITCODE -ne 0) {throw 'Single-factor verification failed'}
    Stage 'SINGLE_FACTOR_COMPLETE' 'Read comparison.json. No defaults changed, no training started.'
} catch {Stage 'FAILED' $_.Exception.Message;throw}
