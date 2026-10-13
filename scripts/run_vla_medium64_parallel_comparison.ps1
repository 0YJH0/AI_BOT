param([string]$RunTag='', [ValidateSet(1,2,4,8,16)][int]$NumEnvs=4, [switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $RunTag) { $RunTag=Get-Date -Format 'yyyyMMdd_HHmmss_fff' }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid run tag' }
$Waves=[int](64/$NumEnvs)
$Output=Join-Path $Root "results\vla\medium64_parallel${NumEnvs}_$RunTag"
$Arms=@(
    @{name='v18_r32_chunk4_execute1';run='smolvla_chunk4_rank32_20261012_212747_052_v18';rank=32;horizon=1},
    @{name='v17_r16_chunk4_execute1';run='smolvla_v9_chunk4_20261012_195615_610_v17';rank=16;horizon=1},
    @{name='v17_r16_chunk4_execute2';run='smolvla_v9_chunk4_20261012_195615_610_v17';rank=16;horizon=2},
    @{name='v17_r16_chunk4_execute4';run='smolvla_v9_chunk4_20261012_195615_610_v17';rank=16;horizon=4}
)
foreach ($Arm in $Arms) {
    $Arm.checkpoint=Join-Path $Root "results\vla\training\$($Arm.run)\checkpoints\006000\pretrained_model"
    $Config=Get-Content (Join-Path $Arm.checkpoint 'config.json') -Raw | ConvertFrom-Json
    $Adapter=Get-Content (Join-Path $Arm.checkpoint 'adapter_config.json') -Raw | ConvertFrom-Json
    if ($Config.chunk_size -ne 4 -or $Adapter.r -ne $Arm.rank -or
        $Config.input_features.'observation.state'.shape[0] -ne 30 -or
        $Config.output_features.action.shape[0] -ne 8) { throw "Checkpoint interface mismatch: $($Arm.name)" }
    $Arm.checkpoint_sha256=(Get-FileHash (Join-Path $Arm.checkpoint 'adapter_model.safetensors')).Hash
}
if ($PreflightOnly) { $Arms | ConvertTo-Json -Depth 8; exit 0 }
if (Test-Path -LiteralPath $Output) { throw 'Refusing to overwrite existing results' }
New-Item -ItemType Directory -Path $Output | Out-Null
$VlaPython='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$IsaacPython='C:\Users\yjh\anaconda3\envs\isaaclab23\python.exe'
$env:OMNI_KIT_ACCEPT_EULA='YES'
$env:PYTHONPATH=Join-Path $Root 'src'
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$env:HF_HUB_OFFLINE='1'
$Protocol=@{difficulty='medium';scene_seed=7102;num_envs=$NumEnvs;waves_per_arm=$Waves;episodes_per_arm=64;
    planned_total_episodes=256;max_steps=600;chunk_size=4;action_repeat=5;
    minimum_actual_lift_m=0.10;phase_mechanism='legacy';orientation_mode='model';
    input='RGB_language_proprio_visual_estimate_phase';grasp_enable_distance_m=0.075;
    retraining=$false;scripted_fallback=$false;truth_in_student_input=$false;
    inference_seed=20261011;headless=$true;camera="standard Camera RGB-D 320x240 ($NumEnvs render products)";
    env_spacing_m=6.0;gpu_memory_fraction=0.35;results=$Output;
    caveat='Same 64 scene samples and wave grouping. Different horizons have different inference schedules/noise draws. Rank arms have different training histories. Not directly pooled with old single-camera runs.'}
$SourceHashes=@{}
foreach ($File in @('scripts\run_vla_parallel_probe.py','scripts\run_vla_benchmark.py',
    'scripts\serve_smolvla.py','scripts\serve_smolvla_windows_v044_v4.py',
    'src\isaac_lab_data_engine\vla\batch_execution.py','src\isaac_lab_data_engine\vla\hybrid_controller.py',
    'src\isaac_lab_data_engine\randomization\domain_randomizer.py',
    'configs\benchmark.yaml','configs\randomization_medium.yaml',
    'results\vision\rgbd_keypoint_supplement_160x120_v3\best_model.pt')) {
    $SourceHashes[$File]=(Get-FileHash (Join-Path $Root $File)).Hash
}
@{protocol=$Protocol;arms=$Arms;source_sha256=$SourceHashes} | ConvertTo-Json -Depth 12 |
    Set-Content (Join-Path $Output 'experiment_plan.json') -Encoding UTF8
$Results=@(); $CurrentArm=$null; $ReferenceManifest=$null
function Status([string]$Stage,[string]$Message) {
    @{stage=$Stage;message=$Message;current_arm=$CurrentArm;results=$Output;
      completed_arms=@($Results);completed_episodes=64*$Results.Count;planned_total_episodes=256;
      difficulty='medium';num_envs=$NumEnvs;episodes_per_arm=64;driver_pid=$PID;
      updated_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 12 |
      Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
    Write-Output "MEDIUM64_STAGE=$Stage ARM=$CurrentArm MESSAGE=$Message"
}
function GpuSample {
    $Line=& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'GPU telemetry unavailable' }
    $Fields=($Line | Select-Object -First 1).Split(',')
    return @{time_utc=[DateTime]::UtcNow.ToString('o');used_mib=[double]$Fields[0];
        total_mib=[double]$Fields[1];utilization=[double]$Fields[2]}
}
try {
    foreach ($Arm in $Arms) {
        $CurrentArm=$Arm.name
        foreach ($File in $SourceHashes.Keys) {
            if ((Get-FileHash (Join-Path $Root $File)).Hash -ne $SourceHashes[$File]) { throw "Source changed: $File" }
        }
        if ((Get-FileHash (Join-Path $Arm.checkpoint 'adapter_model.safetensors')).Hash -ne $Arm.checkpoint_sha256) {
            throw "Checkpoint changed: $CurrentArm"
        }
        $ArmOutput=Join-Path $Output $CurrentArm
        New-Item -ItemType Directory -Path $ArmOutput | Out-Null
        $Server=$null; $Isaac=$null; $Samples=[System.Collections.Generic.List[object]]::new()
        $Failure=$null
        Status 'STARTING' "Loading $CurrentArm; 64 Medium episodes in $Waves waves of $NumEnvs."
        $PortProbe=New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback,0)
        try { $PortProbe.Start(); $Port=$PortProbe.LocalEndpoint.Port } finally { $PortProbe.Stop() }
        try {
            # Always return all four predictions; client alone selects how many to execute.
            $Server=Start-Process -FilePath $VlaPython -ArgumentList @('scripts\serve_smolvla_windows_v044_v4.py',
                '--checkpoint',('"'+$Arm.checkpoint+'"'),'--port',$Port,'--device','cuda',
                '--execution-horizon',4,'--seed',20261011,'--gpu-memory-fraction',0.35,'--profile-memory') `
                -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $ArmOutput 'server_stdout.log') `
                -RedirectStandardError (Join-Path $ArmOutput 'server_stderr.log')
            $Deadline=[DateTime]::UtcNow.AddSeconds(180); $Ready=$false
            while ([DateTime]::UtcNow -lt $Deadline) {
                $Server.Refresh(); if ($Server.HasExited) { throw 'Model server exited while loading' }
                if (Select-String -Path (Join-Path $ArmOutput 'server_stdout.log') -Pattern 'SMOLVLA_SERVER_READY=' -Quiet) {
                    $Ready=$true; break
                }
                Start-Sleep -Milliseconds 500
            }
            if (-not $Ready) { throw 'Model readiness timeout' }
            $Isaac=Start-Process -FilePath $IsaacPython -ArgumentList @('scripts\run_vla_parallel_probe.py',
                '--parallel-envs',$NumEnvs,'--waves',$Waves,'--strict-benchmark','--camera-mode','standard','--max-steps',600,
                '--headless','--device','cuda:0','--rendering_mode','performance',
                '--execution-horizon',$Arm.horizon,'--action-repeat',5,'--vla-port',$Port,'--vla-timeout',120,
                '--minimum-true-lift-m',0.10,'--vla-position-target-mode','delta','--pure-vla','--require-vla',
                '--output-dir',('"'+$ArmOutput+'"')) -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $ArmOutput 'isaac_stdout.log') `
                -RedirectStandardError (Join-Path $ArmOutput 'isaac_stderr.log')
            $null=$Isaac.Handle
            @{server_pid=$Server.Id;isaac_pid=$Isaac.Id;port=$Port} | ConvertTo-Json |
                Set-Content (Join-Path $ArmOutput 'processes.json') -Encoding UTF8
            Status 'RUNNING' "Executing $CurrentArm; see live_progress.json / progress.json inside this arm."
            $Deadline=[DateTime]::UtcNow.AddHours(2)
            while (-not $Isaac.WaitForExit(3000)) {
                $Gpu=GpuSample; $Samples.Add($Gpu)
                if ($Gpu.total_mib-$Gpu.used_mib -lt 256) { throw 'Safety stop: dedicated VRAM free below 256 MiB' }
                $Server.Refresh(); if ($Server.HasExited) { throw 'Model server exited during benchmark' }
                if ([DateTime]::UtcNow -gt $Deadline) { throw 'Arm exceeded two-hour timeout' }
            }
            if ($Isaac.ExitCode -ne 0) { throw "Isaac exit code $($Isaac.ExitCode)" }
            if (-not (Test-Path (Join-Path $ArmOutput 'parallel_report.json'))) { throw 'Missing complete benchmark report' }
        } catch { $Failure=$_.Exception.Message }
        finally {
            foreach ($Child in @($Isaac,$Server)) {
                if ($null -ne $Child) {
                    $Child.Refresh()
                    if (-not $Child.HasExited) { Stop-Process -InputObject $Child -ErrorAction SilentlyContinue; $null=$Child.WaitForExit(10000) }
                }
            }
            $Samples.ToArray() | ConvertTo-Json -Depth 4 |
                Set-Content (Join-Path $ArmOutput 'gpu_samples.json') -Encoding UTF8
        }
        if ($Failure) {
            if (Test-Path (Join-Path $ArmOutput 'probe_error.json')) {
                $Failure=(Get-Content (Join-Path $ArmOutput 'probe_error.json') -Raw | ConvertFrom-Json).error
            }
            throw $Failure
        }
        $Report=Get-Content (Join-Path $ArmOutput 'parallel_report.json') -Raw | ConvertFrom-Json
        if (-not $Report.operationally_stable -or $Report.episodes.Count -ne 64 -or
            $Report.num_envs -ne $NumEnvs -or $Report.waves -ne $Waves -or
            $Report.difficulty -ne 'medium' -or $Report.execution_horizon -ne $Arm.horizon -or
            $Report.scripted_fallback) { throw 'Incomplete/invalid benchmark; do not report a 64-episode success rate' }
        $ManifestHash=(Get-FileHash (Join-Path $ArmOutput 'scene_manifest.json')).Hash
        if ($null -eq $ReferenceManifest) { $ReferenceManifest=$ManifestHash }
        elseif ($ManifestHash -ne $ReferenceManifest) { throw 'Scene samples differ across arms' }
        if ((Get-FileHash (Join-Path $Arm.checkpoint 'adapter_model.safetensors')).Hash -ne $Arm.checkpoint_sha256) {
            throw 'Checkpoint changed during evaluation'
        }
        $Failures=@{}
        foreach ($Row in $Report.episodes) {
            if (-not $Row.success) { $Failures[$Row.failure_reason]=1+$Failures[$Row.failure_reason] }
        }
        $CloseDistances=@($Report.episodes | Where-Object { $null -ne $_.first_close } | ForEach-Object { $_.first_close.true_distance_cm })
        $Results+=@{arm=$CurrentArm;execution_horizon=$Arm.horizon;episodes=64;successes=$Report.successes;
            success_rate=$Report.success_rate;checkpoint=$Arm.checkpoint;checkpoint_sha256=$Arm.checkpoint_sha256;
            report=(Join-Path $ArmOutput 'parallel_report.json');scene_manifest_sha256=$ManifestHash;
            failure_counts=$Failures;mean_first_close_true_distance_cm=($CloseDistances | Measure-Object -Average).Average;
            episodes_with_close=$CloseDistances.Count;
            early_close_requests_over_2_5cm=($Report.episodes | ForEach-Object { $_.early_close_requests_over_2_5cm } | Measure-Object -Sum).Sum;
            mean_maximum_lift_cm=($Report.episodes | ForEach-Object { $_.maximum_lift_cm } | Measure-Object -Average).Average;
            mean_batch_inference_ms=$Report.mean_batch_inference_ms;elapsed_s=$Report.elapsed_s;
            gpu_peak_used_mib=($Samples | ForEach-Object { $_.used_mib } | Measure-Object -Maximum).Maximum}
        @{status='PARTIAL';protocol=$Protocol;results=$Results} | ConvertTo-Json -Depth 12 |
            Set-Content (Join-Path $Output 'comparison_summary.json') -Encoding UTF8
        Status 'ARM_COMPLETE' "Completed $CurrentArm; proceeding automatically."
    }
    @{status='COMPLETE';protocol=$Protocol;results=$Results} | ConvertTo-Json -Depth 12 |
        Set-Content (Join-Path $Output 'comparison_summary.json') -Encoding UTF8
    $CurrentArm=$null
    Status 'COMPLETE' 'All four arms completed 64 Medium episodes each; original weights unchanged.'
} catch { Status 'FAILED' $_.Exception.Message; throw }
