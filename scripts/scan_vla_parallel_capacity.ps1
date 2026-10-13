param([string]$RunTag='', [int[]]$EnvironmentCounts=@(1,2,4,8,16), [int]$Waves=2, [int]$MaxSteps=600,
    [ValidateSet('tiled','standard')][string]$CameraMode='tiled',
    [string]$TrainingRun='smolvla_v9_chunk4_20261012_195615_610_v17')
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $RunTag) { $RunTag=Get-Date -Format 'yyyyMMdd_HHmmss_fff' }
if ($RunTag -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Unsafe tag' }
if ($TrainingRun -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Unsafe training run' }
foreach ($N in $EnvironmentCounts) { if ($N -notin @(1,2,4,8,16)) { throw 'Unsupported environment count' } }
$Output=Join-Path $Root "results\vla\parallel_capacity_$RunTag"
if (Test-Path $Output) { throw 'Output already exists' }
New-Item -ItemType Directory -Path $Output | Out-Null
$Checkpoint=Join-Path $Root "results\vla\training\$TrainingRun\checkpoints\006000\pretrained_model"
$Hash=(Get-FileHash (Join-Path $Checkpoint 'adapter_model.safetensors')).Hash
$VlaPython='C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe'
$IsaacPython='C:\Users\yjh\anaconda3\envs\isaaclab23\python.exe'
$env:OMNI_KIT_ACCEPT_EULA='YES'; $env:PYTHONPATH=Join-Path $Root 'src'
$env:PYTHONUTF8='1'; $env:PYTHONUNBUFFERED='1'; $env:HF_HUB_OFFLINE='1'
$Results=@(); $N=0
function Status([string]$Stage,[string]$Message) {
    @{stage=$Stage;message=$Message;current_num_envs=$N;environment_counts=$EnvironmentCounts;camera_mode=$CameraMode;
      checkpoint=$Checkpoint;checkpoint_sha256=$Hash;waves=$Waves;max_steps=$MaxSteps;results=$Output;
      completed=@($Results);driver_pid=$PID;updated_at_utc=[DateTime]::UtcNow.ToString('o')} |
      ConvertTo-Json -Depth 10 | Set-Content (Join-Path $Output 'pipeline_status.json') -Encoding UTF8
}
function GpuSample {
    $Text=& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'GPU telemetry unavailable' }
    $Fields=($Text | Select-Object -First 1).Split(',')
    return @{time_utc=[DateTime]::UtcNow.ToString('o');used_mib=[double]$Fields[0];total_mib=[double]$Fields[1];utilization=[double]$Fields[2];temperature=[double]$Fields[3]}
}
try {
    foreach ($N in $EnvironmentCounts) {
        $Arm=Join-Path $Output "env_$N"
        New-Item -ItemType Directory -Path $Arm | Out-Null
        $Server=$null; $Isaac=$null; $Samples=[System.Collections.Generic.List[object]]::new()
        Status 'RUNNING' "Testing $N environments, shared model, $CameraMode RGB-D, $Waves reset waves."
        $Probe=New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback,0)
        try { $Probe.Start();$Port=$Probe.LocalEndpoint.Port } finally { $Probe.Stop() }
        $Failure=$null
        try {
            $Samples.Add((GpuSample))
            $Server=Start-Process -FilePath $VlaPython -ArgumentList @('scripts\serve_smolvla_windows_v044_v4.py',
                '--checkpoint',('"'+$Checkpoint+'"'),'--port',$Port,'--device','cuda','--execution-horizon',1,
                '--seed',20261011,'--gpu-memory-fraction',0.35,'--profile-memory') -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $Arm 'server_stdout.log') -RedirectStandardError (Join-Path $Arm 'server_stderr.log')
            $Deadline=[DateTime]::UtcNow.AddSeconds(180); $Ready=$false
            while ([DateTime]::UtcNow -lt $Deadline) {
                $Server.Refresh(); if ($Server.HasExited) { throw 'Model server exited during loading' }
                if (Select-String -Path (Join-Path $Arm 'server_stdout.log') -Pattern 'SMOLVLA_SERVER_READY=' -Quiet) { $Ready=$true;break }
                Start-Sleep -Milliseconds 500
            }
            if (-not $Ready) { throw 'Model server readiness timeout' }
            $Isaac=Start-Process -FilePath $IsaacPython -ArgumentList @('scripts\run_vla_parallel_probe.py',
                '--parallel-envs',$N,'--waves',$Waves,'--camera-mode',$CameraMode,'--max-steps',$MaxSteps,'--headless','--device','cuda:0',
                '--rendering_mode','performance','--execution-horizon',1,'--action-repeat',5,
                '--vla-port',$Port,'--vla-timeout',120,'--minimum-true-lift-m',0.10,
                '--vla-position-target-mode','delta','--pure-vla','--require-vla',
                '--output-dir',('"'+$Arm+'"')) -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $Arm 'isaac_stdout.log') -RedirectStandardError (Join-Path $Arm 'isaac_stderr.log')
            $null=$Isaac.Handle
            @{server_pid=$Server.Id;isaac_pid=$Isaac.Id;port=$Port;num_envs=$N} | ConvertTo-Json |
                Set-Content (Join-Path $Arm 'processes.json') -Encoding UTF8
            $Deadline=[DateTime]::UtcNow.AddMinutes(30)
            while (-not $Isaac.WaitForExit(3000)) {
                $Gpu=GpuSample; $Samples.Add($Gpu)
                if ($Gpu.total_mib-$Gpu.used_mib -lt 256) { throw 'Safety stop: less than 256 MiB dedicated VRAM free' }
                $Server.Refresh(); if ($Server.HasExited) { throw 'Model server exited during simulation' }
                if ([DateTime]::UtcNow -gt $Deadline) { throw '30-minute probe timeout' }
            }
            if ($Isaac.ExitCode -ne 0) { throw "Isaac probe exit code $($Isaac.ExitCode)" }
            if (-not (Test-Path (Join-Path $Arm 'parallel_report.json'))) { throw 'Missing complete report' }
        } catch { $Failure=$_.Exception.Message }
        finally {
            foreach ($Child in @($Isaac,$Server)) {
                if ($null -ne $Child) {
                    $Child.Refresh()
                    if (-not $Child.HasExited) { Stop-Process -InputObject $Child -ErrorAction SilentlyContinue; $null=$Child.WaitForExit(10000) }
                }
            }
            $Samples.ToArray() | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Arm 'gpu_samples.json') -Encoding UTF8
        }
        $Report=$null
        if (-not $Failure) { $Report=Get-Content (Join-Path $Arm 'parallel_report.json') -Raw | ConvertFrom-Json }
        if ($Failure -and (Test-Path (Join-Path $Arm 'probe_error.json'))) {
            $Failure=(Get-Content (Join-Path $Arm 'probe_error.json') -Raw | ConvertFrom-Json).error
        }
        $Peak=($Samples | ForEach-Object { $_.used_mib } | Measure-Object -Maximum).Maximum
        $Result=@{num_envs=$N;completed=($null -eq $Failure);error=$Failure;report=(Join-Path $Arm 'parallel_report.json');
            gpu_peak_used_mib=$Peak;gpu_free_at_peak_mib=($Samples[0].total_mib-$Peak);
            operationally_stable=($null -eq $Failure -and $Report.operationally_stable);
            control_env_steps_per_s=$Report.control_env_steps_per_s;mean_batch_inference_ms=$Report.mean_batch_inference_ms}
        $Results+=$Result
        Status 'SCALE_COMPLETE' "Finished environment count $N."
        # Continuing at a larger scale after a resource failure is unsafe/uninformative.
        if ($Failure) { break }
    }
    if ((Get-FileHash (Join-Path $Checkpoint 'adapter_model.safetensors')).Hash -ne $Hash) { throw 'Checkpoint changed' }
    $Stable=@($Results | Where-Object { $_.operationally_stable })
    $Summary=@{status='SCAN_COMPLETE';tested=$Results;max_verified_stable_envs=($Stable | ForEach-Object { $_.num_envs } | Measure-Object -Maximum).Maximum;
        upper_bound_not_established=$true;
        checkpoint=$Checkpoint;checkpoint_sha256=$Hash;waves=$Waves;max_steps=$MaxSteps;camera_mode=$CameraMode;
        caveat='Medium camera capacity probe; not a proof of success-rate equivalence or indefinite stability. Allocator cap remains 35%; dedicated VRAM from nvidia-smi sampled every 3 seconds.'}
    $Summary | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $Output 'capacity_summary.json') -Encoding UTF8
    Status 'COMPLETE' 'Capacity scan ended. See capacity_summary.json for completed scales and any resource boundary.'
} catch { Status 'FAILED' $_.Exception.Message; throw }
