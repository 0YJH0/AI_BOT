param(
    [string]$Level = "easy",
    [int]$Port = 8766,
    [int]$MaxSteps = 300,
    [ValidateSet("cpu", "cuda")][string]$VlaDevice = "cuda",
    [double]$GpuMemoryFraction = 0.35,
    [switch]$NoKeepOpen,
    [switch]$KeepOpen,
    [switch]$Headless
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VlaPython = "C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe"
$IsaacPython = "C:\Users\yjh\anaconda3\envs\isaaclab23\python.exe"
$Checkpoint = Join-Path $ProjectRoot "results\vla\training\smolvla_visual_fsm_delta_tight_dagger15mm_lora16_v7\checkpoints\001500\pretrained_model"
if ($Level -notin @("easy", "medium", "hard")) { throw "Level must be easy, medium, or hard" }
if ($Port -lt 1 -or $Port -gt 65535 -or $MaxSteps -lt 1) { throw "Invalid port or step count" }
if ($GpuMemoryFraction -le 0 -or $GpuMemoryFraction -gt 1) { throw "GPU memory fraction must be in (0,1]" }
foreach ($RequiredPath in @($VlaPython, $IsaacPython, (Join-Path $Checkpoint "config.json"), (Join-Path $Checkpoint "adapter_model.safetensors"))) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) { throw "Missing required file: $RequiredPath" }
}
$Probe = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
try { $Probe.Start() } finally { $Probe.Stop() }
$RunName = "tight_v7_001500_gui_${Level}_${VlaDevice}_steps${MaxSteps}_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff")
$OutputDir = Join-Path $ProjectRoot ("results\vla\closed_loop_benchmark\" + $RunName)
New-Item -ItemType Directory -Path $OutputDir -ErrorAction Stop | Out-Null
$ServerOut = Join-Path $OutputDir "server_stdout.log"
$ServerErr = Join-Path $OutputDir "server_stderr.log"
$IsaacOut = Join-Path $OutputDir "isaac_stdout.log"
$IsaacErr = Join-Path $OutputDir "isaac_stderr.log"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$Server = $null
$Isaac = $null
try {
    $ServerArgs = @("scripts\serve_smolvla_windows_v044_v4.py", "--checkpoint", ('"' + $Checkpoint + '"'), "--port", $Port, "--execution-horizon", 1, "--device", $VlaDevice, "--seed", 20261010)
    if ($VlaDevice -eq "cuda") { $ServerArgs += @("--gpu-memory-fraction", $GpuMemoryFraction, "--profile-memory") }
    $Server = Start-Process -FilePath $VlaPython -ArgumentList $ServerArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $ServerOut -RedirectStandardError $ServerErr
    $Deadline = [DateTime]::UtcNow.AddSeconds(180)
    $Ready = $false
    while ([DateTime]::UtcNow -lt $Deadline) {
        $Server.Refresh()
        if ($Server.HasExited) { throw "VLA server exited; inspect $ServerErr" }
        if ((Test-Path -LiteralPath $ServerOut) -and (Select-String -LiteralPath $ServerOut -Pattern "SMOLVLA_SERVER_READY=" -Quiet)) { $Ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $Ready) { throw "VLA server readiness timeout; inspect $ServerOut" }
    Write-Output "VLA_DEMO_SERVER_READY PID=$($Server.Id)"
    $IsaacArgs = @("scripts\run_vla_benchmark.py", "--levels", $Level, "--episodes-per-level", 1, "--max-steps", $MaxSteps, "--condition-vla-on-visual-pose", "--condition-vla-on-controller-phase", "--vla-position-target-mode", "delta", "--pure-vla", "--require-vla", "--vla-port", $Port, "--vla-timeout", 60, "--execution-horizon", 1, "--action-repeat", 5, "--camera-ready-timeout-s", 180, "--camera-render-yield-s", 0.02, "--progress-every", 25, "--realtime", "--device", "cuda:0", "--rendering_mode", "performance", "--output-dir", ('"' + $OutputDir + '"'))
    if ($Headless) { $IsaacArgs += "--headless" }
    elseif ($KeepOpen -and -not $NoKeepOpen) { $IsaacArgs += "--keep-open" }
    # Hide only the helper console; Isaac's requested GUI remains visible.
    $Isaac = Start-Process -FilePath $IsaacPython -ArgumentList $IsaacArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $IsaacOut -RedirectStandardError $IsaacErr
    # Retain the native handle before exit; Windows PowerShell can otherwise
    # report a null ExitCode for an already-exited Start-Process child.
    $null = $Isaac.Handle
    Write-Output "VLA_DEMO_ISAAC_STARTED PID=$($Isaac.Id) OUTPUT=$OutputDir"
    $GpuSamples = New-Object 'System.Collections.Generic.List[object]'
    while (-not $Isaac.WaitForExit(1000)) {
        $MemoryLines = & nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $MemoryLines) {
            $Fields = @($MemoryLines)[0].Split(',')
            $GpuSamples.Add([pscustomobject]@{utc=[DateTime]::UtcNow.ToString("o");total_mib=[int]$Fields[0].Trim();used_mib=[int]$Fields[1].Trim();free_mib=[int]$Fields[2].Trim();utilization_percent=[int]$Fields[3].Trim()})
        }
    }
    $GpuSamples | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDir "gpu_memory_samples.json") -Encoding UTF8
    $NativeExitCode = $Isaac.ExitCode
    if ($null -eq $NativeExitCode) {
        $ReportPath = Join-Path $OutputDir "vla_benchmark_report.json"
        if (-not (Test-Path -LiteralPath $ReportPath)) { throw "Native exit code unavailable and benchmark report missing; inspect $IsaacOut and $IsaacErr" }
        $FinishedReport = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
        if ($FinishedReport.status -ne "VLA_BENCHMARK_OK" -or -not (Select-String -LiteralPath $IsaacOut -Pattern "VLA_BENCHMARK_SUMMARY=" -Quiet)) { throw "Native exit code unavailable and completion evidence invalid" }
        Write-Warning "Windows did not return a native exit code; benchmark report and completion marker verified."
    }
    elseif ($NativeExitCode -ne 0) { throw "Isaac demo failed with exit code $NativeExitCode; inspect $IsaacOut and $IsaacErr" }
    Write-Output "VLA_DEMO_COMPLETED OUTPUT=$OutputDir"
}
finally {
    # Stop only this launcher's child service, and only after its GUI exits.
    if ($null -ne $Server) {
        $Server.Refresh()
        if (-not $Server.HasExited) { Stop-Process -InputObject $Server -ErrorAction SilentlyContinue }
    }
}
