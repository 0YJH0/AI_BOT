param(
    [string]$EnvironmentName = "lerobot-vla"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Conda = "C:\Users\yjh\anaconda3\Scripts\conda.exe"

if (-not (Test-Path -LiteralPath $Conda)) {
    throw "Conda executable was not found at $Conda"
}

& $Conda create -n $EnvironmentName python=3.11 pip ffmpeg -y
& $Conda run -n $EnvironmentName python -m pip install --upgrade pip
& $Conda run -n $EnvironmentName python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $Conda run -n $EnvironmentName python -m pip install -r (Join-Path $ProjectRoot "requirements-vla.txt")
& $Conda run -n $EnvironmentName python -m pip install --no-deps -e $ProjectRoot

Write-Host "Created isolated environment: $EnvironmentName"
Write-Host "Run: conda activate $EnvironmentName"
