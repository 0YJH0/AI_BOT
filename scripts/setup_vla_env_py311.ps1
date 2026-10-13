param(
    [string]$EnvironmentName = "lerobot-vla"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Conda = "C:\Users\yjh\anaconda3\Scripts\conda.exe"
$EnvironmentRoot = "C:\Users\yjh\anaconda3\envs\$EnvironmentName"
$Python = Join-Path $EnvironmentRoot "python.exe"

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Description)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    Invoke-Checked { & $Conda create -n $EnvironmentName python=3.11 pip ffmpeg -y } "Conda create"
}

& $Python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
if ($LASTEXITCODE -ne 0) {
    Invoke-Checked {
        & $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    } "CUDA PyTorch install"
}

$Dependencies = @(
    "datasets>=4.0,<5",
    "diffusers>=0.27.2,<0.36",
    "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36",
    "accelerate>=1.10,<2",
    "cmake>=3.29,<4.2",
    "einops>=0.8,<0.9",
    "opencv-python-headless>=4.9,<4.13",
    "av>=15,<16",
    "jsonlines>=4,<5",
    "packaging>=24.2,<26",
    "pynput>=1.7.7,<1.9",
    "pyserial>=3.5,<4",
    "wandb>=0.24,<0.25",
    "draccus==0.10.0",
    "gymnasium>=1.1.1,<2",
    "rerun-sdk>=0.24,<0.27",
    "deepdiff>=7.0.1,<9",
    "imageio[ffmpeg]>=2.34,<3",
    "termcolor>=2.4,<4",
    "num2words>=0.5.14,<0.6",
    "safetensors>=0.4.3,<1",
    "peft>=0.17,<0.19",
    "transformers>=4.53,<5"
)
Invoke-Checked { & $Python -m pip install @Dependencies } "LeRobot dependency install"
Invoke-Checked { & $Python -m pip install --no-deps "lerobot[smolvla]==0.4.4" } "LeRobot install"
Invoke-Checked { & $Python -m pip install --no-deps -e $ProjectRoot } "Project editable install"
Invoke-Checked {
    & $Python -c "import torch, lerobot, peft; from lerobot.policies.smolvla import SmolVLAPolicy; print('VLA_ENV_OK', torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
} "VLA import check"

Write-Host "Created and validated isolated environment: $EnvironmentName"
