# Reinstall torch with CUDA 12.8 wheels (required for RTX 50-series / Blackwell on Windows).
# PyPI's default torch for Windows is CPU-only (torch X.Y.Z+cpu).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..

Write-Host "Reinstalling torch from PyTorch cu128 index (Blackwell-safe)..."
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128

Write-Host ""
Write-Host "CUDA check:"
uv run python -c @"
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
    print('arch_list', torch.cuda.get_arch_list())
else:
    print('If False on RTX 5060 Ti: try nightly cu128 or set CUDA_VISIBLE_DEVICES=0')
"@
