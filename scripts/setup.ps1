<#
.SYNOPSIS
  One-shot bootstrap for the Living State Machine project on Windows.

.DESCRIPTION
  Installs userland dev tooling via Scoop (Python 3.11, uv) and verifies that
  system-wide tooling (Git, Ollama, Docker Desktop) is present. Git, Ollama,
  and Docker Desktop are intentionally installed via their official Windows
  installers because those integrate with the OS more cleanly than Scoop can.
  Pulls default models and starts Open WebUI once prerequisites are available.
  Safe to re-run.

.NOTES
  Non-admin PowerShell is fine for Scoop-managed tools.

.PARAMETER SkipDocker
  Skip Docker Desktop verification and skip the Open WebUI container.

.PARAMETER SkipModels
  Skip the ollama pull step (faster re-runs during dev).
#>

[CmdletBinding()]
param(
    [switch]$SkipDocker,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Test-HasCommand {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Scoop {
    if (-not (Test-HasCommand "scoop")) {
        Write-Host "Scoop not found." -ForegroundColor Red
        Write-Host "Install Scoop first in a non-elevated shell:"
        Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
        Write-Host "  iwr -useb get.scoop.sh | iex"
        throw "Scoop is required. Install it and re-run."
    }
}

function Ensure-Scoop-Bucket {
    param([string]$Bucket)
    $buckets = scoop bucket list 2>$null
    if (-not ($buckets -match ("^\s*" + [regex]::Escape($Bucket) + "\s*$"))) {
        Write-Host "Adding Scoop bucket '$Bucket' ..."
        scoop bucket add $Bucket
    }
}

function Scoop-Install {
    param([string]$Package, [string]$Friendly)
    Write-Host "Installing $Friendly ($Package) via Scoop ..."
    scoop install $Package
}

function Refresh-Path {
    $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

Write-Section "Living State Machine - environment bootstrap"

Ensure-Scoop
Ensure-Scoop-Bucket "main"
Ensure-Scoop-Bucket "versions"

# Git (system-wide, official installer expected)
if (-not (Test-HasCommand "git")) {
    Write-Host "Git not found on PATH." -ForegroundColor Red
    Write-Host "Install Git for Windows from https://git-scm.com/download/win"
    Write-Host "During install pick 'Git from the command line and also from 3rd-party software'."
    Write-Host "Then close and reopen the terminal and re-run this script."
    throw "Git is required."
}
Write-Host ("Git present: " + (git --version))

# Python 3.11 via Scoop
$py311 = $null
if (Test-HasCommand "py") {
    try { $py311 = (& py -3.11 -c "import sys; print(sys.version)") } catch { }
}
if (-not $py311) {
    Write-Warning "If 'python' opens the Microsoft Store, disable App Execution Aliases:"
    Write-Warning "  Settings -> Apps -> Advanced app settings -> App execution aliases"
    Write-Warning "  Turn OFF: python.exe and python3.exe"
    Scoop-Install "python311" "Python 3.11"
} else {
    Write-Host "Python 3.11 present: $py311"
}

# uv via Scoop
if (-not (Test-HasCommand "uv")) {
    Scoop-Install "uv" "uv"
} else {
    Write-Host ("uv present: " + (uv --version))
}

Refresh-Path

# Ollama (system-wide, official installer expected)
if (-not (Test-HasCommand "ollama")) {
    Write-Host "Ollama not found on PATH." -ForegroundColor Red
    Write-Host "Install Ollama from https://ollama.com/download/windows"
    Write-Host "Then close and reopen the terminal and re-run this script."
    throw "Ollama is required."
}
Write-Host ("Ollama present: " + (ollama --version))

# Docker Desktop (system-wide, official installer expected)
if (-not $SkipDocker) {
    if (-not (Test-HasCommand "docker")) {
        Write-Host "Docker not found on PATH." -ForegroundColor Red
        Write-Host "Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
        Write-Host "Launch it once so WSL2 integration completes, then re-run this script."
        throw "Docker Desktop is required (or re-run with -SkipDocker)."
    }
    Write-Host ("Docker present: " + (docker --version))
}

Write-Section "Python environment (uv sync)"

Push-Location (Join-Path $PSScriptRoot "..")
try {
    uv sync --extra dev
    Write-Host "uv sync completed."
    Write-Host "To add Phase 2 research deps later: uv sync --extra dev --extra research"
} finally {
    Pop-Location
}

Write-Section "Ollama models"

if (-not $SkipModels) {
    try {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue | Out-Null
    } catch { }
    Start-Sleep -Seconds 2

    $models = @(
        "qwen2.5:14b-instruct-q4_K_M",
        "qwen2.5:7b-instruct-q8_0",
        "llama3.1:8b-instruct-q4_K_M"
    )
    foreach ($m in $models) {
        Write-Host "Pulling $m ..."
        ollama pull $m
    }
} else {
    Write-Host "Skipping model pulls (-SkipModels set)."
}

Write-Section "Open WebUI"

if (-not $SkipDocker) {
    if (Test-HasCommand "docker") {
        $existing = docker ps -a --filter "name=open-webui" --format "{{.Names}}"
        if ($existing -eq "open-webui") {
            Write-Host "Open WebUI container exists; starting it."
            docker start open-webui | Out-Null
        } else {
            Write-Host "Launching Open WebUI on http://localhost:3000 ..."
            docker run -d --name open-webui `
                -p 3000:8080 `
                -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
                -e WEBUI_AUTH=False `
                --add-host=host.docker.internal:host-gateway `
                --restart unless-stopped `
                -v open-webui:/app/backend/data `
                ghcr.io/open-webui/open-webui:main | Out-Null
        }
        Write-Host "Open WebUI: http://localhost:3000"
    } else {
        Write-Warning "Docker not available. Skipping Open WebUI."
        Write-Warning "Alternative: 'pip install open-webui' and run 'open-webui serve'."
    }
}

Write-Section "Done"

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Start the LSM wrapper server:   uv run lsm serve"
Write-Host "  2. Open Open WebUI:                http://localhost:3000"
Write-Host "  3. Or talk to it from the CLI:     uv run lsm chat"
Write-Host "  4. Run tests:                      uv run pytest"
Write-Host "  5. Install pre-commit hooks:       uv run pre-commit install"
Write-Host ""
Write-Host "One-time Windows recommendations:"
Write-Host "  - Disable App Execution Aliases for python.exe and python3.exe (already done if you followed the setup guide)"
Write-Host "  - Optional (admin): Add Defender exclusion for %USERPROFILE%\scoop"
Write-Host "  - Optional (admin): Enable LongPaths for deep Python envs"
Write-Host ""
Write-Host "Model switching:"
Write-Host "  Default:   qwen2.5:14b-instruct-q4_K_M  (~11GB VRAM, best quality)"
Write-Host "  Fast mode: qwen2.5:7b-instruct-q8_0     (~7GB VRAM, fast + keeps headroom)"
Write-Host ""
Write-Host "Free the GPU:"
Write-Host "  uv run lsm serve --stop             # unload LSM model"
Write-Host "  ollama stop qwen2.5:14b-instruct-q4_K_M"
