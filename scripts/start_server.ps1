<#
.SYNOPSIS
  Start the LSM wrapper server, ensuring Ollama is running first.

.PARAMETER Model
  Model tag to load. Defaults to the value in .env / LSM_MODEL.

.PARAMETER Fast
  Start in fast mode (qwen3.5:4b).

.EXAMPLE
  .\scripts\start_server.ps1
  .\scripts\start_server.ps1 -Fast
  .\scripts\start_server.ps1 -Model qwen2.5:14b-instruct-q4_K_M
#>

[CmdletBinding()]
param(
    [string]$Model,
    [switch]$Fast
)

$ErrorActionPreference = "Stop"

function Has-Command($name) {
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

if (-not (Has-Command ollama)) {
    throw "Ollama is not installed. Run scripts\setup.ps1 first."
}

$ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Write-Host "Starting Ollama daemon ..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

Push-Location $PSScriptRoot\..
try {
    $extra = @()
    if ($Fast) { $extra += @("--model", "qwen3.5:4b") }
    if ($Model) { $extra += @("--model", $Model) }

    uv run lsm serve @extra
} finally {
    Pop-Location
}
