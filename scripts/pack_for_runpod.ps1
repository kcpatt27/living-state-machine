# Pack repo for RunPod upload (no GitHub required).
# Excludes state, memory, checkpoints, venv.
#
# Usage:
#   .\scripts\pack_for_runpod.ps1
#   scp -P PORT lsm_runpod.tgz root@POD_IP:/workspace/
#   # on pod: cd /workspace && tar xzf lsm_runpod.tgz && cd living-state-machine

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Out = Join-Path $RepoRoot "lsm_runpod.tgz"

Push-Location $RepoRoot
try {
    if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
        throw "tar not found. Use GitHub clone on pod or Win11+ built-in tar."
    }
    if (Test-Path $Out) { Remove-Item $Out -Force }
    tar czf $Out `
        --exclude=.git `
        --exclude=.venv `
        --exclude=state `
        --exclude=memory_store `
        --exclude=runs `
        --exclude=data `
        --exclude=__pycache__ `
        --exclude=.pytest_cache `
        --exclude=*.pt `
        --exclude=*.sqlite `
        --exclude=lsm_runpod.tgz `
        .
    $mb = [math]::Round((Get-Item $Out).Length / 1MB, 1)
    Write-Host "Wrote $Out ($mb MB)"
} finally {
    Pop-Location
}
