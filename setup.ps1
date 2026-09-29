# Clean reinstall + launch. Usage: powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"

# Always run from the repo directory, even if the script was launched elsewhere.
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "=== SEA OF STARS AGENTS LAB: CLEAN REINSTALL ==="
Write-Host ""

Write-Host "[1/7] Checking Python..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python was not found."
    Write-Host "Install Python 3.11 or 3.12, reopen PowerShell, then run this script again."
    exit 1
}
python --version

Write-Host ""
Write-Host "[2/7] Removing old virtual environment..."
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force ".venv"
}

Write-Host ""
Write-Host "[3/7] Creating fresh .venv..."
python -m venv .venv

$PY = ".\.venv\Scripts\python.exe"

Write-Host ""
Write-Host "[4/7] Upgrading pip..."
& $PY -m pip install --upgrade pip

Write-Host ""
Write-Host "[5/7] Installing lab packages..."
& $PY -m pip install -r requirements.txt

Write-Host ""
Write-Host "[6/7] Verifying environment..."
& $PY .\verify_setup.py

Write-Host ""
Write-Host "[7/7] Launching JupyterLab..."
Write-Host "Open notebooks/00_01_State_Action_Q.ipynb first."
Write-Host "Next time, just double-click launch.bat."
Write-Host ""
& $PY -m jupyter lab
