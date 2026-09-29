$ErrorActionPreference = "Stop"

Write-Host "=== Elder Mist RL Lab setup ==="
python --version

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Start the lab with:"
Write-Host "  .\start_lab.bat"
