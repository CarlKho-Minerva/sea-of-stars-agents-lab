@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo No virtual environment found.
    echo Run setup.ps1 first:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m jupyter lab
