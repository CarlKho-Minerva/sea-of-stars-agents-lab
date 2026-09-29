@echo off
if not exist .venv (
  echo No .venv found. Run setup_zenbook.ps1 first.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m jupyter lab
