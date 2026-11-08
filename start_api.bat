@echo off
cd /d "%~dp0"
set "PY=%~dp0surya_env\Scripts\python.exe"
if not exist "%PY%" (
  echo Environment not found: %PY%
  pause
  exit /b 1
)
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause