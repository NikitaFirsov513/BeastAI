@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe run_training_pilot.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe analyze_training_pilot.py
:failed
pause
