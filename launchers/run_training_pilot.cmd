@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
.venv\Scripts\python.exe run.py run_training_pilot
if errorlevel 1 goto failed
.venv\Scripts\python.exe run.py analyze_training_pilot
:failed
pause
