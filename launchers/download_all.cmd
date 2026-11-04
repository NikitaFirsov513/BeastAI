@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set "BEASTAI_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "BEASTAI_PYTHON=.venv\Scripts\python.exe"
"%BEASTAI_PYTHON%" run.py waid download --all
echo.
pause
