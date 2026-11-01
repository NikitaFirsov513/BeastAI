@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe waid.py serve --dataset sherbrooke --open
if errorlevel 1 pause
