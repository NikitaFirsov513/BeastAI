@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set "BEASTAI_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "BEASTAI_PYTHON=.venv\Scripts\python.exe"
"%BEASTAI_PYTHON%" waid.py download --limit 30
if errorlevel 1 goto failed
"%BEASTAI_PYTHON%" waid.py serve --open
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Could not start. Copy the error above into the chat.
pause
