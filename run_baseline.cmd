@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe baseline.py --dataset sherbrooke --imgsz 1024 --no-latest
if errorlevel 1 goto failed
echo.
echo Pretrained baseline saved separately. Viewer still shows the trained model.
pause
exit /b 0
:failed
echo Copy the error above into the chat.
pause
