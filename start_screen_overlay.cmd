@echo off
cd /d "%~dp0"
start "BeastAI" "%~dp0data\dronemegadetector\env\Scripts\pythonw.exe" "%~dp0run.py" screen_overlay
