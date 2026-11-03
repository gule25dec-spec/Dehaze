@echo off
REM ==== MUSIQ (no-reference) + LPIPS (perceptual) for all models ====
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=%USERPROFILE%\anaconda3\python.exe
"%PY%" -m pip install pyiqa lpips
"%PY%" run_all.py perceptual
echo.
echo Done. See results\table_MUSIQ.csv and results\table_LPIPS.csv
pause
