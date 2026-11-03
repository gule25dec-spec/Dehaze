@echo off
REM ==== Adds CoA (CVPR 2025) and BiLaLoRA (CVPR 2026): download weights, run, evaluate ====
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=%USERPROFILE%\anaconda3\python.exe
"%PY%" -m pip install -q -r requirements.txt
"%PY%" run_all.py download --models CoA BiLaLoRA
"%PY%" run_all.py infer --models CoA BiLaLoRA
"%PY%" run_all.py evaluate
echo.
echo Done. Results are in the "results" folder.
pause
