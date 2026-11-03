@echo off
REM ==== Adds ConvIR (TPAMI 2024): download weights, run, evaluate (PSNR/SSIM/MAE + MUSIQ/LPIPS) ====
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=%USERPROFILE%\anaconda3\python.exe
"%PY%" run_all.py download --models ConvIR
"%PY%" run_all.py infer --models ConvIR
"%PY%" run_all.py evaluate
"%PY%" run_all.py perceptual
echo.
echo Done. Results are in the "results" folder.
pause
