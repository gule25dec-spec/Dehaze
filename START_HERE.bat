@echo off
REM ==== Assignment 2 dehazing: 4 models x 3 datasets (double-click to run) ====
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=%USERPROFILE%\anaconda3\python.exe
"%PY%" --version || (echo Python not found. Install Anaconda or Python 3.10+ & pause & exit /b 1)

echo.
echo [1/3] Installing PyTorch (only if missing) ...
"%PY%" -c "import torch" 2>nul
if errorlevel 1 (
  where nvidia-smi >nul 2>nul
  if not errorlevel 1 (
    echo NVIDIA GPU found - installing CUDA build of PyTorch
    "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ) else (
    echo No NVIDIA GPU - installing CPU build of PyTorch
    "%PY%" -m pip install torch torchvision
  )
)
echo.
echo [2/3] Installing other packages ...
"%PY%" -m pip install -r requirements.txt

echo.
echo [3/3] Download, organize, run 4 models, evaluate ...
"%PY%" run_all.py all
echo.
echo Done. Results are in the "results" folder (summary.md, table_PSNR.csv, figures\).
pause
