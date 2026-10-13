@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo =====================================================================
echo  Artist Identification Studio - Environment Setup
echo =====================================================================
echo(

:: Step 1: Detect Python interpreter (prefer Python 3.12 via py launcher)
set "PYTHON_CMD="

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3.12"
    echo [INFO] Found Python 3.12 via Python Launcher.
    goto :python_found
)

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
    echo [INFO] Found default Python interpreter.
    goto :python_found
)

echo [ERROR] Python was not found in PATH.
echo Please install Python 3.10+ (recommended 3.12) and ensure it is added to your PATH.
exit /b 1

:python_found
:: Step 2: Create virtual environment if it does not exist
if exist .venv\Scripts\python.exe goto :venv_exists

echo [INFO] Creating Python virtual environment in .venv ...
%PYTHON_CMD% -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)
echo [INFO] Virtual environment created successfully.
goto :install_packages

:venv_exists
echo [INFO] Existing virtual environment found in .venv.

:install_packages
set "VENV_PYTHON=.venv\Scripts\python.exe"

:: Step 3: Upgrade pip
echo [INFO] Upgrading pip ...
%VENV_PYTHON% -m pip install --upgrade pip --quiet

:: Step 4: Install PyTorch CPU wheels (satisfies version if already present)
echo [INFO] Verifying / Installing PyTorch CPU runtime ...
%VENV_PYTHON% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch CPU runtime.
    exit /b 1
)

:: Step 5: Install project dependencies from requirements.txt (verifies pinned versions)
echo [INFO] Verifying / Installing dependencies from requirements.txt ...
%VENV_PYTHON% -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install requirements.
    exit /b 1
)

:: Step 6: Validate imports
echo [INFO] Validating essential libraries ...
%VENV_PYTHON% -c "import torch, torchvision, sklearn, streamlit, PIL, joblib; print('[OK] All essential imports validated successfully.')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency validation failed.
    exit /b 1
)

:check_models
:: Step 7: Verify model checkpoints
echo [INFO] Checking pre-trained model checkpoints ...
if not exist models\svm_baseline.joblib (
    echo [WARNING] models\svm_baseline.joblib is missing.
)
if not exist models\cnn_best.pth (
    echo [WARNING] models\cnn_best.pth is missing.
)
if exist models\svm_baseline.joblib if exist models\cnn_best.pth (
    echo [OK] Both model checkpoints are present.
)

echo(
echo =====================================================================
echo  Setup Completed Successfully!
echo  You can now run:
echo    run-demo.cmd    - to launch the interactive application
echo    run-tests.cmd   - to run the automated test suite
echo =====================================================================
exit /b 0
