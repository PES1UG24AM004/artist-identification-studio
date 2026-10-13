@echo off
setlocal
cd /d "%~dp0"

echo =====================================================================
echo  Artist Identification Studio - Interactive Application
echo =====================================================================
echo(

if not exist .venv\Scripts\python.exe goto :no_venv
if not exist models\svm_baseline.joblib goto :no_svm
if not exist models\cnn_best.pth goto :no_cnn

echo [INFO] Launching Artist Identification Studio on http://127.0.0.1:8501 ...
echo [INFO] Press Ctrl+C in this terminal to stop the application.
echo(

.venv\Scripts\python.exe -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501
exit /b %ERRORLEVEL%

:no_venv
echo [ERROR] Virtual environment (.venv) not found.
echo Please run setup.cmd first to set up the environment.
exit /b 1

:no_svm
echo [ERROR] Checkpoint models\svm_baseline.joblib not found.
echo Please ensure model files are present before starting demo.
exit /b 1

:no_cnn
echo [ERROR] Checkpoint models\cnn_best.pth not found.
echo Please ensure model files are present before starting demo.
exit /b 1
