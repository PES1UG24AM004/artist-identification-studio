<#
.SYNOPSIS
    Launch Artist Identification Studio Streamlit Web Application in Windows PowerShell.
#>
param(
    [string]$Port = "8501",
    [string]$Address = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Artist Identification Studio - Interactive Application" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

$PythonVenv = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$StreamlitVenv = Join-Path $ScriptDir ".venv\Scripts\streamlit.exe"
$SvmModel = Join-Path $ScriptDir "models\svm_baseline.joblib"
$CnnModel = Join-Path $ScriptDir "models\cnn_best.pth"

if (-not (Test-Path $PythonVenv)) {
    Write-Host "[ERROR] Virtual environment (.venv) not found at: $PythonVenv" -ForegroundColor Red
    Write-Host "Please create and install dependencies into .venv first or run setup.cmd" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $SvmModel)) {
    Write-Host "[ERROR] Checkpoint $SvmModel not found." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $CnnModel)) {
    Write-Host "[ERROR] Checkpoint $CnnModel not found." -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Launching Artist Identification Studio on http://${Address}:${Port} ..." -ForegroundColor Green
Write-Host "[INFO] Press Ctrl+C in this terminal to stop the application." -ForegroundColor Gray
Write-Host ""

& $PythonVenv -m streamlit run src/app.py --server.address $Address --server.port $Port
