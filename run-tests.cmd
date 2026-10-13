@echo off
setlocal
cd /d "%~dp0"

echo =====================================================================
echo  Artist Identification Studio - Test Suite
echo =====================================================================
echo(

if not exist .venv\Scripts\python.exe goto :no_venv

echo [INFO] Running test suite via unittest discovery ...
echo(

.venv\Scripts\python.exe -m unittest discover tests -v
if %ERRORLEVEL% EQU 0 goto :test_pass

echo(
echo =====================================================================
echo [FAIL] One or more tests failed.
echo =====================================================================
exit /b 1

:test_pass
echo(
echo =====================================================================
echo [OK] All 23 tests passed successfully.
echo =====================================================================
exit /b 0

:no_venv
echo [ERROR] Virtual environment (.venv) not found.
echo Please run setup.cmd first to set up the environment.
exit /b 1
