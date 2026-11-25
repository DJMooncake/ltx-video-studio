@echo off
rem Starts LTX Video Studio. The first start sets up its own Python environment (it tells you what it is doing).
cd /d "%~dp0"

where pyw >nul 2>nul
if %errorlevel%==0 (
    start "" pyw -3 "%~dp0ltx_app.py" %*
    exit /b 0
)
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%~dp0ltx_app.py" %*
    exit /b 0
)

echo Python was not found.
echo Please install Python 3.11 (64-bit) from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click this file again.
pause
