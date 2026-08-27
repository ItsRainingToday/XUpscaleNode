@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [XUpscaleNode] ERROR: 'python' not found in PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [XUpscaleNode] Creating virtual environment...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
)

echo [XUpscaleNode] Installing build dependencies...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
".venv\Scripts\python.exe" -m pip install -q pyinstaller

echo [XUpscaleNode] Building XUpscaleNode.exe...
rem --onedir, not --onefile: a onefile windowed build's temp-extract-then-
rem re-exec startup throws off customtkinter's window-show timing on
rem Windows badly enough that the window stays invisible forever (confirmed
rem by testing) while the process itself runs fine - onedir starts the real
rem exe directly, no re-exec, no issue.
".venv\Scripts\python.exe" -m PyInstaller --onedir --name XUpscaleNode --windowed --collect-data customtkinter run_node.py
if errorlevel 1 (
    echo [XUpscaleNode] ERROR: build failed. See messages above.
    pause
    exit /b 1
)

robocopy "dist\XUpscaleNode" "." /e >nul
echo [XUpscaleNode] Done: XUpscaleNode.exe (+ _internal\ next to it)
pause
