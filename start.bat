@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist "XUpscaleNode.exe" (
    echo [XUpscaleNode] ERROR: XUpscaleNode.exe not found next to start.bat.
    echo Run build.bat first to build it from source.
    pause
    exit /b 1
)

if not exist "config.yaml" (
    echo [XUpscaleNode] config.yaml not found, copying config.example.yaml...
    copy /y "config.example.yaml" "config.yaml" >nul
)

start "" "XUpscaleNode.exe" -c config.yaml
