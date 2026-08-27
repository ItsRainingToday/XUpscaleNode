@echo off
cd /d "%~dp0"
if exist start_debug.log del start_debug.log

:loop
echo [run_node] ================ launching node %date% %time% ================ >> start_debug.log
"XUpscaleNode.exe" -c config.yaml --headless >> start_debug.log 2>&1
echo [run_node] node exited with code %errorlevel% at %date% %time% - restarting in 3s >> start_debug.log
timeout /t 3 /nobreak > nul
goto loop
