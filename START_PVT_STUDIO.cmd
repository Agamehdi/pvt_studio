@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title PVT Studio Launcher

if not exist "app.py" (
    echo.
    echo ERROR: app.py was not found next to this launcher.
    echo First right-click PVT_Studio_Streamlit.zip, choose Extract All,
    echo then open the extracted pvt_studio folder and run this file again.
    echo.
    pause
    exit /b 1
)

where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: Windows PowerShell is required but was not found.
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_pvt_studio.ps1"
set "PVT_EXIT_CODE=%ERRORLEVEL%"
if not "%PVT_EXIT_CODE%"=="0" (
    echo.
    echo PVT Studio could not start. Review the message above and the logs folder.
    pause
)
exit /b %PVT_EXIT_CODE%
