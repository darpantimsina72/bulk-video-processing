@echo off
title Translation ^& Syncing App
REM Double-click to launch the Translation & Syncing App (Windows).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please run setup_windows.bat first (one-time setup).
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "Translation_and_Syncing_App.py"
if errorlevel 1 (
    echo.
    echo The app exited with an error. Details may be in error_log.txt
    pause
)
