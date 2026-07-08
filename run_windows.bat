@echo off
title Translation ^& Syncing App
REM Double-click to launch the Translation & Syncing App (Windows).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please run setup_windows.bat first ^(one-time setup^).
    echo.
    echo NOTE: if you are running this from inside the downloaded ZIP,
    echo extract the ZIP to a folder first ^(right-click ^> Extract All^).
    pause
    exit /b 1
)

echo Starting the app... (this window stays open while the app runs)
".venv\Scripts\python.exe" "Translation_and_Syncing_App.py" > launch_log.txt 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  The app failed to start. Error details:
    echo ============================================================
    type launch_log.txt
    echo ============================================================
    echo  This is also saved in launch_log.txt - send that file
    echo  when reporting the problem.
    echo  Most fixes: run setup_windows.bat again.
    echo ============================================================
    pause
)
