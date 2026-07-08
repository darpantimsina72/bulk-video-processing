@echo off
setlocal
title Translation ^& Syncing App - Setup
cd /d "%~dp0"

echo ============================================================
echo  Translation ^& Syncing App - one-time Windows setup
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM 1) Find a real Python (the Microsoft Store "python" stub opens
REM    the Store instead of running - "py -3" avoids it).
REM ---------------------------------------------------------------
set "PY_CMD="
py -3 --version >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD (
    python --version >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo Python was not found on this computer.
    echo.
    echo   1. A browser window will open at python.org - download Python 3.11+
    echo   2. IMPORTANT: tick "Add python.exe to PATH" in the installer
    echo   3. After installing, run this setup script again
    echo.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
)

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo Your Python is too old. Please install Python 3.11+ from python.org
    echo and run this script again.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('%PY_CMD% --version 2^>^&1') do echo Found Python %%v

REM ---------------------------------------------------------------
REM 2) Tkinter sanity check (missing on some minimal installs)
REM ---------------------------------------------------------------
%PY_CMD% -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo Tkinter is missing. Re-run the Python installer, choose
    echo "Modify", and enable "tcl/tk and IDLE".
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 3) Create the virtual environment
REM ---------------------------------------------------------------
echo.
echo Creating virtual environment...
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo Failed to create the virtual environment.
    echo If the error mentions "path too long", move this folder to a
    echo shorter location like C:\Apps\TranslationApp and retry.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 4) Install dependencies
REM ---------------------------------------------------------------
echo Installing dependencies (this can take a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency install failed. Check your internet connection
    echo and run this script again - it is safe to re-run.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 5) FFmpeg (needed for MP3 loading / audio export)
REM ---------------------------------------------------------------
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo FFmpeg not found - trying automatic install via winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo Automatic install failed. Install FFmpeg manually:
        echo   Open PowerShell and run:  winget install ffmpeg
        echo   Then restart the computer so PATH updates.
        echo The app will still start, but MP3 files will not load
        echo until FFmpeg is installed.
    ) else (
        echo FFmpeg installed. If the app cannot find it, restart
        echo the computer once so PATH updates.
    )
) else (
    echo FFmpeg found - OK.
)

echo.
echo ============================================================
echo  Setup complete!
echo  Start the app any time by double-clicking  run_windows.bat
echo ============================================================
pause
