#!/bin/bash
# One-time setup for the Translation & Syncing App (macOS).
# Double-click this file in Finder (or run it in Terminal).
cd "$(dirname "$0")"

echo "============================================================"
echo " Translation & Syncing App — one-time macOS setup"
echo "============================================================"
echo

# ---------------------------------------------------------------
# 1) Find Python 3.10+ (prefer newer Homebrew versions)
# ---------------------------------------------------------------
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PY="$cand"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "Python 3.10+ was not found."
    echo
    if command -v brew >/dev/null 2>&1; then
        echo "Installing Python via Homebrew..."
        brew install python@3.13 || { echo "Homebrew install failed."; read -p "Press Enter to close..."; exit 1; }
        PY=python3.13
    else
        echo "Install it one of these ways, then re-run this script:"
        echo "  • Homebrew:  brew install python@3.13"
        echo "  • Or download from https://www.python.org/downloads/"
        open "https://www.python.org/downloads/" 2>/dev/null
        read -p "Press Enter to close..."
        exit 1
    fi
fi
echo "Found $($PY --version)"

# ---------------------------------------------------------------
# 2) Tkinter check (Homebrew Python sometimes needs python-tk)
# ---------------------------------------------------------------
if ! "$PY" -c 'import tkinter' 2>/dev/null; then
    echo "Tkinter is missing — installing python-tk via Homebrew..."
    if command -v brew >/dev/null 2>&1; then
        brew install python-tk@"$("$PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
            || brew install python-tk || true
    fi
    if ! "$PY" -c 'import tkinter' 2>/dev/null; then
        echo "Could not enable Tkinter. Install Python from python.org instead."
        read -p "Press Enter to close..."
        exit 1
    fi
fi

# ---------------------------------------------------------------
# 3) Virtual environment + dependencies
# ---------------------------------------------------------------
echo
echo "Creating virtual environment..."
"$PY" -m venv .venv || { echo "venv creation failed."; read -p "Press Enter to close..."; exit 1; }

echo "Installing dependencies (this can take a few minutes)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt || {
    echo
    echo "Dependency install failed. Check your internet connection and"
    echo "run this script again — it is safe to re-run."
    read -p "Press Enter to close..."
    exit 1
}

# ---------------------------------------------------------------
# 4) FFmpeg (needed for MP3 loading / audio export)
# ---------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/ffmpeg ] && [ ! -x /usr/local/bin/ffmpeg ]; then
    if command -v brew >/dev/null 2>&1; then
        echo "Installing FFmpeg via Homebrew..."
        brew install ffmpeg || echo "FFmpeg install failed — run 'brew install ffmpeg' manually."
    else
        echo "FFmpeg not found. Install Homebrew (https://brew.sh) and run:"
        echo "    brew install ffmpeg"
        echo "The app will still start, but MP3 files won't load without it."
    fi
else
    echo "FFmpeg found — OK."
fi

echo
echo "============================================================"
echo " Setup complete!"
echo " Start the app any time by double-clicking  run_app.command"
echo "============================================================"
read -p "Press Enter to close..."
