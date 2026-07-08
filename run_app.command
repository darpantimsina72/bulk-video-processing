#!/bin/bash
# Double-click to launch the Translation & Syncing App (macOS).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Virtual environment not found."
    echo "Please run setup_mac.command first (one-time setup)."
    osascript -e 'display alert "Setup needed" message "Run setup_mac.command first (one-time setup)."' 2>/dev/null
    read -p "Press Enter to close..."
    exit 1
fi
exec .venv/bin/python "Translation_and_Syncing_App.py"
