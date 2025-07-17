#!/bin/bash
# FixWurx LLM Shell launcher for Unix-based systems (Linux/macOS)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Execute the Python script with all arguments passed through
python3 launchpad.py "$@"
