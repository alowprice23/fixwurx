# FixWurx LLM Shell launcher for PowerShell
# This script is designed to work in PowerShell environments

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to the script directory
Set-Location $scriptDir

# Execute the Python script with all arguments passed through
python launchpad.py $args
