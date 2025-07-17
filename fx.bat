@echo off
REM FixWurx LLM Shell launcher for Windows systems

REM Get the directory where this script is located
SET SCRIPT_DIR=%~dp0

REM Change to the script directory
cd /d "%SCRIPT_DIR%"

REM Execute the Python script with all arguments passed through
python launchpad.py %*
