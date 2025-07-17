@echo off
REM Script to verify LLM integration across all agents
REM This script runs the configuration and verification scripts in sequence

echo ==== VERIFYING LLM INTEGRATION ACROSS ALL AGENTS ====
echo Step 1: Configuring OpenAI API integration
python configure_openai_integration.py

echo.
echo Step 2: Verifying LLM integration for all agents
python verify_llm_integration.py

echo.
echo LLM integration verification complete!
pause
