@echo off
echo ===== FixWurx Testing Framework =====
echo Starting test run using OpenAI o3 model
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Running automated tests with auto-apply mode...
python run_fixwurx_tests.py

echo.
echo Test run completed. Results are in the test_results directory.
pause
