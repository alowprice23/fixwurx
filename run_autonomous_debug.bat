@echo off
echo ===== FixWurx Autonomous Debugging Mode =====
echo Starting autonomous bug detection and fixing...
echo.

echo Step 1: Fixing Basic Operations
python fixwurx.py --analyze calculator/operations/basic_operations.py --run-tests calculator.tests.test_basic_operations

echo.
echo Step 2: Fixing Advanced Operations
python fixwurx.py --analyze calculator/operations/advanced_operations.py --run-tests calculator.tests.test_advanced_operations

echo.
echo Step 3: Fixing Validation Utilities
python fixwurx.py --analyze calculator/utils/validation.py --run-tests calculator.tests.test_validation

echo.
echo Step 4: Fixing Memory Utilities
python fixwurx.py --analyze calculator/utils/memory.py --run-tests calculator.tests.test_memory

echo.
echo Step 5: Fixing CLI Interface
python fixwurx.py --analyze calculator/ui/cli.py --run-tests calculator.tests.test_cli

echo.
echo Step 6: Comprehensive Test
python -m unittest discover calculator.tests

echo.
echo Debugging process completed. Results are in the fixwurx_results directory.
pause
