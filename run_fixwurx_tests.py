#!/usr/bin/env python
"""
FixWurx Test Runner
This script automates the testing process for FixWurx according to the test plan.
"""
import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
TEST_RESULTS_DIR = Path("test_results")
TEST_RESULTS_DIR.mkdir(exist_ok=True)

def setup():
    """Install required dependencies."""
    print("Setting up environment...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("Dependencies installed successfully.")

def run_command(command, input_text=None):
    """Run a command and return its output."""
    print(f"Running command: {' '.join(command)}")
    
    # If input_text is provided, pass it to the subprocess
    if input_text:
        result = subprocess.run(command, input=input_text.encode(), capture_output=True, text=True)
    else:
        result = subprocess.run(command, capture_output=True, text=True)
    
    print("Command output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    return result

def verify_test(test_path):
    """Run a test and check if it passes."""
    print(f"\nVerifying fix with test: {test_path}")
    # Convert path format from 'calculator/tests/test_x.py' to 'calculator.tests.test_x'
    module_path = test_path.replace('/', '.').replace('.py', '')
    result = run_command([sys.executable, "-m", "unittest", module_path])
    return result.returncode == 0

def record_test_result(test_id, file_path, function, description, command, test_path, success, start_time, end_time, notes=""):
    """Record the test result to a JSON file."""
    result = {
        "test_id": test_id,
        "file_path": file_path,
        "function": function,
        "description": description,
        "command": command,
        "test_path": test_path,
        "success": success,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "notes": notes
    }
    
    # Append to results file
    result_file = TEST_RESULTS_DIR / "test_results.json"
    
    # Initialize file if it doesn't exist
    if not result_file.exists():
        with open(result_file, 'w') as f:
            json.dump({"tests": []}, f)
    
    # Read existing results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Append new result
    results["tests"].append(result)
    
    # Write updated results
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test result recorded to {result_file}")

def run_basic_operations_tests():
    """Run tests for basic operations module."""
    print("\n===== BASIC OPERATIONS TESTS =====")
    
    # Test B1: Subtraction Bug
    test_id = "B1"
    file_path = "calculator/operations/basic_operations.py"
    function = "subtract"
    description = "Incorrect order of operands (b - a instead of a - b)"
    command = ["python", "fixwurx.py", "--analyze", file_path, "--focus", function]
    test_path = "calculator/tests/test_basic_operations.py::TestBasicOperations::test_subtract"
    
    print(f"\nRunning test {test_id}: {description}")
    start_time = datetime.now()
    # Auto-approve fixes with 'Y' input
    result = run_command(command, input_text="Y\n")
    success = verify_test(test_path)
    end_time = datetime.now()
    
    record_test_result(
        test_id, file_path, function, description, 
        " ".join(command), test_path, success, 
        start_time, end_time
    )
    
    # Test B2: Multiplication Bug
    test_id = "B2"
    file_path = "calculator/operations/basic_operations.py"
    function = "multiply"
    description = "Using addition instead of multiplication"
    command = ["python", "fixwurx.py", "--analyze", file_path, "--focus", function]
    test_path = "calculator/tests/test_basic_operations.py::TestBasicOperations::test_multiply"
    
    print(f"\nRunning test {test_id}: {description}")
    start_time = datetime.now()
    result = run_command(command, input_text="Y\n")
    success = verify_test(test_path)
    end_time = datetime.now()
    
    record_test_result(
        test_id, file_path, function, description, 
        " ".join(command), test_path, success, 
        start_time, end_time
    )
    
    # Test B3: Division Zero Check Bug
    test_id = "B3"
    file_path = "calculator/operations/basic_operations.py"
    function = "divide"
    description = "No zero division check"
    command = ["python", "fixwurx.py", "--analyze", file_path, "--focus", function]
    test_path = "calculator/tests/test_basic_operations.py::TestBasicOperations::test_divide"
    
    print(f"\nRunning test {test_id}: {description}")
    start_time = datetime.now()
    result = run_command(command, input_text="Y\n")
    success = verify_test(test_path)
    end_time = datetime.now()
    
    record_test_result(
        test_id, file_path, function, description, 
        " ".join(command), test_path, success, 
        start_time, end_time
    )
    
    # Test B4: Power Implementation Bug
    test_id = "B4"
    file_path = "calculator/operations/basic_operations.py"
    function = "power"
    description = "Incorrect implementation (using a * b instead of a ** b)"
    command = ["python", "fixwurx.py", "--analyze", file_path, "--focus", function]
    test_path = "calculator/tests/test_basic_operations.py::TestBasicOperations::test_power"
    
    print(f"\nRunning test {test_id}: {description}")
    start_time = datetime.now()
    result = run_command(command, input_text="Y\n")
    success = verify_test(test_path)
    end_time = datetime.now()
    
    record_test_result(
        test_id, file_path, function, description, 
        " ".join(command), test_path, success, 
        start_time, end_time
    )

def run_advanced_operations_tests():
    """Run tests for advanced operations module."""
    print("\n===== ADVANCED OPERATIONS TESTS =====")
    
    # Test A1 & A6: Random Import and Usage Bugs
    test_id = "A1_A6"
    file_path = "calculator/operations/advanced_operations.py"
    function = "random_operation"
    description = "Missing random import and using undefined 'random' module"
    command = ["python", "fixwurx.py", "--analyze", file_path, "--focus", function]
    test_path = "calculator/tests/test_advanced_operations.py::TestAdvancedOperations::test_random_operation"
    
    print(f"\nRunning test {test_id}: {description}")
    start_time = datetime.now()
    result = run_command(command, input_text="Y\n")
    success = verify_test(test_path)
    end_time = datetime.now()
    
    record_test_result(
        test_id, file_path, function, description, 
        " ".join(command), test_path, success, 
        start_time, end_time
    )
    
    # Test A2: Square Root Negative Check Bug
    test_id = "A2"
    file_path = "calculator/operations/advanced_operations.py"
    function = "square_root"
    description = "No negative number check"
    command = ["python", "fixwurx.py", "--analyze", file_path, "--focus", function]
    test_path = "calculator/tests/test_advanced_operations.py::TestAdvancedOperations::test_square_root"
    
    print(f"\nRunning test {test_id}: {description}")
    start_time = datetime.now()
    try:
        result = run_command(command, input_text="Y\n")
        success = verify_test(test_path)
    except Exception as e:
        print(f"Error during test: {str(e)}")
        success = False
    end_time = datetime.now()
    
    record_test_result(
        test_id, file_path, function, description, 
        " ".join(command), test_path, success, 
        start_time, end_time
    )

def run_integration_tests():
    """Run integration tests."""
    print("\n===== INTEGRATION TESTS =====")
    
    # Integration Test 1: CLI and Operations Interdependencies
    test_id = "I1"
    file_paths = ["calculator/ui/cli.py", "calculator/operations/advanced_operations.py"]
    description = "CLI and Operations Interdependencies"
    command = ["python", "fixwurx.py", "--analyze"] + file_paths
    test_path = "calculator.tests.test_cli calculator.tests.test_advanced_operations"
    
    print(f"\nRunning integration test {test_id}: {description}")
    start_time = datetime.now()
    result = run_command(command, input_text="Y\n")
    
    # Run both test cases
    test_paths = test_path.split()
    success = all(verify_test(tp) for tp in test_paths)
    end_time = datetime.now()
    
    record_test_result(
        test_id, str(file_paths), "N/A", description, 
        " ".join(command), test_path, success, 
        start_time, end_time
    )

def generate_summary():
    """Generate a summary of test results."""
    result_file = TEST_RESULTS_DIR / "test_results.json"
    if not result_file.exists():
        print("No test results found.")
        return
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    total_tests = len(results["tests"])
    successful_tests = sum(1 for test in results["tests"] if test["success"])
    
    summary = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": total_tests - successful_tests,
        "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save summary
    with open(TEST_RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n===== TEST SUMMARY =====")
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {total_tests - successful_tests}")
    print(f"Success rate: {summary['success_rate']:.2f}%")

def main():
    """Main entry point for the test runner."""
    try:
        setup()
        
        # Run tests according to test plan
        run_basic_operations_tests()
        run_advanced_operations_tests()
        run_integration_tests()
        
        # Generate summary
        generate_summary()
        
        print("\nTest run completed successfully.")
        
    except Exception as e:
        print(f"Error during test run: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
