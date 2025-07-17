#!/usr/bin/env python3
"""
Test Script for Verification Module

This script tests the functionality of the verification module by running various verification
operations on test files.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure the verification module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from verification import (
        TestResult, CoverageResult, TestExecutor, SmokeTestRunner,
        CanaryTestRunner, RegressionDetector, Verifier,
        create_verifier, verify_patch, verify_coverage, run_canary_tests
    )
except ImportError:
    print("Error: Could not import verification module")
    sys.exit(1)

def create_test_patch_file():
    """Create a test patch file for testing."""
    patch_content = """diff --git a/test_file.py b/test_file.py
index 1234567..abcdefg 100644
--- a/test_file.py
+++ b/test_file.py
@@ -1,5 +1,5 @@
 def add(a, b):
-    return a + b
+    return a + b  # Fixed addition function
 
 def subtract(a, b):
-    return a - b
+    return a - b  # Fixed subtraction function
"""
    
    # Create temporary file
    fd, path = tempfile.mkstemp(suffix=".patch")
    os.close(fd)
    
    # Write patch content
    with open(path, 'w') as f:
        f.write(patch_content)
    
    return path

def test_test_result():
    """Test the TestResult class."""
    print("\n=== Testing TestResult ===")
    
    # Create a test result
    result = TestResult(True, "Test passed", 0, 1.5)
    
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Exit code: {result.exit_code}")
    print(f"Duration: {result.duration}")
    
    # Test to_dict
    result_dict = result.to_dict()
    print(f"Dictionary representation: {list(result_dict.keys())}")
    
    # Test parsing output
    result = TestResult(False, "Test failed\nERROR: Something went wrong\nFAIL: Assertion failed", 1, 2.0)
    
    print(f"Errors: {result.errors}")
    print(f"Failures: {result.failures}")
    
    return True

def test_coverage_result():
    """Test the CoverageResult class."""
    print("\n=== Testing CoverageResult ===")
    
    # Create test coverage data
    coverage_data = {
        "total_coverage": 85.5,
        "file_coverage": {
            "file1.py": 90.0,
            "file2.py": 75.0
        },
        "uncovered_lines": {
            "file1.py": [10, 15, 20],
            "file2.py": [5, 10, 15, 20, 25]
        }
    }
    
    # Create a coverage result
    result = CoverageResult(coverage_data)
    
    print(f"Total coverage: {result.total_coverage}")
    print(f"File coverage: {len(result.file_coverage)} files")
    print(f"Uncovered lines: {len(result.uncovered_lines)} files")
    
    # Test to_dict
    result_dict = result.to_dict()
    print(f"Dictionary representation: {list(result_dict.keys())}")
    
    return True

def test_test_executor():
    """Test the TestExecutor class (mock execution)."""
    print("\n=== Testing TestExecutor (Mock) ===")
    
    # Create a test executor with mock execution
    class MockTestExecutor(TestExecutor):
        def run_test(self, test_path, env_vars=None):
            print(f"Mock running test: {test_path}")
            success = "fail" not in test_path
            output = "Test passed" if success else "Test failed"
            return TestResult(success, output, 0 if success else 1, 1.0)
        
        def measure_coverage(self, module_path=None):
            print(f"Mock measuring coverage for: {module_path or 'all'}")
            return CoverageResult({
                "total_coverage": 85.5,
                "file_coverage": {"file1.py": 90.0, "file2.py": 75.0},
                "uncovered_lines": {"file2.py": [5, 10, 15, 20, 25]}
            })
    
    # Create a mock executor
    executor = MockTestExecutor(".")
    
    # Test running a test
    print("\nRunning passing test:")
    result = executor.run_test("test_pass.py")
    print(f"Success: {result.success}")
    
    print("\nRunning failing test:")
    result = executor.run_test("test_fail.py")
    print(f"Success: {result.success}")
    
    # Test measuring coverage
    print("\nMeasuring coverage:")
    result = executor.measure_coverage()
    print(f"Total coverage: {result.total_coverage}")
    
    return True

def test_smoke_test_runner():
    """Test the SmokeTestRunner class (mock execution)."""
    print("\n=== Testing SmokeTestRunner (Mock) ===")
    
    # Create a smoke test runner with mock execution
    class MockSmokeTestRunner(SmokeTestRunner):
        def run_smoke_test(self, test_command, expected_output=None, expected_exit_code=0):
            print(f"Mock running smoke test: {test_command}")
            success = "fail" not in test_command
            output = "Test passed" if success else "Test failed"
            return TestResult(success, output, 0 if success else 1, 1.0)
    
    # Create a mock runner
    runner = MockSmokeTestRunner(".")
    
    # Add some smoke tests
    runner.add_smoke_test("python test_pass.py", "Test passed")
    runner.add_smoke_test("python test_fail.py", "Test passed")
    
    # Test running a smoke test
    print("\nRunning passing smoke test:")
    result = runner.run_smoke_test("python test_pass.py")
    print(f"Success: {result.success}")
    
    print("\nRunning failing smoke test:")
    result = runner.run_smoke_test("python test_fail.py")
    print(f"Success: {result.success}")
    
    # Test running all smoke tests
    print("\nRunning all smoke tests:")
    results = runner.run_all_smoke_tests()
    print(f"Results: {len(results)} tests")
    
    return True

def test_canary_test_runner():
    """Test the CanaryTestRunner class (mock execution)."""
    print("\n=== Testing CanaryTestRunner (Mock) ===")
    
    # Create a canary test runner
    runner = CanaryTestRunner(".")
    
    # Add some canary environments
    runner.add_canary_environment("docker", "python-3.9", "python:3.9")
    runner.add_canary_environment("virtualenv", "py39")
    
    print(f"Canary environments: {len(runner.canary_environments)}")
    
    # Test running a canary test
    print("\nRunning docker canary test:")
    result = runner._run_docker_canary("python-3.9", "python:3.9", "python test_pass.py")
    print(f"Success: {result.success}")
    
    print("\nRunning virtualenv canary test:")
    result = runner._run_virtualenv_canary("py39", "python test_pass.py")
    print(f"Success: {result.success}")
    
    # Test running all canary tests
    print("\nRunning all canary tests:")
    results = runner.run_all_canary_tests("python test_pass.py")
    print(f"Results: {len(results)} environments")
    
    return True

def test_regression_detector():
    """Test the RegressionDetector class."""
    print("\n=== Testing RegressionDetector ===")
    
    # Create a temporary directory for test history
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a regression detector
        detector = RegressionDetector(temp_dir, {"history_file": "test_history.json"})
        
        # Add some test results
        print("\nAdding test results:")
        detector.add_test_result("test1", TestResult(True, "Test passed", 0, 1.0))
        detector.add_test_result("test2", TestResult(True, "Test passed", 0, 1.0))
        
        print(f"Test history: {len(detector.test_history)} tests")
        
        # Test detecting regression (no regression)
        print("\nDetecting regression (no regression):")
        result = detector.detect_regression("test1", TestResult(True, "Test passed", 0, 1.0))
        print(f"Regression detected: {result['regression_detected']}")
        
        # Test detecting regression (regression)
        print("\nDetecting regression (with regression):")
        result = detector.detect_regression("test2", TestResult(False, "Test failed\nERROR: Something went wrong", 1, 1.0))
        print(f"Regression detected: {result['regression_detected']}")
        if result["regression_detected"]:
            print(f"Regression type: {result['regression_type']}")
    
    return True

def test_verifier():
    """Test the Verifier class."""
    print("\n=== Testing Verifier ===")
    
    # Create a temporary directory for verifier
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test patch file
        patch_file = create_test_patch_file()
        rel_patch_file = os.path.basename(patch_file)
        
        try:
            # Copy patch file to temp directory
            with open(patch_file, 'r') as f:
                patch_content = f.read()
            
            with open(os.path.join(temp_dir, rel_patch_file), 'w') as f:
                f.write(patch_content)
            
            # Create a verifier with mock components
            verifier = create_verifier(temp_dir)
            
            # Override extract_affected_files to return test files
            verifier._extract_affected_files = lambda patch: ["test_file.py"]
            
            # Override find_tests_for_files to return test paths
            verifier._find_tests_for_files = lambda files: ["test_test_file.py"]
            
            # Mock test executor
            verifier.test_executor.run_test = lambda path, env_vars=None: TestResult(True, "Test passed", 0, 1.0)
            verifier.test_executor.measure_coverage = lambda module_path=None: CoverageResult({
                "total_coverage": 85.5,
                "file_coverage": {"test_file.py": 90.0},
                "uncovered_lines": {}
            })
            
            # Mock smoke test runner
            verifier.smoke_test_runner.run_all_smoke_tests = lambda: {"python test_file.py": TestResult(True, "Test passed", 0, 1.0)}
            
            # Test verify_patch
            print("\nVerifying patch:")
            result = verifier.verify_patch(rel_patch_file)
            print(f"Success: {result['success']}")
            print(f"Affected files: {result['affected_files']}")
            print(f"Test results: {len(result['test_results'])} tests")
            print(f"Coverage: {result['coverage']['total_coverage']}%")
            
            # Test verify_coverage
            print("\nVerifying coverage:")
            result = verifier.verify_coverage("test_file.py", 80.0)
            print(f"Success: {result['success']}")
            print(f"Total coverage: {result['total_coverage']}%")
            
            # Test run_canary_tests
            print("\nRunning canary tests:")
            verifier.canary_test_runner.run_all_canary_tests = lambda cmd: {
                "python-3.9": TestResult(True, "Test passed", 0, 1.0),
                "py39": TestResult(True, "Test passed", 0, 1.0)
            }
            
            result = verifier.run_canary_tests("python test_file.py")
            print(f"Success: {result['success']}")
            print(f"Results: {len(result['results'])} environments")
        finally:
            # Clean up
            if os.path.exists(patch_file):
                os.unlink(patch_file)
    
    return True

def test_api_functions():
    """Test the API functions."""
    print("\n=== Testing API Functions ===")
    
    # Create a temporary directory for API tests
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock verifier creation
        original_create_verifier = create_verifier
        
        class MockVerifier:
            def __init__(self, project_root, config=None):
                self.project_root = project_root
                self.config = config or {}
            
            def verify_patch(self, patch_file, test_paths=None):
                print(f"Mock verifying patch: {patch_file}")
                return {
                    "success": True,
                    "patch_file": patch_file,
                    "affected_files": ["test_file.py"],
                    "test_results": {"test_test_file.py": {"success": True}},
                    "coverage": {"total_coverage": 85.5},
                    "smoke_test_results": {},
                    "regressions": {}
                }
            
            def verify_coverage(self, module_path=None, min_coverage=80.0):
                print(f"Mock verifying coverage: {module_path or 'all'}")
                return {
                    "success": True,
                    "module_path": module_path,
                    "min_coverage": min_coverage,
                    "total_coverage": 85.5,
                    "file_coverage": {"test_file.py": 90.0},
                    "low_coverage_files": {},
                    "uncovered_lines": {}
                }
            
            def run_canary_tests(self, test_command):
                print(f"Mock running canary tests: {test_command}")
                return {
                    "success": True,
                    "test_command": test_command,
                    "results": {
                        "python-3.9": {"success": True},
                        "py39": {"success": True}
                    }
                }
        
        try:
            # Override create_verifier
            globals()["create_verifier"] = lambda project_root, config=None: MockVerifier(project_root, config)
            
            # Test verify_patch
            print("\nTesting verify_patch API:")
            result = verify_patch(temp_dir, "test.patch")
            print(f"Success: {result['success']}")
            
            # Test verify_coverage
            print("\nTesting verify_coverage API:")
            result = verify_coverage(temp_dir, "test_file.py", 80.0)
            print(f"Success: {result['success']}")
            
            # Test run_canary_tests
            print("\nTesting run_canary_tests API:")
            result = run_canary_tests(temp_dir, "python test_file.py")
            print(f"Success: {result['success']}")
        finally:
            # Restore original create_verifier
            globals()["create_verifier"] = original_create_verifier
    
    return True

def main():
    """Main function."""
    print("=== Verification Test Suite ===")
    
    # Run tests
    tests = [
        ("TestResult", test_test_result),
        ("CoverageResult", test_coverage_result),
        ("TestExecutor", test_test_executor),
        ("SmokeTestRunner", test_smoke_test_runner),
        ("CanaryTestRunner", test_canary_test_runner),
        ("RegressionDetector", test_regression_detector),
        ("Verifier", test_verifier),
        ("API Functions", test_api_functions)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\nRunning test: {name}")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error running test: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        
        print(f"{name}: {status}")
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
