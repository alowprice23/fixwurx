#!/usr/bin/env python3
"""
Verification Module

This module provides test execution and validation capabilities for patches,
including canary testing, smoke testing, regression detection, and test coverage tracking.
"""

import os
import sys
import json
import logging
import subprocess
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("verification.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Verification")

class TestResult:
    """
    Container for test execution results.
    """
    
    def __init__(self, success: bool, output: str, exit_code: int, duration: float):
        """
        Initialize test result.
        
        Args:
            success: Whether the test passed
            output: Test output
            exit_code: Exit code from the test process
            duration: Test duration in seconds
        """
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.duration = duration
        self.errors = []
        self.failures = []
        self.skipped = []
        
        # Parse output for errors and failures
        self._parse_output()
    
    def _parse_output(self) -> None:
        """
        Parse test output to extract errors, failures, and skipped tests.
        """
        try:
            # Look for common test output patterns
            lines = self.output.split('\n')
            
            for line in lines:
                # Check for error patterns
                if "ERROR:" in line or "Error:" in line:
                    self.errors.append(line)
                
                # Check for failure patterns
                elif "FAIL:" in line or "Failure:" in line or "AssertionError" in line:
                    self.failures.append(line)
                
                # Check for skipped test patterns
                elif "SKIP:" in line or "Skipped:" in line:
                    self.skipped.append(line)
        except Exception as e:
            logger.error(f"Error parsing test output: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "duration": self.duration,
            "errors": self.errors,
            "failures": self.failures,
            "skipped": self.skipped,
            "output": self.output
        }

class CoverageResult:
    """
    Container for test coverage results.
    """
    
    def __init__(self, coverage_data: Dict[str, Any]):
        """
        Initialize coverage result.
        
        Args:
            coverage_data: Coverage data
        """
        self.total_coverage = coverage_data.get("total_coverage", 0.0)
        self.file_coverage = coverage_data.get("file_coverage", {})
        self.uncovered_lines = coverage_data.get("uncovered_lines", {})
        self.timestamp = coverage_data.get("timestamp", time.time())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "total_coverage": self.total_coverage,
            "file_coverage": self.file_coverage,
            "uncovered_lines": self.uncovered_lines,
            "timestamp": self.timestamp
        }

class TestExecutor:
    """
    Executes tests for patch validation.
    """
    
    def __init__(self, project_root: str, config: Dict[str, Any] = None):
        """
        Initialize test executor.
        
        Args:
            project_root: Root directory of the project
            config: Configuration options
        """
        self.project_root = os.path.abspath(project_root)
        self.config = config or {}
        self.test_results = {}
        self.coverage_results = {}
        
        # Configure from project settings or use defaults
        self.test_command = self.config.get("test_command", "pytest")
        self.coverage_command = self.config.get("coverage_command", "pytest --cov")
        self.test_directory = self.config.get("test_directory", "tests")
        self.timeout = self.config.get("timeout", 60)  # seconds
        
        logger.info(f"Test executor initialized for project: {self.project_root}")
    
    def run_test(self, test_path: str, env_vars: Dict[str, str] = None) -> TestResult:
        """
        Run a specific test.
        
        Args:
            test_path: Path to the test file or directory
            env_vars: Environment variables for the test
            
        Returns:
            Test result
        """
        abs_test_path = os.path.join(self.project_root, test_path)
        
        if not os.path.exists(abs_test_path):
            logger.error(f"Test path does not exist: {abs_test_path}")
            return TestResult(False, f"Test path does not exist: {abs_test_path}", 1, 0.0)
        
        # Prepare environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # Prepare command
        command = f"{self.test_command} {test_path}"
        logger.info(f"Running test: {command}")
        
        # Run test
        start_time = time.time()
        try:
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=self.project_root, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                env=env,
                universal_newlines=True
            )
            
            try:
                stdout, _ = process.communicate(timeout=self.timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = process.communicate()
                exit_code = -1
                logger.warning(f"Test timed out after {self.timeout} seconds: {test_path}")
            
            duration = time.time() - start_time
            success = exit_code == 0
            
            # Create result
            result = TestResult(success, stdout, exit_code, duration)
            
            # Store result
            self.test_results[test_path] = result
            
            logger.info(f"Test completed: {test_path}, success: {success}, duration: {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error running test {test_path}: {e}")
            duration = time.time() - start_time
            return TestResult(False, str(e), 1, duration)
    
    def run_all_tests(self, env_vars: Dict[str, str] = None) -> Dict[str, TestResult]:
        """
        Run all tests in the test directory.
        
        Args:
            env_vars: Environment variables for the tests
            
        Returns:
            Dictionary of test results
        """
        test_dir = os.path.join(self.project_root, self.test_directory)
        
        if not os.path.exists(test_dir):
            logger.error(f"Test directory does not exist: {test_dir}")
            return {}
        
        # Collect test files
        test_files = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), self.project_root)
                    test_files.append(rel_path)
        
        # Run tests
        results = {}
        for test_file in test_files:
            result = self.run_test(test_file, env_vars)
            results[test_file] = result
        
        return results
    
    def measure_coverage(self, module_path: str = None) -> CoverageResult:
        """
        Measure test coverage for a module.
        
        Args:
            module_path: Path to the module to measure coverage for, or None for all
            
        Returns:
            Coverage result
        """
        target = module_path or "."
        abs_target = os.path.join(self.project_root, target)
        
        if not os.path.exists(abs_target):
            logger.error(f"Target path does not exist: {abs_target}")
            return CoverageResult({
                "total_coverage": 0.0,
                "file_coverage": {},
                "uncovered_lines": {}
            })
        
        # Prepare command
        coverage_output = os.path.join(self.project_root, "coverage.json")
        command = f"{self.coverage_command} {target} --cov-report=json:{coverage_output}"
        logger.info(f"Measuring coverage: {command}")
        
        # Run coverage
        try:
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=self.project_root, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            stdout, _ = process.communicate(timeout=self.timeout)
            exit_code = process.returncode
            
            if exit_code != 0:
                logger.error(f"Coverage measurement failed: {stdout}")
                return CoverageResult({
                    "total_coverage": 0.0,
                    "file_coverage": {},
                    "uncovered_lines": {}
                })
            
            # Parse coverage data
            if os.path.exists(coverage_output):
                with open(coverage_output, 'r') as f:
                    coverage_data = json.load(f)
                
                # Extract relevant information
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                file_coverage = {}
                uncovered_lines = {}
                
                for file_path, file_data in coverage_data.get("files", {}).items():
                    # Skip files outside the target
                    if module_path and not file_path.startswith(module_path):
                        continue
                    
                    file_coverage[file_path] = file_data.get("summary", {}).get("percent_covered", 0.0)
                    missing_lines = file_data.get("missing_lines", [])
                    if missing_lines:
                        uncovered_lines[file_path] = missing_lines
                
                # Create result
                result = CoverageResult({
                    "total_coverage": total_coverage,
                    "file_coverage": file_coverage,
                    "uncovered_lines": uncovered_lines,
                    "timestamp": time.time()
                })
                
                # Store result
                self.coverage_results[module_path or "total"] = result
                
                logger.info(f"Coverage measured: {target}, total coverage: {total_coverage:.2f}%")
                return result
            else:
                logger.error(f"Coverage output file not found: {coverage_output}")
                return CoverageResult({
                    "total_coverage": 0.0,
                    "file_coverage": {},
                    "uncovered_lines": {}
                })
        except Exception as e:
            logger.error(f"Error measuring coverage for {target}: {e}")
            return CoverageResult({
                "total_coverage": 0.0,
                "file_coverage": {},
                "uncovered_lines": {}
            })

class SmokeTestRunner:
    """
    Runs smoke tests to verify basic functionality.
    """
    
    def __init__(self, project_root: str, config: Dict[str, Any] = None):
        """
        Initialize smoke test runner.
        
        Args:
            project_root: Root directory of the project
            config: Configuration options
        """
        self.project_root = os.path.abspath(project_root)
        self.config = config or {}
        self.test_results = {}
        
        # Configure from project settings
        self.smoke_tests = self.config.get("smoke_tests", [])
        self.timeout = self.config.get("timeout", 60)  # seconds
        
        logger.info(f"Smoke test runner initialized for project: {self.project_root}")
    
    def run_smoke_test(self, test_command: str, expected_output: str = None, 
                       expected_exit_code: int = 0) -> TestResult:
        """
        Run a smoke test.
        
        Args:
            test_command: Command to run
            expected_output: Expected output (or None to ignore)
            expected_exit_code: Expected exit code
            
        Returns:
            Test result
        """
        logger.info(f"Running smoke test: {test_command}")
        
        # Run command
        start_time = time.time()
        try:
            process = subprocess.Popen(
                test_command, 
                shell=True, 
                cwd=self.project_root, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            try:
                stdout, _ = process.communicate(timeout=self.timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = process.communicate()
                exit_code = -1
                logger.warning(f"Smoke test timed out after {self.timeout} seconds: {test_command}")
            
            duration = time.time() - start_time
            
            # Check results
            exit_code_match = exit_code == expected_exit_code
            output_match = True
            
            if expected_output is not None:
                output_match = expected_output in stdout
            
            success = exit_code_match and output_match
            
            # Create result
            result = TestResult(success, stdout, exit_code, duration)
            
            # Store result
            self.test_results[test_command] = result
            
            logger.info(f"Smoke test completed: {test_command}, success: {success}, duration: {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error running smoke test {test_command}: {e}")
            duration = time.time() - start_time
            return TestResult(False, str(e), 1, duration)
    
    def run_all_smoke_tests(self) -> Dict[str, TestResult]:
        """
        Run all configured smoke tests.
        
        Returns:
            Dictionary of test results
        """
        results = {}
        
        for test in self.smoke_tests:
            command = test.get("command")
            expected_output = test.get("expected_output")
            expected_exit_code = test.get("expected_exit_code", 0)
            
            if command:
                result = self.run_smoke_test(command, expected_output, expected_exit_code)
                results[command] = result
        
        return results
    
    def add_smoke_test(self, command: str, expected_output: str = None, 
                      expected_exit_code: int = 0) -> None:
        """
        Add a smoke test to the configuration.
        
        Args:
            command: Command to run
            expected_output: Expected output (or None to ignore)
            expected_exit_code: Expected exit code
        """
        self.smoke_tests.append({
            "command": command,
            "expected_output": expected_output,
            "expected_exit_code": expected_exit_code
        })
        
        logger.info(f"Smoke test added: {command}")

class CanaryTestRunner:
    """
    Runs canary tests in isolated environments.
    """
    
    def __init__(self, project_root: str, config: Dict[str, Any] = None):
        """
        Initialize canary test runner.
        
        Args:
            project_root: Root directory of the project
            config: Configuration options
        """
        self.project_root = os.path.abspath(project_root)
        self.config = config or {}
        self.test_results = {}
        
        # Configure from project settings
        self.canary_environments = self.config.get("canary_environments", [])
        self.timeout = self.config.get("timeout", 300)  # seconds
        
        logger.info(f"Canary test runner initialized for project: {self.project_root}")
    
    def run_canary_test(self, environment: Dict[str, Any], test_command: str) -> TestResult:
        """
        Run a canary test in an isolated environment.
        
        Args:
            environment: Environment configuration
            test_command: Command to run
            
        Returns:
            Test result
        """
        env_type = environment.get("type", "docker")
        env_name = environment.get("name", f"canary-{int(time.time())}")
        env_image = environment.get("image", "python:3.9")
        
        logger.info(f"Running canary test in environment {env_name} ({env_type}): {test_command}")
        
        if env_type == "docker":
            return self._run_docker_canary(env_name, env_image, test_command)
        elif env_type == "virtualenv":
            return self._run_virtualenv_canary(env_name, test_command)
        else:
            logger.error(f"Unsupported canary environment type: {env_type}")
            return TestResult(False, f"Unsupported canary environment type: {env_type}", 1, 0.0)
    
    def _run_docker_canary(self, env_name: str, env_image: str, test_command: str) -> TestResult:
        """
        Run a canary test in a Docker container.
        
        Args:
            env_name: Environment name
            env_image: Docker image
            test_command: Command to run
            
        Returns:
            Test result
        """
        # In a real implementation, this would create a Docker container
        # and run the test command inside it. For now, we'll just simulate it.
        
        # Mock implementation
        start_time = time.time()
        
        # Simulate success/failure
        success = random.random() > 0.2  # 80% success rate
        
        # Simulate output
        output = f"Running test in Docker container {env_name} using image {env_image}\n"
        output += f"Command: {test_command}\n"
        
        if success:
            output += "Test completed successfully\n"
            exit_code = 0
        else:
            output += "Test failed with an error\n"
            exit_code = 1
        
        duration = time.time() - start_time
        
        # Create result
        result = TestResult(success, output, exit_code, duration)
        
        # Store result
        self.test_results[f"docker:{env_name}:{test_command}"] = result
        
        logger.info(f"Docker canary test completed: {test_command}, success: {success}, duration: {duration:.2f}s")
        return result
    
    def _run_virtualenv_canary(self, env_name: str, test_command: str) -> TestResult:
        """
        Run a canary test in a virtualenv.
        
        Args:
            env_name: Environment name
            test_command: Command to run
            
        Returns:
            Test result
        """
        # In a real implementation, this would create a virtualenv
        # and run the test command inside it. For now, we'll just simulate it.
        
        # Mock implementation
        start_time = time.time()
        
        # Simulate success/failure
        success = random.random() > 0.2  # 80% success rate
        
        # Simulate output
        output = f"Running test in virtualenv {env_name}\n"
        output += f"Command: {test_command}\n"
        
        if success:
            output += "Test completed successfully\n"
            exit_code = 0
        else:
            output += "Test failed with an error\n"
            exit_code = 1
        
        duration = time.time() - start_time
        
        # Create result
        result = TestResult(success, output, exit_code, duration)
        
        # Store result
        self.test_results[f"virtualenv:{env_name}:{test_command}"] = result
        
        logger.info(f"Virtualenv canary test completed: {test_command}, success: {success}, duration: {duration:.2f}s")
        return result
    
    def run_all_canary_tests(self, test_command: str) -> Dict[str, TestResult]:
        """
        Run a test command in all canary environments.
        
        Args:
            test_command: Command to run
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        for env in self.canary_environments:
            result = self.run_canary_test(env, test_command)
            env_name = env.get("name", f"canary-{int(time.time())}")
            results[env_name] = result
        
        return results
    
    def add_canary_environment(self, env_type: str, env_name: str, 
                              image: str = None) -> None:
        """
        Add a canary environment to the configuration.
        
        Args:
            env_type: Environment type (docker, virtualenv)
            env_name: Environment name
            image: Docker image (for docker environments)
        """
        env = {
            "type": env_type,
            "name": env_name
        }
        
        if image:
            env["image"] = image
        
        self.canary_environments.append(env)
        
        logger.info(f"Canary environment added: {env_name} ({env_type})")

class RegressionDetector:
    """
    Detects regressions in test results.
    """
    
    def __init__(self, project_root: str, config: Dict[str, Any] = None):
        """
        Initialize regression detector.
        
        Args:
            project_root: Root directory of the project
            config: Configuration options
        """
        self.project_root = os.path.abspath(project_root)
        self.config = config or {}
        
        # Configure from project settings
        self.history_file = self.config.get("history_file", "test_history.json")
        self.history_size = self.config.get("history_size", 10)
        
        # Load test history
        self.test_history = self._load_history()
        
        logger.info(f"Regression detector initialized for project: {self.project_root}")
    
    def _load_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load test history from file.
        
        Returns:
            Test history dictionary
        """
        history_path = os.path.join(self.project_root, self.history_file)
        
        if not os.path.exists(history_path):
            return {}
        
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading test history: {e}")
            return {}
    
    def _save_history(self) -> None:
        """
        Save test history to file.
        """
        history_path = os.path.join(self.project_root, self.history_file)
        
        try:
            with open(history_path, 'w') as f:
                json.dump(self.test_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving test history: {e}")
    
    def add_test_result(self, test_id: str, result: TestResult) -> None:
        """
        Add a test result to the history.
        
        Args:
            test_id: Test identifier
            result: Test result
        """
        if test_id not in self.test_history:
            self.test_history[test_id] = []
        
        # Add result to history
        self.test_history[test_id].append(result.to_dict())
        
        # Limit history size
        if len(self.test_history[test_id]) > self.history_size:
            self.test_history[test_id] = self.test_history[test_id][-self.history_size:]
        
        # Save history
        self._save_history()
    
    def detect_regression(self, test_id: str, current_result: TestResult) -> Dict[str, Any]:
        """
        Detect if a test result indicates a regression.
        
        Args:
            test_id: Test identifier
            current_result: Current test result
            
        Returns:
            Regression detection result
        """
        history = self.test_history.get(test_id, [])
        
        # Need at least one previous result to detect regression
        if not history:
            logger.info(f"No history for test {test_id}, cannot detect regression")
            return {
                "regression_detected": False,
                "reason": "No history available"
            }
        
        # Get previous results
        previous_results = history[-1:]  # Use only the most recent result
        
        # Check for regression
        regression = False
        regression_type = None
        
        if all(result["success"] for result in previous_results) and not current_result.success:
            regression = True
            regression_type = "test_failure"
        elif all(not result["errors"] for result in previous_results) and current_result.errors:
            regression = True
            regression_type = "new_errors"
        elif all(not result["failures"] for result in previous_results) and current_result.failures:
            regression = True
            regression_type = "new_failures"
        
        # Add current result to history
        self.add_test_result(test_id, current_result)
        
        if regression:
            logger.warning(f"Regression detected for test {test_id}: {regression_type}")
            return {
                "regression_detected": True,
                "regression_type": regression_type,
                "previous_results": previous_results,
                "current_result": current_result.to_dict()
            }
        else:
            logger.info(f"No regression detected for test {test_id}")
            return {
                "regression_detected": False,
                "current_result": current_result.to_dict()
            }

class Verifier:
    """
    Main verification class that combines all verification components.
    """
    
    def __init__(self, project_root: str, config: Dict[str, Any] = None):
        """
        Initialize verifier.
        
        Args:
            project_root: Root directory of the project
            config: Configuration options
        """
        self.project_root = os.path.abspath(project_root)
        self.config = config or {}
        
        # Initialize components
        self.test_executor = TestExecutor(project_root, config)
        self.smoke_test_runner = SmokeTestRunner(project_root, config)
        self.canary_test_runner = CanaryTestRunner(project_root, config)
        self.regression_detector = RegressionDetector(project_root, config)
        
        logger.info(f"Verifier initialized for project: {project_root}")
    
    def verify_patch(self, patch_file: str, test_paths: List[str] = None) -> Dict[str, Any]:
        """
        Verify a patch by running tests.
        
        Args:
            patch_file: Path to the patch file
            test_paths: List of test paths to run, or None for all tests
            
        Returns:
            Verification result
        """
        logger.info(f"Verifying patch: {patch_file}")
        
        # Read patch to extract affected files
        affected_files = self._extract_affected_files(patch_file)
        
        # Determine tests to run
        if not test_paths:
            test_paths = self._find_tests_for_files(affected_files)
        
        if not test_paths:
            logger.warning(f"No tests found for patch {patch_file}")
            return {
                "success": False,
                "error": "No tests found for patch"
            }
        
        # Run tests
        test_results = {}
        regressions = {}
        
        for test_path in test_paths:
            result = self.test_executor.run_test(test_path)
            test_results[test_path] = result.to_dict()
            
            # Check for regressions
            regression = self.regression_detector.detect_regression(test_path, result)
            if regression["regression_detected"]:
                regressions[test_path] = regression
        
        # Measure coverage
        coverage_result = self.test_executor.measure_coverage()
        
        # Run smoke tests
        smoke_results = self.smoke_test_runner.run_all_smoke_tests()
        smoke_test_results = {command: result.to_dict() for command, result in smoke_results.items()}
        
        # Determine overall success
        success = all(result["success"] for result in test_results.values())
        success = success and all(result.to_dict()["success"] for result in smoke_results.values())
        success = success and not regressions
        
        return {
            "success": success,
            "patch_file": patch_file,
            "affected_files": affected_files,
            "test_results": test_results,
            "coverage": coverage_result.to_dict(),
            "smoke_test_results": smoke_test_results,
            "regressions": regressions
        }
    
    def run_canary_tests(self, test_command: str) -> Dict[str, Any]:
        """
        Run canary tests in isolated environments.
        
        Args:
            test_command: Command to run
            
        Returns:
            Canary test results
        """
        logger.info(f"Running canary tests: {test_command}")
        
        # Run canary tests
        canary_results = self.canary_test_runner.run_all_canary_tests(test_command)
        
        # Determine overall success
        success = all(result.success for result in canary_results.values())
        
        return {
            "success": success,
            "test_command": test_command,
            "results": {env: result.to_dict() for env, result in canary_results.items()}
        }
    
    def verify_coverage(self, module_path: str = None, min_coverage: float = 80.0) -> Dict[str, Any]:
        """
        Verify test coverage for a module.
        
        Args:
            module_path: Path to the module to verify coverage for, or None for all
            min_coverage: Minimum required coverage
            
        Returns:
            Verification result
        """
        logger.info(f"Verifying coverage for {module_path or 'all modules'}")
        
        # Measure coverage
        coverage_result = self.test_executor.measure_coverage(module_path)
        
        # Check if coverage meets the minimum requirement
        success = coverage_result.total_coverage >= min_coverage
        
        # Find files with low coverage
        low_coverage_files = {
            file: coverage 
            for file, coverage in coverage_result.file_coverage.items() 
            if coverage < min_coverage
        }
        
        # Create result
        result = {
            "success": success,
            "module_path": module_path,
            "min_coverage": min_coverage,
            "total_coverage": coverage_result.total_coverage,
            "file_coverage": coverage_result.file_coverage,
            "low_coverage_files": low_coverage_files,
            "uncovered_lines": coverage_result.uncovered_lines
        }
        
        if success:
            logger.info(f"Coverage verification passed: {coverage_result.total_coverage:.2f}% >= {min_coverage:.2f}%")
        else:
            logger.warning(f"Coverage verification failed: {coverage_result.total_coverage:.2f}% < {min_coverage:.2f}%")
            logger.warning(f"Files with low coverage: {len(low_coverage_files)}")
        
        return result
    
    def _extract_affected_files(self, patch_file: str) -> List[str]:
        """
        Extract affected files from a patch file.
        
        Args:
            patch_file: Path to the patch file
            
        Returns:
            List of affected files
        """
        abs_patch_file = os.path.join(self.project_root, patch_file)
        
        if not os.path.exists(abs_patch_file):
            logger.error(f"Patch file does not exist: {abs_patch_file}")
            return []
        
        affected_files = []
        
        try:
            # Read patch file
            with open(abs_patch_file, 'r') as f:
                patch_content = f.read()
            
            # Extract file paths
            lines = patch_content.split('\n')
            for line in lines:
                if line.startswith('+++') or line.startswith('---'):
                    # Extract file path
                    parts = line.split()
                    if len(parts) > 1:
                        file_path = parts[1]
                        # Remove prefix (a/ or b/)
                        if file_path.startswith(('a/', 'b/')):
                            file_path = file_path[2:]
                        
                        # Convert to relative path
                        rel_path = os.path.relpath(file_path, self.project_root)
                        
                        # Add to affected files
                        if rel_path not in affected_files:
                            affected_files.append(rel_path)
            
            return affected_files
        except Exception as e:
            logger.error(f"Error extracting affected files from patch {patch_file}: {e}")
            return []
    
    def _find_tests_for_files(self, affected_files: List[str]) -> List[str]:
        """
        Find tests that cover the affected files.
        
        Args:
            affected_files: List of affected files
            
        Returns:
            List of test paths
        """
        test_paths = []
        
        # In a real implementation, this would analyze the project structure
        # to find tests that cover the affected files. For now, we'll use a
        # simple heuristic.
        
        test_dir = os.path.join(self.project_root, self.test_executor.test_directory)
        
        if not os.path.exists(test_dir):
            logger.error(f"Test directory does not exist: {test_dir}")
            return []
        
        # Collect all test files
        all_test_files = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), self.project_root)
                    all_test_files.append(rel_path)
        
        # Find tests that match affected files
        for affected_file in affected_files:
            base_name = os.path.basename(affected_file)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Look for tests that match the affected file name
            for test_file in all_test_files:
                test_base = os.path.basename(test_file)
                
                # Check if test file matches affected file
                if test_base == f"test_{base_name}" or test_base == f"test_{name_without_ext}.py":
                    test_paths.append(test_file)
        
        # If no specific tests were found, run all tests
        if not test_paths:
            logger.warning(f"No specific tests found for affected files, running all tests")
            test_paths = all_test_files
        
        return test_paths

# API Functions

def create_verifier(project_root: str, config: Dict[str, Any] = None) -> Verifier:
    """
    Create and initialize a verifier.
    
    Args:
        project_root: Root directory of the project
        config: Configuration options
        
    Returns:
        Initialized verifier
    """
    verifier = Verifier(project_root, config)
    return verifier

def verify_patch(project_root: str, patch_file: str, 
                test_paths: List[str] = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Verify a patch by running tests.
    
    Args:
        project_root: Root directory of the project
        patch_file: Path to the patch file
        test_paths: List of test paths to run, or None for all tests
        config: Configuration options
        
    Returns:
        Verification result
    """
    verifier = create_verifier(project_root, config)
    return verifier.verify_patch(patch_file, test_paths)

def verify_coverage(project_root: str, module_path: str = None, 
                   min_coverage: float = 80.0, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Verify test coverage for a module.
    
    Args:
        project_root: Root directory of the project
        module_path: Path to the module to verify coverage for, or None for all
        min_coverage: Minimum required coverage
        config: Configuration options
        
    Returns:
        Verification result
    """
    verifier = create_verifier(project_root, config)
    return verifier.verify_coverage(module_path, min_coverage)

def run_canary_tests(project_root: str, test_command: str, 
                    config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run canary tests in isolated environments.
    
    Args:
        project_root: Root directory of the project
        test_command: Command to run
        config: Configuration options
        
    Returns:
        Canary test results
    """
    verifier = create_verifier(project_root, config)
    return verifier.run_canary_tests(test_command)


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Verification Tool")
    parser.add_argument("--project", help="Project root directory", default=".")
    parser.add_argument("--patch", help="Patch file to verify")
    parser.add_argument("--tests", nargs="+", help="Test paths to run")
    parser.add_argument("--module", help="Module to verify coverage for")
    parser.add_argument("--min-coverage", type=float, default=80.0, help="Minimum required coverage")
    parser.add_argument("--canary", help="Run canary tests with the given command")
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = create_verifier(args.project)
    
    # Verify patch
    if args.patch:
        result = verifier.verify_patch(args.patch, args.tests)
        
        print("\nPatch Verification Result:")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"Affected files: {len(result['affected_files'])}")
            print(f"Tests run: {len(result['test_results'])}")
            print(f"Coverage: {result['coverage']['total_coverage']:.2f}%")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
            if 'test_results' in result:
                print("\nFailed Tests:")
                for test_path, test_result in result['test_results'].items():
                    if not test_result['success']:
                        print(f"  {test_path}")
                        print(f"    Errors: {len(test_result['errors'])}")
                        print(f"    Failures: {len(test_result['failures'])}")
            
            if 'regressions' in result and result['regressions']:
                print("\nRegressions:")
                for test_path, regression in result['regressions'].items():
                    print(f"  {test_path}: {regression['regression_type']}")
    
    # Verify coverage
    elif args.module is not None:
        result = verifier.verify_coverage(args.module, args.min_coverage)
        
        print("\nCoverage Verification Result:")
        print(f"Success: {result['success']}")
        print(f"Total coverage: {result['total_coverage']:.2f}%")
        print(f"Minimum required coverage: {result['min_coverage']:.2f}%")
        
        if not result['success']:
            print("\nFiles with low coverage:")
            for file, coverage in result['low_coverage_files'].items():
                print(f"  {file}: {coverage:.2f}%")
    
    # Run canary tests
    elif args.canary:
        result = verifier.run_canary_tests(args.canary)
        
        print("\nCanary Test Result:")
        print(f"Success: {result['success']}")
        print(f"Command: {result['test_command']}")
        
        print("\nEnvironment Results:")
        for env, env_result in result['results'].items():
            print(f"  {env}: {'Passed' if env_result['success'] else 'Failed'}")
    
    # Print help
    else:
        parser.print_help()
