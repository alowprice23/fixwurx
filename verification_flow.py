#!/usr/bin/env python3
"""
verification_flow.py
────────────────────
Implements the verification flow for the FixWurx system.

This module provides the core flow for verifying implemented fixes, including
test execution, code quality checks, regression detection, and verification
reporting. It integrates with various components of the system including
the agent system, triangulation engine, and neural matrix.
"""

import os
import sys
import json
import logging
import time
import uuid
import shutil
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime

# Import core components
from triangulation_engine import TriangulationEngine
from neural_matrix_core import NeuralMatrix
from meta_agent import MetaAgent
from resource_manager import ResourceManager
from storage_manager import StorageManager
from verifier_agent import VerifierAgent

# Configure logging
logger = logging.getLogger("VerificationFlow")

class VerificationFlow:
    """
    Implements the verification flow for the FixWurx system.
    
    This class orchestrates the entire verification process, from retrieving
    implementation results to running tests and generating verification reports.
    It serves as the main entry point for the verification subsystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the verification flow.
        
        Args:
            config: Configuration for the verification flow.
        """
        self.config = config or {}
        self.triangulation_engine = TriangulationEngine()
        self.neural_matrix = NeuralMatrix()
        self.meta_agent = MetaAgent()
        self.resource_manager = ResourceManager()
        self.storage_manager = StorageManager()
        self.verifier_agent = VerifierAgent()
        
        # Initialize state
        self.current_verification_id = None
        self.current_context = {}
        self.verified_fixes = []
        
        # Configure test settings
        self.test_timeout = self.config.get("test_timeout", 60)  # 60 seconds default
        self.coverage_threshold = self.config.get("coverage_threshold", 0.7)  # 70% coverage minimum
        
        logger.info("Verification Flow initialized")
    
    def start_verification(self, 
                          implementation_id: str, 
                          verification_options: Dict[str, Any] = None) -> str:
        """
        Start the verification process for an implemented fix.
        
        Args:
            implementation_id: ID of the implementation to verify.
            verification_options: Options for the verification process.
            
        Returns:
            Verification ID for the verification process.
        """
        verification_options = verification_options or {}
        
        # Generate a verification ID
        timestamp = int(time.time())
        verification_id = f"verify_{timestamp}_{str(uuid.uuid4())[:8]}"
        self.current_verification_id = verification_id
        
        # Get the implementation
        implementation = self.storage_manager.get_implementation(implementation_id)
        if not implementation:
            raise ValueError(f"Implementation with ID {implementation_id} not found")
        
        # Initialize verification context
        self.current_context = {
            "verification_id": verification_id,
            "implementation_id": implementation_id,
            "start_time": timestamp,
            "options": verification_options,
            "status": "started",
            "paths_verified": 0,
            "paths_passed": 0,
            "paths_failed": 0
        }
        
        logger.info(f"Starting verification {verification_id} for implementation {implementation_id}")
        
        # Trigger the verification flow
        self._execute_verification_flow(implementation, verification_options)
        
        return verification_id
    
    def _execute_verification_flow(self, 
                                  implementation: Dict[str, Any], 
                                  verification_options: Dict[str, Any]) -> None:
        """
        Execute the verification flow.
        
        Args:
            implementation: Implementation data.
            verification_options: Options for the verification process.
        """
        try:
            # Phase 1: Prepare verification
            logger.info("Phase 1: Prepare verification")
            verification_plan = self._prepare_verification(implementation, verification_options)
            
            # Phase 2: Run tests
            logger.info("Phase 2: Run tests")
            test_results = self._run_tests(verification_plan)
            
            # Phase 3: Check code quality
            logger.info("Phase 3: Check code quality")
            quality_results = self._check_code_quality(verification_plan)
            
            # Phase 4: Check for regressions
            logger.info("Phase 4: Check for regressions")
            regression_results = self._check_for_regressions(verification_plan, test_results)
            
            # Phase 5: Generate verification report
            logger.info("Phase 5: Generate verification report")
            verification_report = self._generate_verification_report(
                verification_plan, test_results, quality_results, regression_results
            )
            
            # Update context
            self.current_context["status"] = "completed"
            self.current_context["end_time"] = int(time.time())
            self.current_context["paths_verified"] = len(test_results)
            self.current_context["paths_passed"] = sum(1 for r in test_results if r.get("status") == "passed")
            self.current_context["paths_failed"] = sum(1 for r in test_results if r.get("status") == "failed")
            self.current_context["test_results"] = test_results
            self.current_context["quality_results"] = quality_results
            self.current_context["regression_results"] = regression_results
            self.current_context["verification_report"] = verification_report
            
            # Store verification results
            verification_result = {
                "verification_id": self.current_verification_id,
                "implementation_id": implementation.get("implementation_id"),
                "timestamp": int(time.time()),
                "test_results": test_results,
                "quality_results": quality_results,
                "regression_results": regression_results,
                "verification_report": verification_report
            }
            self.verified_fixes.append(verification_result)
            
            # Store verification in storage manager
            self.storage_manager.store_verification(verification_result)
            
            # Notify the Meta Agent
            self.meta_agent.notify_verification_complete(verification_result)
            
            logger.info(f"Verification {self.current_verification_id} completed with {self.current_context['paths_passed']} passed and {self.current_context['paths_failed']} failed paths")
            
        except Exception as e:
            logger.error(f"Error in verification flow: {e}")
            self.current_context["status"] = "failed"
            self.current_context["error"] = str(e)
            self.current_context["end_time"] = int(time.time())
            raise
    
    def _prepare_verification(self, 
                             implementation: Dict[str, Any], 
                             verification_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the verification plan.
        
        Args:
            implementation: Implementation data.
            verification_options: Options for the verification process.
            
        Returns:
            Verification plan.
        """
        # Get implementation results
        results = implementation.get("results", [])
        
        # Filter results if needed
        if "path_ids" in verification_options:
            path_ids = verification_options["path_ids"]
            results = [r for r in results if r.get("path_id") in path_ids]
        
        # Get associated plan if available
        plan_id = implementation.get("plan_id")
        plan = self.storage_manager.get_plan(plan_id) if plan_id else None
        
        # Get modified files
        modified_files = set()
        for result in results:
            if result.get("status") == "success":
                for change in result.get("changes", []):
                    modified_files.add(change.get("file_path"))
                    if change.get("type") == "rename":
                        modified_files.add(change.get("new_path"))
        
        # Determine test scope
        test_scope = verification_options.get("test_scope", "modified")
        if test_scope == "modified":
            # Test files related to modified files
            test_files = self._find_related_test_files(modified_files)
        elif test_scope == "all":
            # Test all available tests
            test_files = self._find_all_test_files()
        else:
            # Specific test files
            test_files = verification_options.get("test_files", [])
        
        # Create verification plan
        verification_plan = {
            "verification_id": self.current_verification_id,
            "implementation_id": implementation.get("implementation_id"),
            "plan_id": plan_id,
            "results": results,
            "modified_files": list(modified_files),
            "test_files": test_files,
            "test_scope": test_scope,
            "options": verification_options
        }
        
        # Update context
        self.current_context["verification_plan"] = verification_plan
        
        return verification_plan
    
    def _find_related_test_files(self, modified_files: set) -> List[str]:
        """
        Find test files related to the modified files.
        
        Args:
            modified_files: Set of modified file paths.
            
        Returns:
            List of related test file paths.
        """
        test_files = []
        
        for file_path in modified_files:
            if not os.path.exists(file_path):
                continue
                
            # Get file name without extension
            file_name = os.path.basename(file_path)
            name_without_ext, _ = os.path.splitext(file_name)
            
            # Find corresponding test files
            if file_path.endswith(".py"):
                test_patterns = [
                    f"test_{name_without_ext}.py",
                    f"{name_without_ext}_test.py",
                    f"tests/test_{name_without_ext}.py",
                    f"tests/{name_without_ext}_test.py"
                ]
                
                # Look for test files
                for pattern in test_patterns:
                    # Try in same directory
                    dir_path = os.path.dirname(file_path)
                    test_path = os.path.join(dir_path, pattern)
                    if os.path.exists(test_path):
                        test_files.append(test_path)
                    
                    # Try in tests directory
                    tests_dir = os.path.join(os.path.dirname(dir_path), "tests")
                    if os.path.exists(tests_dir):
                        test_path = os.path.join(tests_dir, pattern)
                        if os.path.exists(test_path):
                            test_files.append(test_path)
        
        # Remove duplicates
        test_files = list(set(test_files))
        
        return test_files
    
    def _find_all_test_files(self) -> List[str]:
        """
        Find all test files in the project.
        
        Returns:
            List of all test file paths.
        """
        test_files = []
        
        # Walk through all directories
        for root, _, files in os.walk("."):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
                elif file.endswith("_test.py"):
                    test_files.append(os.path.join(root, file))
        
        return test_files
    
    def _run_tests(self, verification_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run tests for the verification.
        
        Args:
            verification_plan: Verification plan.
            
        Returns:
            List of test results.
        """
        test_results = []
        
        # Get test files
        test_files = verification_plan.get("test_files", [])
        
        # Run tests using verifier agent
        for test_file in test_files:
            if not os.path.exists(test_file):
                logger.warning(f"Test file {test_file} not found")
                continue
            
            logger.info(f"Running tests in {test_file}")
            
            # Determine test runner based on file extension
            if test_file.endswith(".py"):
                result = self._run_python_tests(test_file, verification_plan)
            else:
                # Skip unsupported test files
                logger.warning(f"Unsupported test file format: {test_file}")
                continue
            
            test_results.append(result)
        
        # Run specific test cases if provided
        test_cases = verification_plan.get("options", {}).get("test_cases", [])
        for test_case in test_cases:
            result = self.verifier_agent.run_test_case(test_case)
            test_results.append(result)
        
        return test_results
    
    def _run_python_tests(self, test_file: str, verification_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Python tests.
        
        Args:
            test_file: Path to the test file.
            verification_plan: Verification plan.
            
        Returns:
            Test result.
        """
        # Configure test run
        timeout = verification_plan.get("options", {}).get("timeout", self.test_timeout)
        run_with_coverage = verification_plan.get("options", {}).get("coverage", True)
        
        try:
            # Determine test command
            if run_with_coverage:
                cmd = [
                    "python", "-m", "pytest", test_file, "--verbose",
                    "--cov", "--cov-report", "term-missing"
                ]
            else:
                cmd = ["python", "-m", "pytest", test_file, "--verbose"]
            
            # Run test command
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            # Parse result
            status = "passed" if result.returncode == 0 else "failed"
            
            # Parse coverage if available
            coverage_data = None
            if run_with_coverage:
                coverage_data = self._parse_coverage_data(result.stdout)
            
            # Create result
            test_result = {
                "file": test_file,
                "status": status,
                "duration": end_time - start_time,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "coverage": coverage_data,
                "timestamp": int(time.time())
            }
            
            return test_result
            
        except subprocess.TimeoutExpired:
            # Test timed out
            return {
                "file": test_file,
                "status": "timeout",
                "duration": timeout,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Test timed out after {timeout} seconds",
                "coverage": None,
                "timestamp": int(time.time())
            }
        except Exception as e:
            # Other errors
            return {
                "file": test_file,
                "status": "error",
                "duration": 0,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "coverage": None,
                "timestamp": int(time.time())
            }
    
    def _parse_coverage_data(self, stdout: str) -> Optional[Dict[str, Any]]:
        """
        Parse coverage data from pytest-cov output.
        
        Args:
            stdout: Standard output from pytest-cov.
            
        Returns:
            Coverage data or None if not found.
        """
        try:
            # Find coverage line
            coverage_line = None
            for line in stdout.splitlines():
                if "TOTAL" in line and "%" in line:
                    coverage_line = line
                    break
            
            if not coverage_line:
                return None
            
            # Parse coverage percentage
            parts = coverage_line.split()
            for part in parts:
                if "%" in part:
                    percentage = float(part.replace("%", "")) / 100
                    
                    return {
                        "percentage": percentage,
                        "meets_threshold": percentage >= self.coverage_threshold,
                        "threshold": self.coverage_threshold
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing coverage data: {e}")
            return None
    
    def _check_code_quality(self, verification_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check code quality for the verification.
        
        Args:
            verification_plan: Verification plan.
            
        Returns:
            Code quality results.
        """
        quality_results = {
            "linting_results": [],
            "complexity_results": [],
            "style_results": []
        }
        
        # Get modified files
        modified_files = verification_plan.get("modified_files", [])
        
        # Check if code quality checks are enabled
        run_linting = verification_plan.get("options", {}).get("linting", True)
        run_complexity = verification_plan.get("options", {}).get("complexity", True)
        run_style = verification_plan.get("options", {}).get("style", True)
        
        # Run linting if enabled
        if run_linting:
            linting_results = self._run_linting(modified_files)
            quality_results["linting_results"] = linting_results
        
        # Run complexity checks if enabled
        if run_complexity:
            complexity_results = self._run_complexity_checks(modified_files)
            quality_results["complexity_results"] = complexity_results
        
        # Run style checks if enabled
        if run_style:
            style_results = self._run_style_checks(modified_files)
            quality_results["style_results"] = style_results
        
        return quality_results
    
    def _run_linting(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run linting on the specified files.
        
        Args:
            file_paths: List of file paths to lint.
            
        Returns:
            List of linting results.
        """
        linting_results = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            # Determine linter based on file extension
            _, ext = os.path.splitext(file_path)
            
            if ext == ".py":
                # Use pylint
                try:
                    cmd = ["pylint", file_path, "--output-format=json"]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.stdout:
                        try:
                            issues = json.loads(result.stdout)
                            linting_result = {
                                "file": file_path,
                                "tool": "pylint",
                                "issues": issues,
                                "issue_count": len(issues),
                                "status": "has_issues" if issues else "clean"
                            }
                        except json.JSONDecodeError:
                            # Fallback for non-JSON output
                            linting_result = {
                                "file": file_path,
                                "tool": "pylint",
                                "issues": result.stdout,
                                "issue_count": result.stdout.count("\n"),
                                "status": "has_issues" if result.stdout.strip() else "clean"
                            }
                    else:
                        linting_result = {
                            "file": file_path,
                            "tool": "pylint",
                            "issues": [],
                            "issue_count": 0,
                            "status": "clean"
                        }
                    
                    linting_results.append(linting_result)
                    
                except Exception as e:
                    logger.error(f"Error running pylint on {file_path}: {e}")
            
            # Add support for other file types as needed
        
        return linting_results
    
    def _run_complexity_checks(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run complexity checks on the specified files.
        
        Args:
            file_paths: List of file paths to check.
            
        Returns:
            List of complexity check results.
        """
        complexity_results = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            # Determine complexity checker based on file extension
            _, ext = os.path.splitext(file_path)
            
            if ext == ".py":
                # Use radon
                try:
                    cmd = ["radon", "cc", file_path, "--json"]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.stdout:
                        try:
                            complexity_data = json.loads(result.stdout)
                            
                            # Calculate average complexity
                            total_complexity = 0
                            function_count = 0
                            
                            for function, metrics in complexity_data.get(file_path, {}).items():
                                total_complexity += metrics.get("complexity", 0)
                                function_count += 1
                            
                            avg_complexity = total_complexity / function_count if function_count > 0 else 0
                            
                            complexity_result = {
                                "file": file_path,
                                "tool": "radon",
                                "data": complexity_data,
                                "function_count": function_count,
                                "average_complexity": avg_complexity,
                                "status": "high_complexity" if avg_complexity > 10 else "medium_complexity" if avg_complexity > 5 else "low_complexity"
                            }
                        except json.JSONDecodeError:
                            # Fallback for non-JSON output
                            complexity_result = {
                                "file": file_path,
                                "tool": "radon",
                                "data": result.stdout,
                                "status": "unknown"
                            }
                    else:
                        complexity_result = {
                            "file": file_path,
                            "tool": "radon",
                            "data": {},
                            "function_count": 0,
                            "average_complexity": 0,
                            "status": "low_complexity"
                        }
                    
                    complexity_results.append(complexity_result)
                    
                except Exception as e:
                    logger.error(f"Error running radon on {file_path}: {e}")
            
            # Add support for other file types as needed
        
        return complexity_results
    
    def _run_style_checks(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run style checks on the specified files.
        
        Args:
            file_paths: List of file paths to check.
            
        Returns:
            List of style check results.
        """
        style_results = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            # Determine style checker based on file extension
            _, ext = os.path.splitext(file_path)
            
            if ext == ".py":
                # Use pycodestyle
                try:
                    cmd = ["pycodestyle", file_path]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True
                    )
                    
                    # Parse issues
                    issues = []
                    for line in result.stdout.splitlines():
                        if line.strip():
                            issues.append(line)
                    
                    style_result = {
                        "file": file_path,
                        "tool": "pycodestyle",
                        "issues": issues,
                        "issue_count": len(issues),
                        "status": "has_issues" if issues else "clean"
                    }
                    
                    style_results.append(style_result)
                    
                except Exception as e:
                    logger.error(f"Error running pycodestyle on {file_path}: {e}")
            
            # Add support for other file types as needed
        
        return style_results
    
    def _check_for_regressions(self, 
                              verification_plan: Dict[str, Any], 
                              test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for regressions in the verification.
        
        Args:
            verification_plan: Verification plan.
            test_results: Test results.
            
        Returns:
            Regression check results.
        """
        regression_results = {
            "has_regressions": False,
            "regression_tests": [],
            "new_failures": []
        }
        
        # Get historical test results if available
        implementation_id = verification_plan.get("implementation_id")
        plan_id = verification_plan.get("plan_id")
        
        # Get bug information
        bugs_fixed = []
        if plan_id:
            plan = self.storage_manager.get_plan(plan_id)
            if plan:
                for path in plan.get("paths", []):
                    bug_id = path.get("bug_id")
                    if bug_id and bug_id not in bugs_fixed:
                        bugs_fixed.append(bug_id)
        
        # Check for test failures
        failures = []
        for result in test_results:
            if result.get("status") == "failed":
                failures.append(result)
        
        # Check if any bug-specific tests are failing
        regression_tests = []
        for failure in failures:
            test_file = failure.get("file", "")
            
            # Check if this test file is associated with a fixed bug
            for bug_id in bugs_fixed:
                if bug_id in test_file:
                    regression_tests.append({
                        "test_file": test_file,
                        "bug_id": bug_id,
                        "stdout": failure.get("stdout", ""),
                        "stderr": failure.get("stderr", "")
                    })
        
        # Compare with historical test results
        new_failures = []
        historical_test_results = self._get_historical_test_results()
        
        for failure in failures:
            test_file = failure.get("file", "")
            
            # Check if this test previously passed
            if test_file in historical_test_results and historical_test_results[test_file].get("status") == "passed":
                new_failures.append({
                    "test_file": test_file,
                    "previous_status": "passed",
                    "stdout": failure.get("stdout", ""),
                    "stderr": failure.get("stderr", "")
                })
        
        # Update regression results
        regression_results["regression_tests"] = regression_tests
        regression_results["new_failures"] = new_failures
        regression_results["has_regressions"] = bool(regression_tests) or bool(new_failures)
        
        return regression_results
    
    def _get_historical_test_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get historical test results from previous verifications.
        
        Returns:
            Dictionary mapping test file paths to their most recent results.
        """
        historical_results = {}
        
        # Get previous verification results
        previous_verifications = self.verified_fixes
        
        # Process in reverse order (most recent first)
        for verification in reversed(previous_verifications):
            for test_result in verification.get("test_results", []):
                test_file = test_result.get("file", "")
                
                # Only add if not already present (most recent result)
                if test_file and test_file not in historical_results:
                    historical_results[test_file] = test_result
        
        return historical_results
    
    def _generate_verification_report(self, 
                                     verification_plan: Dict[str, Any], 
                                     test_results: List[Dict[str, Any]], 
                                     quality_results: Dict[str, Any], 
                                     regression_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a verification report.
        
        Args:
            verification_plan: Verification plan.
            test_results: Test results.
            quality_results: Code quality results.
            regression_results: Regression check results.
            
        Returns:
            Verification report.
        """
        # Calculate overall statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.get("status") == "passed")
        failed_tests = sum(1 for r in test_results if r.get("status") == "failed")
        error_tests = sum(1 for r in test_results if r.get("status") == "error")
        timeout_tests = sum(1 for r in test_results if r.get("status") == "timeout")
        
        # Calculate code quality statistics
        linting_issues = sum(r.get("issue_count", 0) for r in quality_results.get("linting_results", []))
        style_issues = sum(r.get("issue_count", 0) for r in quality_results.get("style_results", []))
        
        # Calculate coverage statistics
        coverage_values = [r.get("coverage", {}).get("percentage", 0) for r in test_results if r.get("coverage")]
        avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0
        
        # Determine verification status
        verification_status = "passed"
        
        if regression_results.get("has_regressions"):
            verification_status = "failed"
        elif failed_tests > 0 or error_tests > 0:
            verification_status = "partial"
        
        # Create verification report
        verification_report = {
            "verification_id": self.current_verification_id,
            "timestamp": int(time.time()),
            "status": verification_status,
            "statistics": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "timeout_tests": timeout_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "linting_issues": linting_issues,
                "style_issues": style_issues,
                "average_coverage": avg_coverage
            },
            "has_regressions": regression_results.get("has_regressions", False),
            "regression_count": len(regression_results.get("regression_tests", [])) + len(regression_results.get("new_failures", [])),
            "recommendations": self._generate_recommendations(
                verification_status, 
                test_results, 
                quality_results, 
                regression_results
            )
        }
        
        return verification_report
    
    def _generate_recommendations(self,
                                 verification_status: str,
                                 test_results: List[Dict[str, Any]],
                                 quality_results: Dict[str, Any],
                                 regression_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on verification results.
        
        Args:
            verification_status: Verification status (passed, partial, or failed).
            test_results: Test results.
            quality_results: Code quality results.
            regression_results: Regression check results.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Handle regressions
        if regression_results.get("has_regressions"):
            regression_tests = regression_results.get("regression_tests", [])
            new_failures = regression_results.get("new_failures", [])
            
            if regression_tests:
                recommendations.append({
                    "type": "critical",
                    "title": "Fix bug-specific test failures",
                    "description": f"There are {len(regression_tests)} failing tests that are associated with fixed bugs. This suggests that the fixes may not have fully resolved the issues.",
                    "files": [r.get("test_file") for r in regression_tests]
                })
            
            if new_failures:
                recommendations.append({
                    "type": "critical",
                    "title": "Fix newly failing tests",
                    "description": f"There are {len(new_failures)} tests that were previously passing but are now failing. This suggests that the implemented changes have introduced regressions.",
                    "files": [r.get("test_file") for r in new_failures]
                })
        
        # Handle test failures
        failed_tests = [r for r in test_results if r.get("status") == "failed"]
        if failed_tests and verification_status != "failed":  # Skip if already handling regressions
            recommendations.append({
                "type": "high",
                "title": "Fix failing tests",
                "description": f"There are {len(failed_tests)} failing tests. These failures should be addressed to ensure the implemented changes are working correctly.",
                "files": [r.get("file") for r in failed_tests]
            })
        
        # Handle test errors
        error_tests = [r for r in test_results if r.get("status") == "error"]
        if error_tests:
            recommendations.append({
                "type": "high",
                "title": "Fix test errors",
                "description": f"There are {len(error_tests)} tests with errors. These errors should be addressed to ensure proper test execution.",
                "files": [r.get("file") for r in error_tests]
            })
        
        # Handle test timeouts
        timeout_tests = [r for r in test_results if r.get("status") == "timeout"]
        if timeout_tests:
            recommendations.append({
                "type": "medium",
                "title": "Fix test timeouts",
                "description": f"There are {len(timeout_tests)} tests that timed out. These tests should be optimized or fixed to ensure they complete within the timeout period.",
                "files": [r.get("file") for r in timeout_tests]
            })
        
        # Handle coverage issues
        low_coverage_tests = [r for r in test_results if r.get("coverage") and not r.get("coverage", {}).get("meets_threshold", False)]
        if low_coverage_tests:
            recommendations.append({
                "type": "medium",
                "title": "Improve test coverage",
                "description": f"There are {len(low_coverage_tests)} tests with coverage below the threshold of {self.coverage_threshold * 100}%. Additional tests should be added to improve coverage.",
                "files": [r.get("file") for r in low_coverage_tests]
            })
        
        # Handle linting issues
        linting_results = quality_results.get("linting_results", [])
        issue_files = [r.get("file") for r in linting_results if r.get("status") == "has_issues"]
        if issue_files:
            recommendations.append({
                "type": "low",
                "title": "Fix linting issues",
                "description": f"There are linting issues in {len(issue_files)} files. These issues should be addressed to improve code quality.",
                "files": issue_files
            })
        
        # Handle style issues
        style_results = quality_results.get("style_results", [])
        issue_files = [r.get("file") for r in style_results if r.get("status") == "has_issues"]
        if issue_files:
            recommendations.append({
                "type": "low",
                "title": "Fix style issues",
                "description": f"There are style issues in {len(issue_files)} files. These issues should be addressed to improve code quality.",
                "files": issue_files
            })
        
        # Handle complexity issues
        complexity_results = quality_results.get("complexity_results", [])
        high_complexity_files = [r.get("file") for r in complexity_results if r.get("status") == "high_complexity"]
        if high_complexity_files:
            recommendations.append({
                "type": "medium",
                "title": "Reduce code complexity",
                "description": f"There are {len(high_complexity_files)} files with high code complexity. These files should be refactored to reduce complexity and improve maintainability.",
                "files": high_complexity_files
            })
        
        return recommendations
        
    def get_verification_by_id(self, verification_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a verification by its ID.
        
        Args:
            verification_id: Verification ID.
            
        Returns:
            Verification data or None if not found.
        """
        if verification_id == self.current_verification_id and self.current_context:
            return self.current_context
        
        # Try to find in previous verifications
        for verification in self.verified_fixes:
            if verification.get("verification_id") == verification_id:
                return verification
        
        # Try to retrieve from storage
        return self.storage_manager.get_verification(verification_id)
    
    def get_verification_status(self) -> Dict[str, Any]:
        """
        Get the status of the current verification process.
        
        Returns:
            Verification status data.
        """
        return self.current_context
    
    def save_verification_report(self, 
                               output_path: Optional[str] = None, 
                               format: str = "json") -> str:
        """
        Save the verification report to a file.
        
        Args:
            output_path: Path to save the report. If None, a default path is used.
            format: Report format (json or html).
            
        Returns:
            Path to the saved report.
        """
        if not self.current_context.get("verification_report"):
            raise ValueError("No verification report available to save")
        
        # Create default output path if not provided
        if not output_path:
            timestamp = self.current_context.get("start_time", int(time.time()))
            filename = f"verification_report_{self.current_verification_id}.{format}"
            output_path = os.path.join(os.getcwd(), filename)
        
        # Save report in the specified format
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.current_context["verification_report"], f, indent=2)
        elif format == "html":
            # Generate HTML report
            html_report = self._generate_html_report(self.current_context["verification_report"])
            
            with open(output_path, "w") as f:
                f.write(html_report)
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        logger.info(f"Verification report saved to {output_path}")
        
        return output_path
    
    def _generate_html_report(self, verification_report: Dict[str, Any]) -> str:
        """
        Generate an HTML report from the verification report.
        
        Args:
            verification_report: Verification report data.
            
        Returns:
            HTML report as a string.
        """
        # Generate HTML report
        # This is a simplified implementation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Verification Report: {verification_report.get('verification_id')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .passed {{ background-color: #d4edda; color: #155724; }}
                .partial {{ background-color: #fff3cd; color: #856404; }}
                .failed {{ background-color: #f8d7da; color: #721c24; }}
                .recommendation-critical {{ background-color: #f8d7da; }}
                .recommendation-high {{ background-color: #fff3cd; }}
                .recommendation-medium {{ background-color: #e6f3ff; }}
                .recommendation-low {{ background-color: #f0f0f0; }}
            </style>
        </head>
        <body>
            <h1>Verification Report</h1>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Verification ID</th><td>{verification_report.get('verification_id')}</td></tr>
                <tr><th>Timestamp</th><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(verification_report.get('timestamp', 0)))}</td></tr>
                <tr><th>Status</th><td class="{verification_report.get('status', 'unknown')}">{verification_report.get('status', 'unknown').upper()}</td></tr>
                <tr><th>Has Regressions</th><td>{verification_report.get('has_regressions', False)}</td></tr>
                <tr><th>Regression Count</th><td>{verification_report.get('regression_count', 0)}</td></tr>
            </table>
            
            <h2>Statistics</h2>
            <table>
                <tr>
                    <th>Total Tests</th>
                    <td>{verification_report.get('statistics', {}).get('total_tests', 0)}</td>
                </tr>
                <tr>
                    <th>Passed Tests</th>
                    <td>{verification_report.get('statistics', {}).get('passed_tests', 0)}</td>
                </tr>
                <tr>
                    <th>Failed Tests</th>
                    <td>{verification_report.get('statistics', {}).get('failed_tests', 0)}</td>
                </tr>
                <tr>
                    <th>Error Tests</th>
                    <td>{verification_report.get('statistics', {}).get('error_tests', 0)}</td>
                </tr>
                <tr>
                    <th>Timeout Tests</th>
                    <td>{verification_report.get('statistics', {}).get('timeout_tests', 0)}</td>
                </tr>
                <tr>
                    <th>Pass Rate</th>
                    <td>{verification_report.get('statistics', {}).get('pass_rate', 0) * 100:.2f}%</td>
                </tr>
                <tr>
                    <th>Linting Issues</th>
                    <td>{verification_report.get('statistics', {}).get('linting_issues', 0)}</td>
                </tr>
                <tr>
                    <th>Style Issues</th>
                    <td>{verification_report.get('statistics', {}).get('style_issues', 0)}</td>
                </tr>
                <tr>
                    <th>Average Coverage</th>
                    <td>{verification_report.get('statistics', {}).get('average_coverage', 0) * 100:.2f}%</td>
                </tr>
            </table>
            
            <h2>Recommendations</h2>
        """
        
        # Add recommendations
        for recommendation in verification_report.get('recommendations', []):
            rec_type = recommendation.get('type', 'low')
            html += f"""
            <div class="recommendation-{rec_type}">
                <h3>{recommendation.get('title', 'Untitled Recommendation')}</h3>
                <p>{recommendation.get('description', 'No description available.')}</p>
                <h4>Affected Files:</h4>
                <ul>
            """
            
            for file in recommendation.get('files', []):
                html += f"<li>{file}</li>"
            
            html += """
                </ul>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html

# Main entry point
def verify_implementation(implementation_id: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to verify an implemented fix.
    
    Args:
        implementation_id: ID of the implementation to verify.
        options: Verification options.
        
    Returns:
        Verification report.
    """
    flow = VerificationFlow()
    verification_id = flow.start_verification(implementation_id, options)
    
    # Wait for verification to complete
    while flow.get_verification_status()["status"] not in ["completed", "failed"]:
        time.sleep(0.1)
    
    # Get verification results
    verification_status = flow.get_verification_status()
    
    if verification_status["status"] == "failed":
        logger.error(f"Verification failed: {verification_status.get('error', 'Unknown error')}")
        raise RuntimeError(f"Verification failed: {verification_status.get('error', 'Unknown error')}")
    
    return verification_status["verification_report"]

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python verification_flow.py <implementation_id> [options_json]")
        sys.exit(1)
    
    implementation_id = sys.argv[1]
    
    # Parse options if provided
    options = {}
    if len(sys.argv) > 2:
        try:
            options = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print("Error: options must be a valid JSON string")
            sys.exit(1)
    
    # Run verification
    try:
        report = verify_implementation(implementation_id, options)
        
        # Create output path
        output_path = options.get("output_path", f"verification_report_{int(time.time())}.json")
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Verification report saved to {output_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"Status: {report['status'].upper()}")
        print(f"Tests: {report['statistics']['passed_tests']} passed, {report['statistics']['failed_tests']} failed, {report['statistics']['error_tests']} errors")
        print(f"Pass Rate: {report['statistics']['pass_rate'] * 100:.2f}%")
        print(f"Average Coverage: {report['statistics']['average_coverage'] * 100:.2f}%")
        print(f"Has Regressions: {report['has_regressions']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
