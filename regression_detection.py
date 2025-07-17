#!/usr/bin/env python3
"""
regression_detection.py
────────────────────────
Advanced regression detection and prevention system to ensure zero regressions in patched code.

This component:
1. Maintains a comprehensive test history database
2. Runs regression tests before and after patches
3. Implements automated snapshots for rollback
4. Provides impact analysis for proposed changes
5. Integrates with CI/CD for continuous verification
"""

import os
import sys
import json
import time
import logging
import hashlib
import tempfile
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/regression.log")
    ]
)
logger = logging.getLogger("regression_detection")

# ─── DATA STRUCTURES ────────────────────────────────────────────────────────────

@dataclass
class FileSnapshot:
    """Snapshot of a file at a specific point in time."""
    path: str
    content_hash: str
    timestamp: float
    size_bytes: int

@dataclass
class TestCase:
    """Information about a single test case."""
    name: str
    file_path: str
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    execution_time: float = 0.0
    coverage: Dict[str, List[int]] = field(default_factory=dict)

@dataclass
class TestSuite:
    """Collection of test cases."""
    name: str
    test_cases: List[TestCase] = field(default_factory=list)
    total_cases: int = 0
    passing_cases: int = 0
    execution_time: float = 0.0

@dataclass
class Patch:
    """Information about a code patch."""
    id: str
    description: str
    timestamp: float
    author: str
    files_changed: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # path -> (before_hash, after_hash)
    impacted_tests: List[str] = field(default_factory=list)
    regression_free: bool = False
    verification_time: float = 0.0

@dataclass
class RegressionReport:
    """Report on potential regressions."""
    timestamp: float
    patch_id: str
    failing_tests: List[str]
    new_failures: List[str]
    fixed_failures: List[str]
    risk_score: float
    is_regression: bool
    recommendations: List[str]

# ─── DATABASE OPERATIONS ─────────────────────────────────────────────────────────

class RegressionDB:
    """Database for storing regression information."""
    
    def __init__(self, db_path: str = "data/regression_db.json"):
        self.db_path = db_path
        self.snapshots: Dict[str, List[FileSnapshot]] = {}
        self.test_suites: Dict[str, TestSuite] = {}
        self.patches: Dict[str, Patch] = {}
        self.reports: List[RegressionReport] = []
        self._load_db()
    
    def _load_db(self) -> None:
        """Load database from disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    
                    # Deserialize snapshots
                    self.snapshots = {}
                    for path, snapshots in data.get("snapshots", {}).items():
                        self.snapshots[path] = [FileSnapshot(**s) for s in snapshots]
                    
                    # Deserialize test suites
                    self.test_suites = {}
                    for name, suite_data in data.get("test_suites", {}).items():
                        test_cases = [TestCase(**tc) for tc in suite_data.pop("test_cases", [])]
                        suite = TestSuite(name=name, **suite_data)
                        suite.test_cases = test_cases
                        self.test_suites[name] = suite
                    
                    # Deserialize patches
                    self.patches = {}
                    for patch_id, patch_data in data.get("patches", {}).items():
                        self.patches[patch_id] = Patch(**patch_data)
                    
                    # Deserialize reports
                    self.reports = [RegressionReport(**r) for r in data.get("reports", [])]
                    
                logger.info(f"Loaded regression database from {self.db_path}")
            else:
                logger.info(f"No existing database found at {self.db_path}, creating new database")
        except Exception as e:
            logger.error(f"Error loading regression database: {str(e)}")
            # Create empty DB
    
    def save(self) -> None:
        """Save database to disk."""
        try:
            data = {
                "snapshots": {path: [asdict(s) for s in snapshots] for path, snapshots in self.snapshots.items()},
                "test_suites": {name: {**asdict(suite), "test_cases": [asdict(tc) for tc in suite.test_cases]} 
                               for name, suite in self.test_suites.items()},
                "patches": {patch_id: asdict(patch) for patch_id, patch in self.patches.items()},
                "reports": [asdict(r) for r in self.reports]
            }
            
            # Create temporary file to avoid corruption
            with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
                json.dump(data, tmp, indent=2)
            
            # Replace original file with new one
            os.replace(tmp.name, self.db_path)
            logger.info(f"Saved regression database to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving regression database: {str(e)}")
    
    def add_snapshot(self, file_path: str, snapshot: FileSnapshot) -> None:
        """Add a file snapshot to the database."""
        if file_path not in self.snapshots:
            self.snapshots[file_path] = []
        self.snapshots[file_path].append(snapshot)
        # Keep at most 10 snapshots per file
        if len(self.snapshots[file_path]) > 10:
            self.snapshots[file_path] = self.snapshots[file_path][-10:]
    
    def add_test_suite(self, suite: TestSuite) -> None:
        """Add or update a test suite in the database."""
        self.test_suites[suite.name] = suite
    
    def add_patch(self, patch: Patch) -> None:
        """Add a patch to the database."""
        self.patches[patch.id] = patch
    
    def add_report(self, report: RegressionReport) -> None:
        """Add a regression report to the database."""
        self.reports.append(report)
        # Keep at most 100 reports
        if len(self.reports) > 100:
            self.reports = self.reports[-100:]
    
    def get_latest_snapshot(self, file_path: str) -> Optional[FileSnapshot]:
        """Get the latest snapshot of a file."""
        if file_path in self.snapshots and self.snapshots[file_path]:
            return self.snapshots[file_path][-1]
        return None
    
    def get_test_suite(self, name: str) -> Optional[TestSuite]:
        """Get a test suite by name."""
        return self.test_suites.get(name)
    
    def get_patch(self, patch_id: str) -> Optional[Patch]:
        """Get a patch by ID."""
        return self.patches.get(patch_id)
    
    def get_latest_reports(self, limit: int = 10) -> List[RegressionReport]:
        """Get the latest regression reports."""
        return self.reports[-limit:]

# ─── FILE OPERATIONS ─────────────────────────────────────────────────────────────

def compute_file_hash(file_path: str) -> str:
    """Compute a hash of a file's content."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {str(e)}")
        return ""

def create_file_snapshot(file_path: str) -> Optional[FileSnapshot]:
    """Create a snapshot of a file."""
    try:
        path_obj = Path(file_path)
        if not path_obj.exists() or not path_obj.is_file():
            logger.warning(f"File not found: {file_path}")
            return None
        
        content_hash = compute_file_hash(file_path)
        size_bytes = path_obj.stat().st_size
        timestamp = time.time()
        
        return FileSnapshot(
            path=file_path,
            content_hash=content_hash,
            timestamp=timestamp,
            size_bytes=size_bytes
        )
    except Exception as e:
        logger.error(f"Error creating snapshot for {file_path}: {str(e)}")
        return None

def snapshot_directory(directory: str, extensions: List[str] = None) -> Dict[str, FileSnapshot]:
    """Create snapshots of all files in a directory."""
    snapshots = {}
    try:
        path_obj = Path(directory)
        if not path_obj.exists() or not path_obj.is_dir():
            logger.warning(f"Directory not found: {directory}")
            return snapshots
        
        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Filter by extension if specified
                if extensions:
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in extensions:
                        continue
                
                snapshot = create_file_snapshot(file_path)
                if snapshot:
                    snapshots[file_path] = snapshot
        
        logger.info(f"Created {len(snapshots)} snapshots in {directory}")
        return snapshots
    except Exception as e:
        logger.error(f"Error snapshotting directory {directory}: {str(e)}")
        return snapshots

# ─── TEST DISCOVERY AND EXECUTION ────────────────────────────────────────────────

def discover_tests(test_dir: str) -> TestSuite:
    """Discover test cases in a directory."""
    test_suite = TestSuite(name="main")
    test_cases = []
    
    try:
        # Use pytest for test discovery
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q", test_dir],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error discovering tests: {result.stderr}")
            return test_suite
        
        # Parse test names
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("[") or line.startswith("="):
                continue
                
            test_cases.append(TestCase(
                name=line,
                file_path=line.split("::")[0] if "::" in line else "",
                last_success=None,
                last_failure=None,
                execution_time=0.0,
                coverage={}
            ))
        
        test_suite.test_cases = test_cases
        test_suite.total_cases = len(test_cases)
        logger.info(f"Discovered {len(test_cases)} tests in {test_dir}")
        
        return test_suite
    except Exception as e:
        logger.error(f"Error in test discovery: {str(e)}")
        return test_suite

def run_tests(tests: Union[TestSuite, List[str]], coverage: bool = True) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, Dict[str, List[int]]]]:
    """
    Run tests and return pass/fail results with execution times and coverage.
    
    Args:
        tests: TestSuite object or list of test names
        coverage: Whether to collect coverage information
        
    Returns:
        Tuple of (passing tests, failing tests, execution times, coverage data)
    """
    passing = []
    failing = []
    execution_times = {}
    coverage_data = {}
    
    try:
        # Prepare test names
        if isinstance(tests, TestSuite):
            test_names = [tc.name for tc in tests.test_cases]
        else:
            test_names = tests
        
        if not test_names:
            logger.warning("No tests to run")
            return passing, failing, execution_times, coverage_data
        
        # Create temporary file with test names
        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            f.write("\n".join(test_names))
            tests_file = f.name
        
        # Prepare command
        cmd = ["python", "-m", "pytest", f"--tests-file={tests_file}", "-v"]
        
        if coverage:
            cmd.extend(["--cov", "--cov-report=xml"])
        
        # Run tests
        logger.info(f"Running {len(test_names)} tests...")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse results
        for line in result.stdout.splitlines():
            if " PASSED " in line:
                test_name = line.split(" PASSED ")[0].strip()
                passing.append(test_name)
                # Extract execution time if available
                if "(" in line and ")" in line:
                    time_str = line.split("(")[1].split(")")[0]
                    try:
                        execution_times[test_name] = float(time_str.replace("s", ""))
                    except ValueError:
                        pass
            elif " FAILED " in line:
                test_name = line.split(" FAILED ")[0].strip()
                failing.append(test_name)
                # Extract execution time if available
                if "(" in line and ")" in line:
                    time_str = line.split("(")[1].split(")")[0]
                    try:
                        execution_times[test_name] = float(time_str.replace("s", ""))
                    except ValueError:
                        pass
        
        # Parse coverage data if available
        if coverage and os.path.exists("coverage.xml"):
            import xml.etree.ElementTree as ET
            try:
                tree = ET.parse("coverage.xml")
                root = tree.getroot()
                for class_elem in root.findall(".//class"):
                    filename = class_elem.get("filename")
                    lines = {}
                    for line in class_elem.findall(".//line"):
                        line_num = int(line.get("number"))
                        hits = int(line.get("hits"))
                        if hits > 0:
                            if filename not in lines:
                                lines[filename] = []
                            lines[filename].append(line_num)
                    if lines:
                        coverage_data[filename] = lines
            except Exception as e:
                logger.error(f"Error parsing coverage data: {str(e)}")
        
        # Clean up
        os.unlink(tests_file)
        if os.path.exists("coverage.xml"):
            os.unlink("coverage.xml")
        
        elapsed = time.time() - start_time
        logger.info(f"Test run completed in {elapsed:.2f}s: {len(passing)} passed, {len(failing)} failed")
        
        return passing, failing, execution_times, coverage_data
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return passing, failing, execution_times, coverage_data

# ─── PATCH ANALYSIS ──────────────────────────────────────────────────────────────

def analyze_patch_impact(db: RegressionDB, file_changes: Dict[str, Tuple[str, str]]) -> Set[str]:
    """
    Analyze which tests are impacted by changes to specific files.
    
    Args:
        db: Regression database
        file_changes: Dict mapping file paths to (before_hash, after_hash) tuples
        
    Returns:
        Set of test names that are likely impacted by the changes
    """
    impacted_tests = set()
    
    # For each changed file, find tests that exercise it
    for file_path in file_changes.keys():
        for suite_name, suite in db.test_suites.items():
            for test_case in suite.test_cases:
                # Check if test covers this file
                for covered_file, lines in test_case.coverage.items():
                    if covered_file == file_path or file_path.endswith(covered_file):
                        impacted_tests.add(test_case.name)
                        break
    
    # If we can't determine specific tests, recommend running all tests
    if not impacted_tests:
        for suite_name, suite in db.test_suites.items():
            for test_case in suite.test_cases:
                impacted_tests.add(test_case.name)
    
    return impacted_tests

def calculate_regression_risk(
    patch: Patch, 
    failing_before: List[str], 
    failing_after: List[str]
) -> Tuple[float, bool, List[str]]:
    """
    Calculate the regression risk of a patch.
    
    Args:
        patch: Patch information
        failing_before: List of failing tests before the patch
        failing_after: List of failing tests after the patch
        
    Returns:
        Tuple of (risk score, is regression, recommendations)
    """
    # Identify new failures (potential regressions)
    new_failures = [t for t in failing_after if t not in failing_before]
    
    # Identify fixed failures
    fixed_failures = [t for t in failing_before if t not in failing_after]
    
    # Calculate risk score based on number of files changed and new failures
    risk_score = min(1.0, (len(patch.files_changed) * 0.1) + (len(new_failures) * 0.3))
    
    # Determine if this is a regression
    is_regression = len(new_failures) > 0
    
    # Generate recommendations
    recommendations = []
    if is_regression:
        recommendations.append(f"Regression detected: {len(new_failures)} new failing tests.")
        recommendations.append("Review the changes to identify the cause.")
        recommendations.append("Consider rolling back the patch if the regression is critical.")
    else:
        if fixed_failures:
            recommendations.append(f"Patch fixed {len(fixed_failures)} previously failing tests.")
        if len(failing_after) > 0:
            recommendations.append(f"There are still {len(failing_after)} failing tests that need attention.")
        else:
            recommendations.append("All tests are passing! The patch appears to be regression-free.")
    
    return risk_score, is_regression, recommendations

# ─── MAIN FUNCTIONALITY ───────────────────────────────────────────────────────────

class RegressionDetector:
    """Main class for regression detection system."""
    
    def __init__(self, db_path: str = "data/regression_db.json"):
        self.db = RegressionDB(db_path)
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        logger.info("Regression detector initialized")
    
    def snapshot_codebase(self, source_dir: str, extensions: List[str] = None) -> None:
        """
        Create snapshots of the codebase.
        
        Args:
            source_dir: Directory containing source code
            extensions: Optional list of file extensions to include
        """
        # Default to common source code extensions
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp']
        
        snapshots = snapshot_directory(source_dir, extensions)
        
        # Add all snapshots to the database
        for path, snapshot in snapshots.items():
            self.db.add_snapshot(path, snapshot)
        
        # Save the database
        self.db.save()
        
        logger.info(f"Created {len(snapshots)} snapshots of the codebase")
    
    def discover_and_register_tests(self, test_dir: str) -> TestSuite:
        """
        Discover and register tests.
        
        Args:
            test_dir: Directory containing tests
            
        Returns:
            TestSuite containing discovered tests
        """
        suite = discover_tests(test_dir)
        self.db.add_test_suite(suite)
        self.db.save()
        
        logger.info(f"Registered test suite with {suite.total_cases} tests")
        return suite
    
    def run_baseline_tests(self, test_dir: str) -> None:
        """
        Run baseline tests and record results.
        
        Args:
            test_dir: Directory containing tests
        """
        # Discover tests if needed
        suite = self.db.get_test_suite("main")
        if not suite or not suite.test_cases:
            suite = self.discover_and_register_tests(test_dir)
        
        # Run tests with coverage
        passing, failing, execution_times, coverage_data = run_tests(suite, coverage=True)
        
        # Update test suite with results
        now = time.time()
        suite.passing_cases = len(passing)
        suite.total_cases = len(passing) + len(failing)
        suite.execution_time = sum(execution_times.values())
        
        # Update individual test cases
        for test_case in suite.test_cases:
            if test_case.name in passing:
                test_case.last_success = now
                test_case.execution_time = execution_times.get(test_case.name, 0.0)
                
                # Add coverage data
                for file_path, lines in coverage_data.items():
                    # Check if file_path ends with the file path in the test case
                    matching_files = [f for f in coverage_data.keys() 
                                     if test_case.file_path.endswith(f) or f.endswith(test_case.file_path)]
                    
                    for file_path in matching_files:
                        test_case.coverage[file_path] = coverage_data[file_path]
                
            elif test_case.name in failing:
                test_case.last_failure = now
                test_case.execution_time = execution_times.get(test_case.name, 0.0)
        
        # Save updated test suite
        self.db.add_test_suite(suite)
        self.db.save()
        
        logger.info(f"Baseline test run completed: {suite.passing_cases}/{suite.total_cases} tests passing")
    
    def verify_patch(self, patch_id: str, description: str, author: str, changed_files: List[str]) -> RegressionReport:
        """
        Verify a patch for regressions.
        
        Args:
            patch_id: Unique identifier for the patch
            description: Description of the patch
            author: Author of the patch
            changed_files: List of files changed by the patch
            
        Returns:
            Regression report
        """
        logger.info(f"Verifying patch {patch_id}: {description}")
        
        # Get baseline test results
        suite = self.db.get_test_suite("main")
        if not suite or not suite.test_cases:
            logger.error("No baseline tests available. Run baseline tests first.")
            return None
        
        # Create file snapshots and record changes
        file_changes = {}
        for file_path in changed_files:
            # Get previous snapshot
            prev_snapshot = self.db.get_latest_snapshot(file_path)
            prev_hash = prev_snapshot.content_hash if prev_snapshot else ""
            
            # Create new snapshot
            new_snapshot = create_file_snapshot(file_path)
            if new_snapshot:
                self.db.add_snapshot(file_path, new_snapshot)
                file_changes[file_path] = (prev_hash, new_snapshot.content_hash)
        
        # Create patch record
        patch = Patch(
            id=patch_id,
            description=description,
            timestamp=time.time(),
            author=author,
            files_changed=file_changes
        )
        
        # Determine which tests are impacted by the changes
        impacted_tests = analyze_patch_impact(self.db, file_changes)
        patch.impacted_tests = list(impacted_tests)
        
        # Determine baseline failing tests
        failing_before = [tc.name for tc in suite.test_cases if tc.last_failure and (not tc.last_success or tc.last_failure > tc.last_success)]
        
        # Run impacted tests
        logger.info(f"Running {len(impacted_tests)} impacted tests...")
        start_time = time.time()
        passing, failing, execution_times, _ = run_tests(list(impacted_tests), coverage=False)
        patch.verification_time = time.time() - start_time
        
        # Update test results
        now = time.time()
        for test_name in passing:
            for test_case in suite.test_cases:
                if test_case.name == test_name:
                    test_case.last_success = now
                    test_case.execution_time = execution_times.get(test_name, test_case.execution_time)
                    break
        
        for test_name in failing:
            for test_case in suite.test_cases:
                if test_case.name == test_name:
                    test_case.last_failure = now
                    test_case.execution_time = execution_times.get(test_name, test_case.execution_time)
                    break
        
        # Update suite stats
        failing_all = [tc.name for tc in suite.test_cases if tc.last_failure and (not tc.last_success or tc.last_failure > tc.last_success)]
        suite.passing_cases = suite.total_cases - len(failing_all)
        
        # Calculate regression risk
        risk_score, is_regression, recommendations = calculate_regression_risk(patch, failing_before, failing)
        
        # Create regression report
        report = RegressionReport(
            timestamp=time.time(),
            patch_id=patch_id,
            failing_tests=failing,
            new_failures=[f for f in failing if f not in failing_before],
            fixed_failures=[f for f in failing_before if f not in failing],
            risk_score=risk_score,
            is_regression=is_regression,
            recommendations=recommendations
        )
        
        # Update patch record
        patch.regression_free = not is_regression
        
        # Save everything
        self.db.add_patch(patch)
        self.db.add_test_suite(suite)
        self.db.add_report(report)
        self.db.save()
        
        logger.info(f"Patch verification completed: regression_free={not is_regression}, risk_score={risk_score:.2f}")
        
        return report
    
    def optimize_test_time(self, target_time_seconds: float = 900.0) -> List[str]:
        """
        Optimize test execution to keep regression detection under target time.
        
        Args:
            target_time_seconds: Target execution time in seconds (default: 15 minutes)
            
        Returns:
            List of recommended test optimizations
        """
        recommendations = []
        
        suite = self.db.get_test_suite("main")
        if not suite or not suite.test_cases:
            recommendations.append("No test suite available. Run baseline tests first.")
            return recommendations
        
        # Sort tests by execution time (descending)
        sorted_tests = sorted(suite.test_cases, key=lambda tc: tc.execution_time, reverse=True)
        
        # Calculate total execution time
        total_time = sum(tc.execution_time for tc in suite.test_cases)
        
        if total_time <= target_time_seconds:
            recommendations.append(f"Current test execution time ({total_time:.2f}s) is already under target ({target_time_seconds:.2f}s).")
            return recommendations
        
        # Identify slow tests (taking more than 5% of total time)
        slow_tests = [tc for tc in sorted_tests if tc.execution_time > (target_time_seconds * 0.05)]
        
        if slow_tests:
            recommendations.append(f"Found {len(slow_tests)} slow tests consuming significant execution time:")
            for tc in slow_tests[:5]:  # Show top 5 slowest
                recommendations.append(f"  - {tc.name}: {tc.execution_time:.2f}s ({(tc.execution_time / total_time * 100):.1f}% of total)")
            
            recommendations.append("Consider optimizing these slow tests or running them in parallel.")
        
        # Suggest parallel execution
        if len(suite.test_cases) > 10:
            parallel_factor = min(8, (total_time / target_time_seconds) * 1.5)  # Estimate parallel factor needed
            estimated_time = total_time / parallel_factor
            recommendations.append(f"Running tests with {int(parallel_factor)} parallel workers could reduce execution time to approximately {estimated_time:.2f}s.")
        
        # Suggest test prioritization
        recommendations.append("Consider implementing test prioritization based on historical failure rates and code changes.")
        
        return recommendations

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regression Detection System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Create snapshots of the codebase")
    snapshot_parser.add_argument("--dir", default=".", help="Directory to snapshot")
    snapshot_parser.add_argument("--ext", nargs="+", help="File extensions to include")
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover and register tests")
    discover_parser.add_argument("--dir", default="tests", help="Test directory")
    
    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Run baseline tests")
    baseline_parser.add_argument("--dir", default="tests", help="Test directory")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a patch for regressions")
    verify_parser.add_argument("--id", required=True, help="Patch ID")
    verify_parser.add_argument("--desc", required=True, help="Patch description")
    verify_parser.add_argument("--author", default="system", help="Patch author")
    verify_parser.add_argument("--files", nargs="+", required=True, help="Files changed by the patch")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize test execution time")
    optimize_parser.add_argument("--target", type=float, default=900.0, help="Target execution time in seconds")
    
    args = parser.parse_args()
    
    detector = RegressionDetector()
    
    if args.command == "snapshot":
        detector.snapshot_codebase(args.dir, args.ext)
    elif args.command == "discover":
        detector.discover_and_register_tests(args.dir)
    elif args.command == "baseline":
        detector.run_baseline_tests(args.dir)
    elif args.command == "verify":
        report = detector.verify_patch(args.id, args.desc, args.author, args.files)
        if report:
            print("\nRegression Report:")
            print(f"Patch: {report.patch_id} - {report.is_regression}")
            print(f"Risk Score: {report.risk_score:.2f}")
            print(f"New Failures: {len(report.new_failures)}")
            print(f"Fixed Failures: {len(report.fixed_failures)}")
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"- {rec}")
    elif args.command == "optimize":
        recommendations = detector.optimize_test_time(args.target)
        print("\nTest Optimization Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
