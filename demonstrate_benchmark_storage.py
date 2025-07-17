#!/usr/bin/env python3
"""
FixWurx Auditor Benchmark Storage Demonstration

This script demonstrates how benchmark metrics are stored and organized by
session ID and project folder. It shows the complete flow from benchmark
collection to storage, retrieval, and analysis.
"""

import os
import json
import datetime
import logging
import time
import random
from typing import Dict, Any

# Import sensor components
from performance_benchmark_sensor import PerformanceBenchmarkSensor
from benchmark_storage import BenchmarkStorage, integrate_with_benchmark_sensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [BenchmarkStorageDemo] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('benchmark_storage_demo')


class SimulatedProject:
    """Simulate a project with debugging sessions for demonstration."""
    
    def __init__(self, name: str, complexity: float = 0.5):
        """
        Initialize a simulated project.
        
        Args:
            name: Project name
            complexity: Project complexity (0-1)
        """
        self.name = name
        self.complexity = complexity
        self.bugs_detected = 0
        self.bugs_fixed = 0
        self.total_bugs = int(20 * complexity)
        self.tests_total = int(50 * complexity)
        self.tests_passed = int(self.tests_total * (1 - 0.3 * complexity))
        
    def simulate_debugging_progress(self, iterations: int = 5) -> Dict[str, Any]:
        """
        Simulate debugging progress over multiple iterations.
        
        Args:
            iterations: Number of iterations
            
        Returns:
            Dictionary of metrics
        """
        results = []
        
        for i in range(iterations):
            # Detect new bugs
            new_bugs = random.randint(0, 2)
            self.bugs_detected += new_bugs
            self.bugs_detected = min(self.bugs_detected, self.total_bugs)
            
            # Fix bugs
            if self.bugs_detected > self.bugs_fixed:
                fixes = random.randint(0, self.bugs_detected - self.bugs_fixed)
                self.bugs_fixed += fixes
            
            # Improve test coverage
            new_passing = random.randint(0, 2)
            self.tests_passed += new_passing
            self.tests_passed = min(self.tests_passed, self.tests_total)
            
            # Calculate metrics
            metrics = {
                "bug_detection_recall": self.bugs_detected / max(1, self.total_bugs),
                "bug_fix_yield": self.bugs_fixed / max(1, self.bugs_detected),
                "test_pass_ratio": self.tests_passed / max(1, self.tests_total),
                "energy_reduction_pct": (self.bugs_fixed / max(1, self.total_bugs)) * 0.8,
                "mttd": 5 + (10 * self.complexity),
                "mttr": 8 + (15 * self.complexity),
                "iteration": i + 1,
                "bugs_detected": self.bugs_detected,
                "bugs_fixed": self.bugs_fixed,
                "tests_passed": self.tests_passed,
                "tests_total": self.tests_total
            }
            
            results.append(metrics)
            
        return results


def setup_benchmark_storage() -> BenchmarkStorage:
    """
    Set up the benchmark storage system.
    
    Returns:
        BenchmarkStorage instance
    """
    # Create benchmark storage
    storage = BenchmarkStorage("auditor_data/benchmarks")
    
    # Clean up existing data for demonstration purposes
    if os.path.exists("auditor_data/benchmarks/sessions_index.json"):
        os.remove("auditor_data/benchmarks/sessions_index.json")
    
    return storage


def demonstrate_single_project(storage: BenchmarkStorage, 
                             project: SimulatedProject) -> str:
    """
    Demonstrate benchmark storage for a single project.
    
    Args:
        storage: BenchmarkStorage instance
        project: SimulatedProject instance
        
    Returns:
        Session ID
    """
    print(f"\n=== Demonstrating Benchmark Storage for Project: {project.name} ===\n")
    
    # Create benchmark sensor
    sensor = PerformanceBenchmarkSensor(config={
        "thresholds": {
            "bug_detection_recall": 0.7,
            "bug_fix_yield": 0.6,
            "test_pass_ratio": 0.9
        }
    })
    
    # Create session and integrate with sensor
    session_id = integrate_with_benchmark_sensor(
        storage, 
        sensor, 
        project.name,
        metadata={
            "description": f"Debugging session for {project.name}",
            "complexity": project.complexity,
            "total_bugs": project.total_bugs
        }
    )
    
    print(f"Created benchmark session: {session_id}")
    print(f"Session data stored in: auditor_data/benchmarks/{project.name}/{session_id}/")
    print(f"  - Metrics stored in: auditor_data/benchmarks/{project.name}/{session_id}/metrics/")
    print(f"  - Reports stored in: auditor_data/benchmarks/{project.name}/{session_id}/reports/")
    
    # Simulate debugging progress and store metrics
    progress = project.simulate_debugging_progress(iterations=5)
    
    for i, metrics in enumerate(progress):
        print(f"\nIteration {i+1}/{len(progress)}")
        print("-" * 50)
        
        # Update sensor with metrics
        for key, value in metrics.items():
            if key in sensor.session_metrics:
                sensor.session_metrics[key] = value
        
        # Manually calculate derived metrics
        sensor._calculate_derived_metrics()
        
        # Store metrics
        storage.store_metrics(session_id, sensor.session_metrics)
        
        # Check for issues and store reports
        reports = sensor._check_detection_fix_metrics()
        for report in reports:
            storage.store_error_report(session_id, report.to_dict())
            print(f"  Stored error report: {report.error_type} - {report.severity}")
        
        # Show some metrics
        print(f"  Bug Detection Recall: {metrics['bug_detection_recall']:.2f}")
        print(f"  Bug Fix Yield: {metrics['bug_fix_yield']:.2f}")
        print(f"  Test Pass Ratio: {metrics['test_pass_ratio']:.2f}")
        
        # Pause between iterations
        time.sleep(0.5)
    
    print(f"\nCompleted benchmark session for {project.name}")
    
    return session_id


def demonstrate_multiple_projects(storage: BenchmarkStorage) -> Dict[str, str]:
    """
    Demonstrate benchmark storage for multiple projects.
    
    Args:
        storage: BenchmarkStorage instance
        
    Returns:
        Dictionary of project names to session IDs
    """
    print("\n=== Demonstrating Benchmark Storage for Multiple Projects ===\n")
    
    # Create projects with varying complexity
    projects = [
        SimulatedProject("Authentication", 0.3),
        SimulatedProject("DataProcessing", 0.7),
        SimulatedProject("UserInterface", 0.5)
    ]
    
    sessions = {}
    
    for project in projects:
        session_id = demonstrate_single_project(storage, project)
        sessions[project.name] = session_id
    
    return sessions


def demonstrate_data_retrieval(storage: BenchmarkStorage, 
                             project_sessions: Dict[str, str]) -> None:
    """
    Demonstrate retrieving benchmark data.
    
    Args:
        storage: BenchmarkStorage instance
        project_sessions: Dictionary of project names to session IDs
    """
    print("\n=== Demonstrating Benchmark Data Retrieval ===\n")
    
    # List all projects
    projects = storage.get_all_projects()
    print(f"Available projects: {', '.join(projects)}")
    
    # Show sessions for each project
    for project in projects:
        sessions = storage.get_project_sessions(project)
        print(f"\nSessions for project {project}:")
        for session in sessions:
            print(f"  - {session}")
    
    # Get metrics for a session
    for project, session_id in project_sessions.items():
        print(f"\nMetrics for project {project}, session {session_id}:")
        metrics_list = storage.get_session_metrics(session_id)
        print(f"  Found {len(metrics_list)} metric snapshots")
        
        if metrics_list:
            # Show first and last metrics
            first_metrics = metrics_list[0]["metrics"]
            last_metrics = metrics_list[-1]["metrics"]
            
            print("  Initial metrics:")
            print(f"    Bug Detection Recall: {first_metrics.get('bug_detection_recall', 0):.2f}")
            print(f"    Bug Fix Yield: {first_metrics.get('bug_fix_yield', 0):.2f}")
            print(f"    Test Pass Ratio: {first_metrics.get('test_pass_ratio', 0):.2f}")
            
            print("  Final metrics:")
            print(f"    Bug Detection Recall: {last_metrics.get('bug_detection_recall', 0):.2f}")
            print(f"    Bug Fix Yield: {last_metrics.get('bug_fix_yield', 0):.2f}")
            print(f"    Test Pass Ratio: {last_metrics.get('test_pass_ratio', 0):.2f}")
        
        # Get reports for the session
        reports = storage.get_session_reports(session_id)
        print(f"  Found {len(reports)} error reports")
        
        if reports:
            print("  Sample error reports:")
            for i, report_data in enumerate(reports[:3]):  # Show first 3
                report = report_data["report"]
                print(f"    {i+1}. {report.get('error_type', 'Unknown')} - "
                     f"{report.get('severity', 'Unknown')}")
                if "details" in report:
                    print(f"       {report['details'].get('message', 'No message')}")


def demonstrate_data_analysis(storage: BenchmarkStorage, 
                            project_sessions: Dict[str, str]) -> None:
    """
    Demonstrate analyzing benchmark data.
    
    Args:
        storage: BenchmarkStorage instance
        project_sessions: Dictionary of project names to session IDs
    """
    print("\n=== Demonstrating Benchmark Data Analysis ===\n")
    
    # Generate session summaries
    for project, session_id in project_sessions.items():
        print(f"\nSummary for project {project}, session {session_id}:")
        
        summary = storage.generate_session_summary(session_id)
        
        print(f"  Project: {summary['project_name']}")
        print(f"  Session: {summary['session_id']}")
        print(f"  Created: {summary['created_at']}")
        print(f"  Metrics snapshots: {summary['metrics_count']}")
        print(f"  Error reports: {summary['reports_count']}")
        
        if "key_metrics" in summary and summary["key_metrics"]:
            print("  Key metrics improvement:")
            for metric, values in summary["key_metrics"].items():
                change = values["final"] - values["initial"]
                change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
                print(f"    {metric}: {values['initial']:.2f} → {values['final']:.2f} ({change_str})")
    
    # Compare projects
    print("\nProject Comparison:")
    project_metrics = {}
    
    for project, session_id in project_sessions.items():
        summary = storage.generate_session_summary(session_id)
        if "last_metrics" in summary and summary["last_metrics"]:
            metrics = summary["last_metrics"]
            project_metrics[project] = {
                "bug_detection_recall": metrics.get("bug_detection_recall", 0),
                "bug_fix_yield": metrics.get("bug_fix_yield", 0),
                "test_pass_ratio": metrics.get("test_pass_ratio", 0)
            }
    
    # Print comparison table
    if project_metrics:
        metrics_to_compare = ["bug_detection_recall", "bug_fix_yield", "test_pass_ratio"]
        
        # Print header
        print(f"{'Project':<20}", end="")
        for metric in metrics_to_compare:
            print(f"{metric:<20}", end="")
        print()
        
        # Print separator
        print("-" * (20 + 20 * len(metrics_to_compare)))
        
        # Print data
        for project, metrics in project_metrics.items():
            print(f"{project:<20}", end="")
            for metric in metrics_to_compare:
                print(f"{metrics.get(metric, 0):.2f}{'':<16}", end="")
            print()


def demonstrate_data_export(storage: BenchmarkStorage, 
                          project_sessions: Dict[str, str]) -> None:
    """
    Demonstrate exporting benchmark data.
    
    Args:
        storage: BenchmarkStorage instance
        project_sessions: Dictionary of project names to session IDs
    """
    print("\n=== Demonstrating Benchmark Data Export ===\n")
    
    # Export each session
    for project, session_id in project_sessions.items():
        export_path = f"auditor_data/exports/{project}_{session_id}"
        
        print(f"Exporting data for project {project}, session {session_id}:")
        print(f"  Export path: {export_path}")
        
        success = storage.export_session_data(session_id, export_path)
        
        if success:
            print("  Export successful")
            print(f"  Files exported:")
            print(f"    - {export_path}/session_metadata.json")
            print(f"    - {export_path}/metrics/*.json")
            print(f"    - {export_path}/reports/*.json")
        else:
            print("  Export failed")


def main():
    """Main demonstration function."""
    print("\n" + "="*60)
    print("AUDITOR BENCHMARK STORAGE DEMONSTRATION")
    print("="*60 + "\n")
    
    print("This demonstration shows how benchmark metrics are stored")
    print("organized by session ID and project folder.")
    
    # Set up benchmark storage
    storage = setup_benchmark_storage()
    
    # Create export directory
    os.makedirs("auditor_data/exports", exist_ok=True)
    
    # Demonstrate with multiple projects
    project_sessions = demonstrate_multiple_projects(storage)
    
    # Demonstrate data retrieval
    demonstrate_data_retrieval(storage, project_sessions)
    
    # Demonstrate data analysis
    demonstrate_data_analysis(storage, project_sessions)
    
    # Demonstrate data export
    demonstrate_data_export(storage, project_sessions)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60 + "\n")
    
    print("The benchmark storage system provides persistent storage for")
    print("performance metrics, organized by session ID and project.")
    print("This ensures that benchmark data is properly stored, retrievable,")
    print("and can be analyzed over time to track progress and improvements.")
    
    return 0


if __name__ == "__main__":
    exit(main())
