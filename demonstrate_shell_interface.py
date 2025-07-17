#!/usr/bin/env python3
"""
FixWurx Auditor Shell Interface Demonstration

This script initializes the sensor system, registers some example sensors,
and launches the shell interface to demonstrate its management capabilities.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List, Optional

# Create a simplified mock LLM bridge for the demonstration
class MockLLMBridge:
    """Mock implementation of SensorLLMBridge for demonstration purposes."""
    
    def __init__(self):
        """Initialize without registry requirement."""
        self.queries = []
    
    def query(self, question: str) -> str:
        """Mock query method."""
        self.queries.append(question)
        return "No, I cannot perform that operation due to system limitations."
    
    def ask(self, question: str) -> str:
        """Mock ask method."""
        return self.query(question)

# Make the mock bridge available
import sys
import types
mock_llm_module = types.ModuleType('mock_llm_bridge')
mock_llm_module.MockLLMBridge = MockLLMBridge
mock_llm_module.llm_bridge = MockLLMBridge()
sys.modules['mock_llm_bridge'] = mock_llm_module

# Import core components
from sensor_registry import SensorRegistry, create_sensor_registry
from sensor_manager import SensorManager
from error_report import ErrorReport
from benchmark_storage import BenchmarkStorage

# Import sensors
from obligation_ledger_sensor import ObligationLedgerSensor
from graph_database_sensor import GraphDatabaseSensor
from meta_awareness_sensor import MetaAwarenessSensor
from performance_benchmark_sensor import PerformanceBenchmarkSensor

# Import shell interface
from auditor_shell_interface import AuditorShellInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ShellDemo] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('shell_demo')


def setup_sensor_system() -> tuple:
    """
    Set up the sensor system with example sensors.
    
    Returns:
        Tuple of (SensorRegistry, SensorManager, BenchmarkStorage)
    """
    # Create directories for benchmark storage
    os.makedirs("auditor_data/benchmarks", exist_ok=True)
    os.makedirs("auditor_data/exports", exist_ok=True)
    
    # Create sensor registry
    registry = create_sensor_registry()
    
    # Create sensor manager
    manager = SensorManager(registry)
    
    # Create benchmark storage
    storage = BenchmarkStorage("auditor_data/benchmarks")
    
    # Create and register example sensors
    sensors = [
        ObligationLedgerSensor(
            component_name="ObligationTracker",
            config={
                "check_interval": 60,
                "max_obligation_age": 3600
            }
        ),
        GraphDatabaseSensor(
            component_name="GraphDatabase",
            config={
                "performance_check_interval": 30,
                "max_query_time_ms": 300
            }
        ),
        # Modified to use the mock LLM bridge
        MetaAwarenessSensor(
            component_name="AuditorSelfAwareness",
            config={
                "consistency_check_interval": 120,
                "min_consistency_score": 0.8,
                "use_mock_llm": True  # Signal to use mock
            }
        ),
        PerformanceBenchmarkSensor(
            config={
                "thresholds": {
                    "bug_detection_recall": 0.7,
                    "bug_fix_yield": 0.6,
                    "test_pass_ratio": 0.9
                }
            }
        )
    ]
    
    # Register all sensors
    for sensor in sensors:
        registry.register_sensor(sensor)
        logger.info(f"Registered sensor: {sensor.sensor_id}")
    
    # Set up some example projects and sessions
    setup_example_data(storage)
    
    return registry, manager, storage


def setup_example_data(storage: BenchmarkStorage) -> None:
    """
    Set up example data in the benchmark storage.
    
    Args:
        storage: BenchmarkStorage instance
    """
    # Create some example projects
    projects = ["WebAuthentication", "DataProcessing", "APIIntegration"]
    
    for project in projects:
        # Create a session for each project
        session_id = storage.create_session(
            project_name=project,
            metadata={
                "description": f"Example session for {project}",
                "created_by": "demonstrate_shell_interface.py"
            }
        )
        logger.info(f"Created session {session_id} for project {project}")
        
        # Add some sample metrics
        metrics = {
            "bug_detection_recall": 0.65 + (hash(project) % 10) / 50,
            "bug_fix_yield": 0.72 + (hash(project) % 10) / 40,
            "test_pass_ratio": 0.88 + (hash(project) % 10) / 100,
            "energy_reduction_pct": 0.25 + (hash(project) % 10) / 80,
            "mttd": 4.5 + (hash(project) % 10) / 5,
            "mttr": 12.3 - (hash(project) % 10) / 10,
            "aggregate_confidence_score": 0.77 + (hash(project) % 10) / 50
        }
        
        # Store metrics
        storage.store_metrics(session_id, metrics)
        
        # Add a second set of metrics with some improvements
        improved_metrics = metrics.copy()
        improved_metrics["bug_detection_recall"] += 0.1
        improved_metrics["bug_fix_yield"] += 0.08
        improved_metrics["test_pass_ratio"] += 0.05
        improved_metrics["energy_reduction_pct"] += 0.15
        improved_metrics["mttd"] -= 0.5
        improved_metrics["mttr"] -= 1.2
        improved_metrics["aggregate_confidence_score"] += 0.07
        
        # Store improved metrics
        time.sleep(0.5)  # Small delay for timestamp difference
        storage.store_metrics(session_id, improved_metrics)
        
        # Create some example error reports
        if project == "APIIntegration":
            report = ErrorReport(
                error_type="API_INTEGRATION_FAILURE",
                severity="HIGH",
                component_name="APIIntegration",
                sensor_id="integration_sensor",
                details={
                    "message": "Failed to connect to external API endpoint",
                    "endpoint": "https://api.example.com/v2/data",
                    "status_code": 503
                },
                context={
                    "request_headers": {"Accept": "application/json"},
                    "response_body": "Service Unavailable"
                }
            )
            storage.store_error_report(session_id, report.to_dict())
            logger.info(f"Added example error report to session {session_id}")


def generate_shell_commands() -> str:
    """
    Generate example shell commands to demonstrate in the help text.
    
    Returns:
        String of example commands
    """
    return """
Example commands to try:
  status                          - Show system status
  sensor list                     - List registered sensors (sensor/sensors both work)
  project list                    - List benchmark projects
  1, 2, 3...                      - Select project by number
  session list                    - List sessions for current project
  metrics list                    - Show metrics for current session
  error list                      - List error reports (error/errors both work)
  visualize summary               - Show system health summary
  visualize compare bug_fix_yield - Compare metric across projects
  help                            - Show available commands
  exit                            - Exit the shell
"""


def main():
    """Main demonstration function."""
    print("\n" + "="*70)
    print("AUDITOR SHELL INTERFACE DEMONSTRATION")
    print("="*70 + "\n")
    
    print("This demonstration shows how to use the shell interface")
    print("to manage sensors, analyze benchmark metrics, and handle errors.")
    
    # Set up the sensor system
    logger.info("Setting up sensor system...")
    registry, manager, storage = setup_sensor_system()
    
    # Create the shell interface
    shell = AuditorShellInterface(
        sensor_registry=registry,
        sensor_manager=manager,
        benchmark_storage=storage,
        config={
            "default_project": "WebAuthentication",
            "refresh_interval": 5  # Short interval for demo
        }
    )
    
    # Show usage instructions
    print("\nSensor system initialized with example sensors and data.")
    print(generate_shell_commands())
    
    # Launch the shell
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt.")
    except Exception as e:
        logger.error(f"Error in shell interface: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
