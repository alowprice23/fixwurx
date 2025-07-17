#!/usr/bin/env python3
"""
FixWurx Auditor Self-Awareness Demonstration

This script demonstrates the auditor's self-awareness capabilities through
the integrated sensor system. It shows how the auditor can detect issues in
its own components, generate natural language explanations, and provide
introspection about its internal state.
"""

import os
import logging
import json
import datetime
import time
from typing import Dict, Any, List

# Import sensor components
from error_report import ErrorReport
from sensor_base import ErrorSensor
from sensor_registry import SensorRegistry, create_sensor_registry
from sensor_manager import SensorManager
from obligation_ledger_sensor import ObligationLedgerSensor
from sensor_llm_bridge import SensorLLMBridge

# Import auditor components
from auditor import Auditor, ObligationLedger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [SelfAwarenessDemonstration] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('auditor_self_awareness')


class DemonstrationObligationLedger(ObligationLedger):
    """A demo version of ObligationLedger that can be manipulated to show errors."""
    
    def __init__(self):
        self.obligations = set()
        self.delta_rules = []
        self.simulated_errors = []
    
    def get_all(self):
        return self.obligations
    
    def clear(self):
        """Clear all obligations to trigger an error."""
        self.obligations = set()
    
    def add_circular_dependency(self):
        """Add circular dependencies to trigger an error."""
        self.delta_rules = [
            {"pattern": "obligation1", "transforms_to": ["obligation2"]},
            {"pattern": "obligation2", "transforms_to": ["obligation1"]}
        ]
    
    def populate_valid_data(self):
        """Populate with valid data."""
        self.obligations = {"obligation1", "obligation2", "obligation3"}
        self.delta_rules = [
            {"pattern": "obligation", "transforms_to": ["transformed1"]},
            {"pattern": "transformed1", "transforms_to": ["transformed2"]}
        ]


def create_sensor_system():
    """Create and configure the sensor system."""
    # Create directories for sensor data
    os.makedirs("auditor_data/sensors", exist_ok=True)
    
    # Create registry and manager
    registry = create_sensor_registry({
        "sensors": {
            "storage_path": "auditor_data/sensors"
        }
    })
    
    manager = SensorManager(registry, {
        "sensors_enabled": True,
        "collection_interval_seconds": 5
    })
    
    # Create and register sensors
    obligation_sensor = ObligationLedgerSensor(config={
        "rule_application_threshold": 0.8,
        "max_circular_dependencies": 0  # Set to 0 to detect any circular dependencies
    })
    registry.register_sensor(obligation_sensor)
    
    # Create LLM bridge
    bridge = SensorLLMBridge(registry)
    
    return registry, manager, bridge, obligation_sensor


def demonstrate_error_detection(ledger, sensor):
    """Demonstrate error detection with the ObligationLedger."""
    print("\n=== Demonstrating Error Detection ===\n")
    
    # First, populate with valid data and check (should be no errors)
    ledger.populate_valid_data()
    reports = sensor.monitor(ledger)
    print(f"Monitoring with valid data: {len(reports)} errors detected")
    
    # Clear obligations to trigger an error
    ledger.clear()
    reports = sensor.monitor(ledger)
    print(f"After clearing obligations: {len(reports)} errors detected")
    for i, report in enumerate(reports):
        print(f"  Error {i+1}: {report.error_type} - {report.severity}")
        print(f"    Details: {report.details.get('message', 'No details')}")
    
    # Add circular dependency to trigger another error
    ledger.add_circular_dependency()
    reports = sensor.monitor(ledger)
    print(f"After adding circular dependency: {len(reports)} errors detected")
    for i, report in enumerate(reports):
        print(f"  Error {i+1}: {report.error_type} - {report.severity}")
        print(f"    Details: {report.details.get('message', 'No details')}")
    
    return reports


def demonstrate_llm_awareness(bridge, ledger, sensor):
    """Demonstrate the LLM's awareness of the system state."""
    print("\n=== Demonstrating LLM Self-Awareness ===\n")
    
    # Generate some errors
    ledger.clear()
    ledger.add_circular_dependency()
    reports = sensor.monitor(ledger)
    
    # Collect errors
    bridge.registry.collect_errors()
    
    # Analyze errors with bridge
    analysis = bridge.analyze_errors()
    
    print("LLM Analysis of Errors:")
    print(f"Total errors: {analysis['total_errors']}")
    print(f"Components with issues: {', '.join(analysis['by_component'].keys())}")
    print("\nNatural Language Summary:")
    print(analysis["natural_language_summary"])
    
    # Get component health
    health = bridge.get_component_health("ObligationLedger")
    print("\nComponent Health:")
    print(f"Health score: {health['health_score']}/100")
    print(f"Status: {health['status']}")
    print("\nNatural Language Health Summary:")
    print(health["natural_language_summary"])
    
    # Get system introspection
    introspection = bridge.get_system_introspection()
    print("\nSystem Introspection:")
    print(f"Overall health: {introspection['overall_health']}/100")
    print(f"Overall status: {introspection['overall_status']}")
    print("\nNatural Language System Summary:")
    print(introspection["natural_language_summary"])
    
    return analysis, health, introspection


def demonstrate_error_lifecycle(registry):
    """Demonstrate the lifecycle of an error report."""
    print("\n=== Demonstrating Error Lifecycle ===\n")
    
    # Create a new error report
    report = ErrorReport(
        sensor_id="demo_sensor",
        component_name="DemoComponent",
        error_type="DEMO_ERROR",
        severity="HIGH",
        details={"message": "This is a demonstration error"},
        context={"demo_id": 123}
    )
    
    print(f"New error created: {report.error_id}")
    print(f"Status: {report.status}")
    
    # Acknowledge the error
    report.acknowledge()
    print(f"After acknowledgement - Status: {report.status}")
    
    # Add root cause
    report.add_root_cause("Configuration parameter 'max_retries' is set too low")
    print(f"Root cause added: {report.root_cause}")
    
    # Add recommendation
    report.add_recommendation({
        "message": "Increase 'max_retries' to at least 5",
        "code_change": "config['max_retries'] = 5",
        "priority": "HIGH"
    })
    print(f"Recommendation added: {report.recommendations[0]['message']}")
    
    # Resolve the error
    report.resolve("Increased 'max_retries' to 5 as recommended")
    print(f"After resolution - Status: {report.status}")
    print(f"Resolution: {report.resolution}")
    print(f"Resolved at: {report.resolution_timestamp}")
    
    return report


def main():
    """Main demonstration function."""
    print("\n" + "="*50)
    print("AUDITOR SELF-AWARENESS DEMONSTRATION")
    print("="*50 + "\n")
    
    print("Initializing sensor system...")
    registry, manager, bridge, obligation_sensor = create_sensor_system()
    
    print("Creating demonstration ledger...")
    ledger = DemonstrationObligationLedger()
    
    # Demonstrate error detection
    reports = demonstrate_error_detection(ledger, obligation_sensor)
    
    # Demonstrate LLM awareness
    analysis, health, introspection = demonstrate_llm_awareness(bridge, ledger, obligation_sensor)
    
    # Demonstrate error lifecycle
    report = demonstrate_error_lifecycle(registry)
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50 + "\n")
    
    print("The auditor system now has comprehensive self-awareness through its")
    print("integrated sensor framework. It can detect issues in its own components,")
    print("generate natural language explanations, and provide introspection about")
    print("its internal state.")
    
    return 0


if __name__ == "__main__":
    exit(main())
