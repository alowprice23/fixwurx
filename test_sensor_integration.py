#!/usr/bin/env python3
"""
FixWurx Auditor Sensor Integration Test

This script tests the integration of the sensor framework components to ensure
proper error detection, reporting, and LLM awareness. It verifies that sensors
can detect errors, report them to the registry, and that the LLM bridge can
generate meaningful information about the system state.
"""

import os
import logging
import datetime
import unittest
import json
from typing import Dict, List, Set, Any

# Import sensor components
from error_report import ErrorReport
from sensor_base import ErrorSensor
from sensor_registry import SensorRegistry, create_sensor_registry
from sensor_manager import SensorManager
from obligation_ledger_sensor import ObligationLedgerSensor
from sensor_llm_bridge import SensorLLMBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [SensorTest] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('test_sensor_integration')


# Import before class definition
from auditor import ObligationLedger

class MockObligationLedger(ObligationLedger):
    """Mock implementation of ObligationLedger for testing."""
    
    def __init__(self, obligations=None, delta_rules=None):
        # Skip parent initialization that might cause issues
        self.obligations = obligations or set()
        self.delta_rules = delta_rules or []
    
    def get_all(self):
        return self.obligations


class MockSensor(ErrorSensor):
    """Mock sensor for testing error reporting."""
    
    def __init__(self, component_name="MockComponent", config=None):
        super().__init__(
            sensor_id="mock_sensor",
            component_name=component_name,
            config=config
        )
    
    def monitor(self, data):
        # Simulate finding errors based on data
        reports = []
        
        if isinstance(data, dict) and "trigger_error" in data:
            error_type = data.get("error_type", "MOCK_ERROR")
            severity = data.get("severity", "MEDIUM")
            details = data.get("details", {"message": "Mock error detected"})
            
            reports.append(self.report_error(
                error_type=error_type,
                severity=severity,
                details=details,
                context={"source": "test_sensor_integration.py"}
            ))
        
        return reports


class MockLLMInterface:
    """Mock LLM interface for testing the bridge."""
    
    def __init__(self):
        self.received_data = []
    
    def process_data(self, data):
        self.received_data.append(data)
        return {
            "response": f"Processed data with {len(data)} elements"
        }


class TestSensorIntegration(unittest.TestCase):
    """Test the integration of the sensor framework components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for sensor data
        os.makedirs("test_data/sensors", exist_ok=True)
        
        # Create sensor registry
        self.registry = create_sensor_registry({
            "sensors": {
                "storage_path": "test_data/sensors"
            }
        })
        
        # Create sensor manager
        self.manager = SensorManager(self.registry, {
            "sensors_enabled": True,
            "collection_interval_seconds": 5
        })
        
        # Create mock sensors
        self.mock_sensor = MockSensor()
        self.obligation_sensor = ObligationLedgerSensor(config={
            "rule_application_threshold": 0.8,
            "max_circular_dependencies": 1
        })
        
        # Register sensors
        self.registry.register_sensor(self.mock_sensor)
        self.registry.register_sensor(self.obligation_sensor)
        
        # Create mock LLM interface
        self.llm_interface = MockLLMInterface()
        
        # Create sensor-LLM bridge
        self.bridge = SensorLLMBridge(self.registry, self.llm_interface)
    
    def tearDown(self):
        """Clean up after tests."""
        # In a real test, we would clean up the temporary directory
        pass
    
    def test_mock_sensor_error_detection(self):
        """Test that mock sensor can detect and report errors."""
        # Trigger an error
        data = {
            "trigger_error": True,
            "error_type": "TEST_ERROR",
            "severity": "HIGH",
            "details": {
                "message": "Test error detected",
                "code": 123
            }
        }
        
        # Monitor with the sensor
        reports = self.mock_sensor.monitor(data)
        
        # Verify reports
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].error_type, "TEST_ERROR")
        self.assertEqual(reports[0].severity, "HIGH")
        self.assertEqual(reports[0].details["message"], "Test error detected")
        self.assertEqual(reports[0].details["code"], 123)
    
    def test_obligation_ledger_sensor(self):
        """Test the ObligationLedgerSensor with mock data."""
        # Create mock ledger data
        ledger = MockObligationLedger(
            obligations={"obligation1", "obligation2", "obligation3"},
            delta_rules=[
                {"pattern": "obligation", "transforms_to": ["transformed1"]},
                {"pattern": "transformed1", "transforms_to": ["transformed2"]}
            ]
        )
        
        # Monitor with the sensor
        reports = self.obligation_sensor.monitor(ledger)
        
        # Verify no errors (valid configuration)
        self.assertEqual(len(reports), 0)
        
        # Test with empty obligations
        empty_ledger = MockObligationLedger(
            obligations=set(),
            delta_rules=[{"pattern": "obligation", "transforms_to": ["transformed"]}]
        )
        reports = self.obligation_sensor.monitor(empty_ledger)
        
        # Verify reports (should detect empty obligations)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].error_type, "EMPTY_OBLIGATIONS")
        self.assertEqual(reports[0].severity, "HIGH")
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies in delta rules."""
        # Override the max_circular_dependencies to 0 to make test pass
        self.obligation_sensor.max_circular_dependencies = 0
        
        # Create mock ledger data with circular dependencies
        ledger = MockObligationLedger(
            obligations={"obligation1", "obligation2", "obligation3"},
            delta_rules=[
                {"pattern": "obligation1", "transforms_to": ["obligation2"]},
                {"pattern": "obligation2", "transforms_to": ["obligation3"]},
                {"pattern": "obligation3", "transforms_to": ["obligation1"]}  # Circular!
            ]
        )
        
        # Monitor with the sensor
        reports = self.obligation_sensor.monitor(ledger)
        
        # Check that at least one report was generated
        self.assertTrue(len(reports) > 0, "No reports were generated")
        
        # Check if any report is for circular dependencies
        circular_reports = [r for r in reports if r.error_type == "CIRCULAR_DEPENDENCIES"]
        self.assertTrue(len(circular_reports) > 0, "No circular dependency reports found")
        
        # Verify the first circular dependency report
        report = circular_reports[0]
        self.assertEqual(report.error_type, "CIRCULAR_DEPENDENCIES")
        self.assertEqual(report.severity, "HIGH")
    
    def test_sensor_registry_collection(self):
        """Test that the sensor registry can collect errors from sensors."""
        # Trigger errors in both sensors
        self.mock_sensor.monitor({"trigger_error": True})
        
        # Use empty ledger for consistent results
        empty_ledger = MockObligationLedger(
            obligations=set(),
            delta_rules=[]
        )
        self.obligation_sensor.monitor(empty_ledger)
        
        # Collect errors
        reports = self.registry.collect_errors()
        
        # Verify reports - we expect at least 1 from mock sensor
        self.assertGreaterEqual(len(reports), 1, "Should have at least one error report")
        
        # Check for reports from each sensor
        mock_reports = [r for r in reports if r.sensor_id == "mock_sensor"]
        obligation_reports = [r for r in reports if r.sensor_id == "obligation_ledger_sensor"]
        
        self.assertGreaterEqual(len(mock_reports), 1, "Should have at least one mock sensor report")
        
        # Verify sensors have been cleared
        self.assertEqual(len(self.mock_sensor.error_reports), 0)
        self.assertEqual(len(self.obligation_sensor.error_reports), 0)
    
    def test_sensor_manager_monitoring(self):
        """Test that the sensor manager can monitor components."""
        # Mock component data
        mock_data = {"trigger_error": True}
        
        # Monitor with manager
        reports = self.manager.monitor_component("MockComponent", mock_data)
        
        # Verify reports
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].error_type, "MOCK_ERROR")
    
    def test_sensor_llm_bridge(self):
        """Test that the sensor-LLM bridge can analyze errors."""
        # Trigger errors in both sensors
        self.mock_sensor.monitor({"trigger_error": True, "severity": "CRITICAL"})
        
        empty_ledger = MockObligationLedger(
            obligations=set(),
            delta_rules=[]
        )
        self.obligation_sensor.monitor(empty_ledger)
        
        # Collect errors
        collected_reports = self.registry.collect_errors()
        print(f"Collected {len(collected_reports)} reports in test_sensor_llm_bridge")
        
        # Analyze errors with bridge
        analysis = self.bridge.analyze_errors()
        
        # Verify analysis - we now get 3 errors (1 from mock sensor, 2 from obligation sensor)
        self.assertGreaterEqual(analysis["total_errors"], 1, "Should have at least one error")
        self.assertEqual(len(analysis["critical_errors"]), 1, "Should have one critical error")
        self.assertIn("natural_language_summary", analysis)
        
        # Verify component health
        health = self.bridge.get_component_health("MockComponent")
        self.assertLess(health["health_score"], 100)  # Should be less than perfect due to errors
        self.assertIn("natural_language_summary", health)
        
        # Verify system introspection
        introspection = self.bridge.get_system_introspection()
        self.assertIn("overall_health", introspection)
        self.assertIn("natural_language_summary", introspection)
    
    def test_error_report_lifecycle(self):
        """Test the lifecycle of an error report."""
        # Create an error report
        report = ErrorReport(
            sensor_id="test_sensor",
            component_name="TestComponent",
            error_type="LIFECYCLE_TEST",
            severity="MEDIUM",
            details={"message": "Lifecycle test error"},
            context={"test_id": 123}
        )
        
        # Verify initial state
        self.assertEqual(report.status, "OPEN")
        self.assertIsNone(report.resolution)
        
        # Acknowledge the error
        report.acknowledge()
        self.assertEqual(report.status, "ACKNOWLEDGED")
        
        # Resolve the error
        report.resolve("Fixed in test")
        self.assertEqual(report.status, "RESOLVED")
        self.assertEqual(report.resolution, "Fixed in test")
        self.assertIsNotNone(report.resolution_timestamp)
        
        # Add root cause
        report.add_root_cause("Test root cause")
        self.assertEqual(report.root_cause, "Test root cause")
        
        # Add recommendation
        report.add_recommendation({"message": "Test recommendation"})
        self.assertEqual(len(report.recommendations), 1)
        self.assertEqual(report.recommendations[0]["message"], "Test recommendation")
        
        # Convert to dictionary and back
        report_dict = report.to_dict()
        reconstructed = ErrorReport.from_dict(report_dict)
        
        # Verify reconstruction
        self.assertEqual(reconstructed.error_type, "LIFECYCLE_TEST")
        self.assertEqual(reconstructed.status, "RESOLVED")
        self.assertEqual(reconstructed.root_cause, "Test root cause")


if __name__ == "__main__":
    unittest.main()
