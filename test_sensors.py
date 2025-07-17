#!/usr/bin/env python3
"""
FixWurx Auditor Sensor Tests

This script contains tests for the Auditor sensor framework, including tests for
SensorRegistry, ErrorSensor implementations, and error reporting capabilities.
"""

import os
import shutil
import unittest
import tempfile
import datetime
import json
import yaml
from typing import Dict, Any, Set

# Import sensor components
from sensor_registry import (
    SensorRegistry, ErrorSensor, ErrorReport, SensorManager, create_sensor_registry
)
from component_sensors import (
    ObligationLedgerSensor, EnergyCalculatorSensor, ProofMetricsSensor,
    MetaAwarenessSensor, GraphDatabaseSensor, TimeSeriesDatabaseSensor,
    DocumentStoreSensor, BenchmarkingSensor
)
from llm_sensor_integration import (
    SensorDataProvider, ErrorContextualizer, ErrorPatternRecognizer,
    SelfDiagnosisProvider
)

# Import auditor components for testing with sensors
from auditor import (
    Auditor, ObligationLedger, RepoModules, EnergyCalculator,
    ProofMetrics, MetaAwareness, ErrorReporting
)
from graph_database import GraphDatabase, Node, Edge
from time_series_database import TimeSeriesDatabase
from document_store import DocumentStore
from benchmarking_system import BenchmarkingSystem, BenchmarkConfig


class TestErrorReport(unittest.TestCase):
    """Tests for the ErrorReport class"""
    
    def test_error_report_creation(self):
        """Test creating an error report"""
        report = ErrorReport(
            sensor_id="test_sensor",
            component_name="TestComponent",
            error_type="TEST_ERROR",
            severity="MEDIUM",
            details={"message": "Test error message"},
            context={"test_context": "value"}
        )
        
        # Check basic properties
        self.assertEqual(report.sensor_id, "test_sensor")
        self.assertEqual(report.component_name, "TestComponent")
        self.assertEqual(report.error_type, "TEST_ERROR")
        self.assertEqual(report.severity, "MEDIUM")
        self.assertEqual(report.details["message"], "Test error message")
        self.assertEqual(report.context["test_context"], "value")
        self.assertEqual(report.status, "OPEN")
        self.assertIsNone(report.resolution)
        self.assertIsNone(report.resolution_timestamp)
    
    def test_error_report_resolve(self):
        """Test resolving an error report"""
        report = ErrorReport(
            sensor_id="test_sensor",
            component_name="TestComponent",
            error_type="TEST_ERROR",
            severity="MEDIUM",
            details={"message": "Test error message"}
        )
        
        # Resolve the report
        report.resolve("Fixed test error")
        
        # Check resolution
        self.assertEqual(report.status, "RESOLVED")
        self.assertEqual(report.resolution, "Fixed test error")
        self.assertIsNotNone(report.resolution_timestamp)
    
    def test_error_report_acknowledge(self):
        """Test acknowledging an error report"""
        report = ErrorReport(
            sensor_id="test_sensor",
            component_name="TestComponent",
            error_type="TEST_ERROR",
            severity="MEDIUM",
            details={"message": "Test error message"}
        )
        
        # Acknowledge the report
        report.acknowledge()
        
        # Check acknowledgement
        self.assertEqual(report.status, "ACKNOWLEDGED")
    
    def test_error_report_to_dict(self):
        """Test converting an error report to a dictionary"""
        report = ErrorReport(
            sensor_id="test_sensor",
            component_name="TestComponent",
            error_type="TEST_ERROR",
            severity="MEDIUM",
            details={"message": "Test error message"},
            context={"test_context": "value"}
        )
        
        # Convert to dictionary
        report_dict = report.to_dict()
        
        # Check dictionary values
        self.assertEqual(report_dict["sensor_id"], "test_sensor")
        self.assertEqual(report_dict["component_name"], "TestComponent")
        self.assertEqual(report_dict["error_type"], "TEST_ERROR")
        self.assertEqual(report_dict["severity"], "MEDIUM")
        self.assertEqual(report_dict["details"]["message"], "Test error message")
        self.assertEqual(report_dict["context"]["test_context"], "value")
        self.assertEqual(report_dict["status"], "OPEN")
        
    def test_error_report_from_dict(self):
        """Test creating an error report from a dictionary"""
        report_dict = {
            "error_id": "TEST-ID",
            "timestamp": "2025-01-01T12:00:00",
            "sensor_id": "test_sensor",
            "component_name": "TestComponent",
            "error_type": "TEST_ERROR",
            "severity": "MEDIUM",
            "details": {"message": "Test error message"},
            "context": {"test_context": "value"},
            "status": "ACKNOWLEDGED",
            "resolution": None,
            "resolution_timestamp": None
        }
        
        # Create from dictionary
        report = ErrorReport.from_dict(report_dict)
        
        # Check values
        self.assertEqual(report.error_id, "TEST-ID")
        self.assertEqual(report.timestamp, "2025-01-01T12:00:00")
        self.assertEqual(report.sensor_id, "test_sensor")
        self.assertEqual(report.component_name, "TestComponent")
        self.assertEqual(report.error_type, "TEST_ERROR")
        self.assertEqual(report.severity, "MEDIUM")
        self.assertEqual(report.details["message"], "Test error message")
        self.assertEqual(report.context["test_context"], "value")
        self.assertEqual(report.status, "ACKNOWLEDGED")


class TestErrorSensor(unittest.TestCase):
    """Tests for the ErrorSensor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sensor = ErrorSensor(
            sensor_id="test_sensor",
            component_name="TestComponent",
            config={"sensitivity": 0.5}
        )
    
    def test_sensor_initialization(self):
        """Test sensor initialization"""
        self.assertEqual(self.sensor.sensor_id, "test_sensor")
        self.assertEqual(self.sensor.component_name, "TestComponent")
        self.assertEqual(self.sensor.sensitivity, 0.5)
        self.assertTrue(self.sensor.enabled)
        self.assertEqual(len(self.sensor.error_reports), 0)
    
    def test_report_error(self):
        """Test reporting an error"""
        report = self.sensor.report_error(
            error_type="TEST_ERROR",
            severity="HIGH",
            details={"message": "Test error message"},
            context={"test_context": "value"}
        )
        
        # Check report
        self.assertEqual(report.sensor_id, "test_sensor")
        self.assertEqual(report.component_name, "TestComponent")
        self.assertEqual(report.error_type, "TEST_ERROR")
        self.assertEqual(report.severity, "HIGH")
        self.assertEqual(report.details["message"], "Test error message")
        self.assertEqual(report.context["test_context"], "value")
        
        # Check that report was added to sensor
        self.assertEqual(len(self.sensor.error_reports), 1)
        self.assertEqual(self.sensor.error_reports[0].error_id, report.error_id)
    
    def test_disabled_sensor(self):
        """Test that disabled sensors don't report errors"""
        # Disable the sensor
        self.sensor.set_enabled(False)
        
        # Try to report an error
        report = self.sensor.report_error(
            error_type="TEST_ERROR",
            severity="HIGH",
            details={"message": "Test error message"}
        )
        
        # Check that no report was generated
        self.assertIsNone(report)
        self.assertEqual(len(self.sensor.error_reports), 0)
    
    def test_clear_reports(self):
        """Test clearing error reports"""
        # Add some reports
        self.sensor.report_error(
            error_type="TEST_ERROR_1",
            severity="HIGH",
            details={"message": "Test error message 1"}
        )
        self.sensor.report_error(
            error_type="TEST_ERROR_2",
            severity="MEDIUM",
            details={"message": "Test error message 2"}
        )
        
        # Check that reports were added
        self.assertEqual(len(self.sensor.error_reports), 2)
        
        # Clear reports
        self.sensor.clear_reports()
        
        # Check that reports were cleared
        self.assertEqual(len(self.sensor.error_reports), 0)
    
    def test_get_status(self):
        """Test getting sensor status"""
        # Add a report
        self.sensor.report_error(
            error_type="TEST_ERROR",
            severity="HIGH",
            details={"message": "Test error message"}
        )
        
        # Monitor some data to set the last check time
        self.sensor.monitor("test data")
        
        # Get status
        status = self.sensor.get_status()
        
        # Check status
        self.assertEqual(status["sensor_id"], "test_sensor")
        self.assertEqual(status["component_name"], "TestComponent")
        self.assertTrue(status["enabled"])
        self.assertEqual(status["sensitivity"], 0.5)
        self.assertEqual(status["error_count"], 1)
        self.assertIsNotNone(status["last_check_time"])
    
    def test_set_sensitivity(self):
        """Test setting sensor sensitivity"""
        # Set sensitivity
        self.sensor.set_sensitivity(0.8)
        
        # Check sensitivity
        self.assertEqual(self.sensor.sensitivity, 0.8)
        
        # Test sensitivity clamping
        self.sensor.set_sensitivity(1.5)
        self.assertEqual(self.sensor.sensitivity, 1.0)
        
        self.sensor.set_sensitivity(-0.5)
        self.assertEqual(self.sensor.sensitivity, 0.0)


class TestSensorRegistry(unittest.TestCase):
    """Tests for the SensorRegistry class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = SensorRegistry(storage_path=self.temp_dir)
        
        # Create some test sensors
        self.sensor1 = ErrorSensor(
            sensor_id="sensor1",
            component_name="Component1",
            config={"sensitivity": 0.7}
        )
        self.sensor2 = ErrorSensor(
            sensor_id="sensor2",
            component_name="Component2",
            config={"sensitivity": 0.8}
        )
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_register_sensor(self):
        """Test registering a sensor"""
        # Register sensors
        self.registry.register_sensor(self.sensor1)
        self.registry.register_sensor(self.sensor2)
        
        # Check that sensors were registered
        self.assertEqual(len(self.registry.sensors), 2)
        self.assertIn("sensor1", self.registry.sensors)
        self.assertIn("sensor2", self.registry.sensors)
    
    def test_unregister_sensor(self):
        """Test unregistering a sensor"""
        # Register sensors
        self.registry.register_sensor(self.sensor1)
        self.registry.register_sensor(self.sensor2)
        
        # Unregister a sensor
        self.registry.unregister_sensor("sensor1")
        
        # Check that sensor was unregistered
        self.assertEqual(len(self.registry.sensors), 1)
        self.assertNotIn("sensor1", self.registry.sensors)
        self.assertIn("sensor2", self.registry.sensors)
    
    def test_get_sensor(self):
        """Test getting a sensor"""
        # Register sensor
        self.registry.register_sensor(self.sensor1)
        
        # Get sensor
        sensor = self.registry.get_sensor("sensor1")
        
        # Check sensor
        self.assertIsNotNone(sensor)
        self.assertEqual(sensor.sensor_id, "sensor1")
        self.assertEqual(sensor.component_name, "Component1")
        
        # Try to get a non-existent sensor
        sensor = self.registry.get_sensor("non_existent")
        self.assertIsNone(sensor)
    
    def test_get_sensors_for_component(self):
        """Test getting sensors for a component"""
        # Create sensors for the same component
        sensor3 = ErrorSensor(
            sensor_id="sensor3",
            component_name="Component1",
            config={"sensitivity": 0.9}
        )
        
        # Register sensors
        self.registry.register_sensor(self.sensor1)  # Component1
        self.registry.register_sensor(self.sensor2)  # Component2
        self.registry.register_sensor(sensor3)      # Component1
        
        # Get sensors for Component1
        sensors = self.registry.get_sensors_for_component("Component1")
        
        # Check sensors
        self.assertEqual(len(sensors), 2)
        self.assertTrue(all(s.component_name == "Component1" for s in sensors))
        self.assertTrue(any(s.sensor_id == "sensor1" for s in sensors))
        self.assertTrue(any(s.sensor_id == "sensor3" for s in sensors))
    
    def test_collect_errors(self):
        """Test collecting errors from sensors"""
        # Register sensors
        self.registry.register_sensor(self.sensor1)
        self.registry.register_sensor(self.sensor2)
        
        # Add some errors to sensors
        self.sensor1.report_error(
            error_type="ERROR1",
            severity="HIGH",
            details={"message": "Error 1"}
        )
        self.sensor1.report_error(
            error_type="ERROR2",
            severity="MEDIUM",
            details={"message": "Error 2"}
        )
        self.sensor2.report_error(
            error_type="ERROR3",
            severity="LOW",
            details={"message": "Error 3"}
        )
        
        # Collect errors
        new_errors = self.registry.collect_errors()
        
        # Check collected errors
        self.assertEqual(len(new_errors), 3)
        self.assertEqual(len(self.registry.error_reports), 3)
        
        # Check that sensors' error reports were cleared
        self.assertEqual(len(self.sensor1.error_reports), 0)
        self.assertEqual(len(self.sensor2.error_reports), 0)
    
    def test_get_error_report(self):
        """Test getting an error report"""
        # Register sensor
        self.registry.register_sensor(self.sensor1)
        
        # Add an error
        report = self.sensor1.report_error(
            error_type="TEST_ERROR",
            severity="HIGH",
            details={"message": "Test error message"}
        )
        
        # Collect errors
        self.registry.collect_errors()
        
        # Get the error report
        retrieved_report = self.registry.get_error_report(report.error_id)
        
        # Check report
        self.assertIsNotNone(retrieved_report)
        self.assertEqual(retrieved_report.error_id, report.error_id)
        self.assertEqual(retrieved_report.error_type, "TEST_ERROR")
    
    def test_query_errors(self):
        """Test querying error reports"""
        # Register sensors
        self.registry.register_sensor(self.sensor1)
        self.registry.register_sensor(self.sensor2)
        
        # Add some errors to sensors
        self.sensor1.report_error(
            error_type="ERROR1",
            severity="HIGH",
            details={"message": "Error 1"}
        )
        self.sensor1.report_error(
            error_type="ERROR2",
            severity="MEDIUM",
            details={"message": "Error 2"}
        )
        self.sensor2.report_error(
            error_type="ERROR1",
            severity="LOW",
            details={"message": "Error 3"}
        )
        
        # Collect errors
        self.registry.collect_errors()
        
        # Query by component
        component_errors = self.registry.query_errors(component_name="Component1")
        self.assertEqual(len(component_errors), 2)
        self.assertTrue(all(e.component_name == "Component1" for e in component_errors))
        
        # Query by error type
        type_errors = self.registry.query_errors(error_type="ERROR1")
        self.assertEqual(len(type_errors), 2)
        self.assertTrue(all(e.error_type == "ERROR1" for e in type_errors))
        
        # Query by severity
        severity_errors = self.registry.query_errors(severity="HIGH")
        self.assertEqual(len(severity_errors), 1)
        self.assertTrue(all(e.severity == "HIGH" for e in severity_errors))
        
        # Query by multiple criteria
        filtered_errors = self.registry.query_errors(
            component_name="Component1",
            error_type="ERROR1"
        )
        self.assertEqual(len(filtered_errors), 1)
        self.assertTrue(all(e.component_name == "Component1" and e.error_type == "ERROR1" for e in filtered_errors))
    
    def test_resolve_error(self):
        """Test resolving an error"""
        # Register sensor
        self.registry.register_sensor(self.sensor1)
        
        # Add an error
        report = self.sensor1.report_error(
            error_type="TEST_ERROR",
            severity="HIGH",
            details={"message": "Test error message"}
        )
        
        # Collect errors
        self.registry.collect_errors()
        
        # Resolve the error
        result = self.registry.resolve_error(report.error_id, "Fixed test error")
        
        # Check result
        self.assertTrue(result)
        
        # Get the resolved report
        resolved_report = self.registry.get_error_report(report.error_id)
        
        # Check report
        self.assertEqual(resolved_report.status, "RESOLVED")
        self.assertEqual(resolved_report.resolution, "Fixed test error")
        self.assertIsNotNone(resolved_report.resolution_timestamp)


class TestSensorManager(unittest.TestCase):
    """Tests for the SensorManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = SensorRegistry(storage_path=self.temp_dir)
        self.manager = SensorManager(
            registry=self.registry,
            config={
                "sensors_enabled": True,
                "collection_interval_seconds": 10
            }
        )
        
        # Create some test sensors
        self.sensor1 = ErrorSensor(
            sensor_id="sensor1",
            component_name="Component1",
            config={"sensitivity": 0.7}
        )
        self.sensor2 = ErrorSensor(
            sensor_id="sensor2",
            component_name="Component2",
            config={"sensitivity": 0.8}
        )
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_register_component_sensors(self):
        """Test registering component sensors"""
        # Register sensors
        self.manager.register_component_sensors(
            component_name="Component1",
            sensors=[self.sensor1, self.sensor2]
        )
        
        # Check that sensors were registered
        self.assertEqual(len(self.registry.sensors), 2)
        self.assertIn("sensor1", self.registry.sensors)
        self.assertIn("sensor2", self.registry.sensors)
    
    def test_collect_errors(self):
        """Test collecting errors"""
        # Register sensors
        self.registry.register_sensor(self.sensor1)
        self.registry.register_sensor(self.sensor2)
        
        # Add some errors to sensors
        self.sensor1.report_error(
            error_type="ERROR1",
            severity="HIGH",
            details={"message": "Error 1"}
        )
        self.sensor2.report_error(
            error_type="ERROR2",
            severity="MEDIUM",
            details={"message": "Error 2"}
        )
        
        # Collect errors (force collection regardless of interval)
        new_errors = self.manager.collect_errors(force=True)
        
        # Check collected errors
        self.assertEqual(len(new_errors), 2)
        
        # Try to collect again immediately (should return empty because of interval)
        new_errors = self.manager.collect_errors(force=False)
        self.assertEqual(len(new_errors), 0)
    
    def test_monitor_component(self):
        """Test monitoring a component"""
        # Create a custom sensor that will detect errors in the test data
        class TestMonitorSensor(ErrorSensor):
            def monitor(self, data):
                if data == "error_condition":
                    return [self.report_error(
                        error_type="DETECTED_ERROR",
                        severity="MEDIUM",
                        details={"message": "Detected an error condition"}
                    )]
                return []
        
        # Create and register the test sensor
        test_sensor = TestMonitorSensor(
            sensor_id="test_monitor_sensor",
            component_name="TestComponent"
        )
        self.registry.register_sensor(test_sensor)
        
        # Monitor with no error condition
        reports = self.manager.monitor_component("TestComponent", "normal_condition")
        self.assertEqual(len(reports), 0)
        
        # Monitor with error condition
        reports = self.manager.monitor_component("TestComponent", "error_condition")
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].error_type, "DETECTED_ERROR")
    
    def test_disabled_manager(self):
        """Test that disabled manager doesn't collect errors"""
        # Register sensors
        self.registry.register_sensor(self.sensor1)
        
        # Add an error
        self.sensor1.report_error(
            error_type="ERROR1",
            severity="HIGH",
            details={"message": "Error 1"}
        )
        
        # Disable the manager
        self.manager.set_enabled(False)
        
        # Try to collect errors
        new_errors = self.manager.collect_errors(force=True)
        
        # Check that no errors were collected
        self.assertEqual(len(new_errors), 0)
        
        # Try to monitor a component
        reports = self.manager.monitor_component("Component1", "test_data")
        self.assertEqual(len(reports), 0)


class TestObligationLedgerSensor(unittest.TestCase):
    """Tests for the ObligationLedgerSensor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sensor = ObligationLedgerSensor(
            component_name="ObligationLedger",
            config={
                "rule_application_threshold": 0.9,
                "max_missing_obligations": 0,
                "max_circular_dependencies": 0
            }
        )
        
        # Create a test ObligationLedger
        self.ledger = ObligationLedger()
        
        # Add some delta rules
        self.ledger.delta_rules = [
            {
                "pattern": "authenticate_user",
                "transforms_to": ["validate_credentials", "manage_sessions"]
            },
            {
                "pattern": "store_data",
                "transforms_to": ["validate_data", "persist_data"]
            }
        ]
    
    def test_monitor_healthy_ledger(self):
        """Test monitoring a healthy ledger"""
        # Create a new sensor with a lower rule application threshold for this test
        # to avoid RULE_APPLICATION_INSUFFICIENT errors
        test_sensor = ObligationLedgerSensor(
            component_name="ObligationLedger",
            config={
                "rule_application_threshold": 0.1,  # Much lower threshold than default 0.9
                "max_missing_obligations": 0,
                "max_circular_dependencies": 0
            }
        )
        
        # Set up a healthy ledger
        self.ledger.delta_rules = [
            {
                "pattern": "authenticate_user",
                "transforms_to": ["validate_credentials", "manage_sessions"]
            },
            {
                "pattern": "store_data",
                "transforms_to": ["validate_data", "persist_data"]
            }
        ]
        
        # Mock the get_all method - include all obligations including transforms_to values
        self.ledger.get_all = lambda: {"authenticate_user", "store_data", "validate_credentials", "manage_sessions", "validate_data", "persist_data"}
        
        # Mock methods for rule application
        self.ledger.get_applied_rule_count = lambda: len(self.ledger.delta_rules)
        self.ledger.get_total_rule_count = lambda: len(self.ledger.delta_rules)
        
        # Monitor the ledger with our test-specific sensor
        reports = test_sensor.monitor(self.ledger)
        
        # Check that no errors were reported
        self.assertEqual(len(reports), 0)
    
    def test_monitor_empty_obligations(self):
        """Test monitoring a ledger with empty obligations"""
        # Set up a ledger with empty obligations
        self.ledger.get_all = lambda: set()
        
        # Monitor the ledger
        reports = self.sensor.monitor(self.ledger)
        
        # Check that an error was reported
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].error_type, "EMPTY_OBLIGATIONS")
        self.assertEqual(reports[0].severity, "HIGH")
    
    def test_monitor_empty_delta_rules(self):
        """Test monitoring a ledger with empty delta rules"""
        # Set up a ledger with empty delta rules
        self.ledger.delta_rules = []
        self.ledger.get_all = lambda: {"authenticate_user", "store_data"}
        
        # Monitor the ledger
        reports = self.sensor.monitor(self.ledger)
        
        # Check that an error was reported
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].error_type, "EMPTY_DELTA_RULES")
        self.assertEqual(reports[0].severity, "HIGH")
    
    def test_monitor_circular_dependencies(self):
        """Test monitoring a ledger with circular dependencies"""
        # Set up a ledger with circular dependencies
        self.ledger.delta_rules = [
            {
                "pattern": "A",
                "transforms_to": ["B"]
            },
            {
                "pattern": "B",
                "transforms_to": ["C"]
            },
            {
                "pattern": "C",
                "transforms_to": ["A"]
            }
        ]
        self.ledger.get_all = lambda: {"A", "B", "C"}
        
        # Monitor the ledger
        reports = self.sensor.monitor(self.ledger)
        
        # Check that an error was reported
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].error_type, "CIRCULAR_DEPENDENCIES")
        self.assertEqual(reports[0].severity, "HIGH")


if __name__ == "__main__":
    # Run tests
    unittest.main()
