"""
FixWurx Auditor - Sensor System Design Test

This script tests all the implemented solutions to the design questions.
"""

import logging
import time
import os
import json
import threading
import random
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('test_sensor_system')

# Import the sensor base and error report
from sensor_base import ErrorSensor
from error_report import ErrorReport

# Import our implementations
from sensor_threading_manager import SensorThreadingManager, create_sensor_threading_manager
from sensor_error_handler import SensorErrorHandler, create_sensor_error_handler
from error_aggregation_system import ErrorAggregationSystem, create_error_aggregation_system
from scalable_benchmark_storage import ScalableBenchmarkStorage, create_scalable_benchmark_storage

# Create a test directory
os.makedirs('test_output', exist_ok=True)

# Mock sensors for testing
class MockSensor(ErrorSensor):
    """A mock sensor for testing."""
    
    def __init__(self, sensor_id: str, component_name: str, 
                behavior: str = "normal", 
                error_probability: float = 0.2,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock sensor.
        
        Args:
            sensor_id: Sensor ID
            component_name: Component name
            behavior: Sensor behavior (normal, error_prone, oscillating)
            error_probability: Probability of generating an error on each monitoring cycle
            config: Optional configuration
        """
        super().__init__(sensor_id, component_name, config or {})
        self.behavior = behavior
        self.error_probability = error_probability
        self.call_count = 0
        self.last_health = 100.0
        self.error_types = ["CONNECTIVITY", "TIMEOUT", "DATA_CORRUPTION", "RESOURCE_EXHAUSTION"]
        
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """Monitor the component and return error reports."""
        self.call_count += 1
        reports = []
        
        # Determine if we should generate an error
        generate_error = False
        
        if self.behavior == "normal":
            # Occasionally generate an error
            generate_error = random.random() < self.error_probability
        elif self.behavior == "error_prone":
            # Frequently generate errors
            generate_error = random.random() < (self.error_probability * 3)
        elif self.behavior == "oscillating":
            # Alternate between normal and error states
            generate_error = (self.call_count % 2 == 0)
        
        # Generate an error if needed
        if generate_error:
            error_type = random.choice(self.error_types)
            severity = random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            
            details = {
                "message": f"Mock {error_type} error in {self.component_name}",
                "value": random.randint(1, 100),
                "threshold": random.randint(50, 150)
            }
            
            context = {
                "sensor_call_count": self.call_count,
                "component_state": "degraded" if severity in ["HIGH", "CRITICAL"] else "warning",
                "generated_at": time.time()
            }
            
            # Create error report
            report = self.report_error(
                error_type=error_type,
                severity=severity,
                details=details,
                context=context
            )
            
            reports.append(report)
            
            # Update health
            self.last_health = max(0, self.last_health - random.randint(5, 20))
        else:
            # Recover health slightly
            self.last_health = min(100, self.last_health + random.randint(1, 5))
        
        return reports
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor."""
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "health_score": self.last_health,
            "call_count": self.call_count,
            "behavior": self.behavior
        }

class TestSensorThreadingManager:
    """Test the sensor threading manager."""
    
    def __init__(self):
        """Initialize the test."""
        self.registry = MockSensorRegistry()
        
        # Create sensors with different behaviors
        self.registry.register_sensor(MockSensor("sensor1", "Component1", "normal", 0.2))
        self.registry.register_sensor(MockSensor("sensor2", "Component2", "error_prone", 0.5))
        self.registry.register_sensor(MockSensor("sensor3", "Component3", "oscillating", 0.3))
        
        # Create threading manager
        self.threading_manager = create_sensor_threading_manager(self.registry, interval=1)
        
        # Report queue for testing
        self.report_queue = []
        
        # Replace the report processor
        self.original_process_reports = self.threading_manager._process_reports
        self.threading_manager._process_reports = self._mock_process_reports
    
    def _mock_process_reports(self):
        """Mock process reports for testing."""
        try:
            while not self.threading_manager.report_processor_stop.is_set():
                try:
                    # Get report from queue with timeout
                    report = self.threading_manager.report_queue.get(timeout=1)
                    
                    # Store for testing
                    self.report_queue.append(report)
                    
                    # Mark as done
                    self.threading_manager.report_queue.task_done()
                    
                except Exception:
                    # No reports available, just continue
                    pass
                    
        except Exception as e:
            logger.error(f"Error in mock report processor: {str(e)}")
    
    def run_test(self):
        """Run the test."""
        logger.info("Testing SensorThreadingManager...")
        
        try:
            # Start all sensors
            self.threading_manager.start_all_sensors()
            
            # Let them run for a bit
            logger.info("Running sensors for 5 seconds...")
            time.sleep(5)
            
            # Check status
            status = self.threading_manager.get_thread_status()
            logger.info(f"Thread status: {json.dumps(status, indent=2)}")
            
            # Check report queue
            logger.info(f"Received {len(self.report_queue)} reports")
            
            # Stop specific sensor
            logger.info("Stopping sensor2...")
            self.threading_manager.stop_sensor_thread("sensor2")
            
            # Check status again
            status = self.threading_manager.get_thread_status()
            logger.info(f"Thread status after stopping sensor2: {json.dumps(status, indent=2)}")
            
            # Let run a bit more
            logger.info("Running remaining sensors for 3 more seconds...")
            time.sleep(3)
            
            # Stop all sensors
            logger.info("Stopping all sensors...")
            self.threading_manager.stop_all_sensors()
            
            logger.info("SensorThreadingManager test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in SensorThreadingManager test: {str(e)}")
            return False
        finally:
            # Restore original method
            self.threading_manager._process_reports = self.original_process_reports

class TestSensorErrorHandler:
    """Test the sensor error handler."""
    
    def __init__(self):
        """Initialize the test."""
        self.error_handler = create_sensor_error_handler({
            "error_dir": "test_output/sensor_errors",
            "max_errors_per_hour": 3,
            "max_consecutive_errors": 2
        })
        
        # Test sensors
        self.sensor_ids = ["db_sensor", "api_sensor", "cache_sensor"]
        
        # Recovery tracking
        self.recovery_actions = {}
        
        # Register recovery callbacks
        self.error_handler.register_recovery_callback("restart", self._mock_restart)
        self.error_handler.register_recovery_callback("quarantine", self._mock_quarantine)
    
    def _mock_restart(self, sensor_id: str, context: Optional[Dict[str, Any]] = None):
        """Mock restart action."""
        self.recovery_actions[sensor_id] = self.recovery_actions.get(sensor_id, []) + ["restart"]
        logger.info(f"Mock restart action for {sensor_id}")
    
    def _mock_quarantine(self, sensor_id: str, context: Optional[Dict[str, Any]] = None):
        """Mock quarantine action."""
        self.recovery_actions[sensor_id] = self.recovery_actions.get(sensor_id, []) + ["quarantine"]
        logger.info(f"Mock quarantine action for {sensor_id}")
    
    def run_test(self):
        """Run the test."""
        logger.info("Testing SensorErrorHandler...")
        
        try:
            # Test single error handling
            logger.info("Testing single error handling...")
            self.error_handler.handle_sensor_error(
                "db_sensor", 
                ValueError("Database connection failed"),
                {"attempt": 1, "db_host": "db1.example.com"}
            )
            
            # Check health
            health = self.error_handler.get_sensor_health("db_sensor")
            logger.info(f"Sensor health after single error: {json.dumps(health, indent=2)}")
            
            # Test consecutive errors to trigger quarantine
            logger.info("Testing consecutive errors to trigger quarantine...")
            for i in range(3):
                self.error_handler.handle_sensor_error(
                    "api_sensor",
                    TimeoutError(f"API timeout #{i+1}"),
                    {"endpoint": "/api/data", "timeout_ms": 5000}
                )
            
            # Check health
            health = self.error_handler.get_sensor_health("api_sensor")
            logger.info(f"Sensor health after consecutive errors: {json.dumps(health, indent=2)}")
            
            # Test different error types
            logger.info("Testing different error types...")
            self.error_handler.handle_sensor_error(
                "cache_sensor",
                MemoryError("Out of memory when allocating cache"),
                {"cache_size_mb": 512}
            )
            
            # Check recovery actions
            logger.info(f"Recovery actions: {json.dumps(self.recovery_actions, indent=2)}")
            
            logger.info("SensorErrorHandler test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in SensorErrorHandler test: {str(e)}")
            return False

class TestErrorAggregationSystem:
    """Test the error aggregation system."""
    
    def __init__(self):
        """Initialize the test."""
        self.aggregation_system = create_error_aggregation_system({
            "reports_dir": "test_output/reports",
            "time_window": 10,  # Smaller window for testing
            "component_priorities": {
                "Database": 80,
                "API": 70,
                "Cache": 60
            }
        })
    
    def run_test(self):
        """Run the test."""
        logger.info("Testing ErrorAggregationSystem...")
        
        try:
            # Create some error reports
            reports = []
            
            # Database errors (same component, same error type)
            for i in range(3):
                reports.append(ErrorReport(
                    sensor_id=f"db_sensor",
                    component_name="Database",
                    error_type="CONNECTION_FAILURE",
                    severity="HIGH",
                    details={
                        "message": f"Failed to connect to database (attempt {i+1})",
                        "host": "db.example.com",
                        "port": 5432
                    },
                    context={
                        "connection_attempts": i+1,
                        "last_success": time.time() - 300
                    }
                ))
            
            # API errors (different component, same error type)
            for i in range(2):
                reports.append(ErrorReport(
                    sensor_id=f"api_sensor",
                    component_name="API",
                    error_type="CONNECTION_FAILURE",
                    severity="MEDIUM",
                    details={
                        "message": f"Failed to connect to API (attempt {i+1})",
                        "endpoint": "/api/data",
                        "status_code": 503
                    },
                    context={
                        "retry_count": i+1
                    }
                ))
            
            # Cache errors (different component, different error type)
            reports.append(ErrorReport(
                sensor_id="cache_sensor",
                component_name="Cache",
                error_type="RESOURCE_EXHAUSTION",
                severity="CRITICAL",
                details={
                    "message": "Cache memory exhausted",
                    "used_mb": 980,
                    "total_mb": 1024
                },
                context={
                    "cache_keys": 15000,
                    "eviction_count": 500
                }
            ))
            
            # Process all reports
            logger.info("Processing error reports...")
            group_ids = []
            for report in reports:
                group_id = self.aggregation_system.process_report(report)
                if group_id and group_id not in group_ids:
                    group_ids.append(group_id)
            
            logger.info(f"Created {len(group_ids)} error groups")
            
            # Get prioritized groups
            prioritized = self.aggregation_system.get_prioritized_groups(limit=10)
            logger.info(f"Prioritized groups: {len(prioritized)}")
            
            for i, group in enumerate(prioritized):
                logger.info(f"Group {i+1}: {group['component']} - {group['error_type']} ({group['severity']}) - {group['count']} reports")
            
            # Test acknowledging a group
            if group_ids:
                logger.info(f"Acknowledging group {group_ids[0]}...")
                self.aggregation_system.acknowledge_group(
                    group_ids[0],
                    {
                        "acknowledged_by": "test_user",
                        "notes": "Investigating this issue",
                        "timestamp": time.time()
                    }
                )
                
                # Get the group to verify
                group = self.aggregation_system.get_group(group_ids[0])
                logger.info(f"Group status after acknowledgement: {group['status']}")
            
            # Get statistics
            stats = self.aggregation_system.get_statistics()
            logger.info(f"System statistics: {json.dumps(stats, indent=2)}")
            
            logger.info("ErrorAggregationSystem test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in ErrorAggregationSystem test: {str(e)}")
            return False

class TestScalableBenchmarkStorage:
    """Test the scalable benchmark storage."""
    
    def __init__(self):
        """Initialize the test."""
        self.storage = create_scalable_benchmark_storage({
            "base_dir": "test_output/benchmarks",
            "cache_size": 10,  # Small cache for testing
            "cache_ttl": 5,     # Short TTL for testing
            "cleanup_interval": 3  # Short cleanup interval for testing
        })
    
    def run_test(self):
        """Run the test."""
        logger.info("Testing ScalableBenchmarkStorage...")
        
        try:
            # Create some test projects and sessions
            projects = ["WebApp", "APIService", "DatabaseCluster"]
            sessions = {}
            
            for project in projects:
                sessions[project] = [f"{project}_Session_{i+1}" for i in range(2)]
                
                for session in sessions[project]:
                    # Create session
                    self.storage.create_session(
                        project, 
                        session, 
                        {
                            "description": f"Test session for {project}",
                            "created_by": "test_user"
                        }
                    )
                    
                    # Store some benchmarks
                    for i in range(5):
                        self.storage.store_benchmark(
                            project,
                            session,
                            {
                                "timestamp": time.time() - (i * 60),  # Spaced 1 minute apart
                                "cpu_usage": random.uniform(10, 90),
                                "memory_mb": random.uniform(100, 1000),
                                "response_time_ms": random.uniform(10, 500),
                                "requests_per_second": random.uniform(10, 100),
                                "error_rate": random.uniform(0, 5)
                            }
                        )
            
            # Test retrieving benchmarks
            logger.info("Testing benchmark retrieval...")
            for project in projects:
                for session in sessions[project]:
                    benchmarks = self.storage.get_benchmarks(
                        project,
                        session,
                        limit=10
                    )
                    
                    logger.info(f"Retrieved {len(benchmarks)} benchmarks for {project}/{session}")
            
            # Test getting sessions
            logger.info("Testing session retrieval...")
            for project in projects:
                project_sessions = self.storage.get_sessions(project)
                logger.info(f"Retrieved {len(project_sessions)} sessions for {project}")
            
            # Test ending a session
            if projects and sessions[projects[0]]:
                test_project = projects[0]
                test_session = sessions[test_project][0]
                
                logger.info(f"Ending session {test_project}/{test_session}...")
                self.storage.end_session(
                    test_project,
                    test_session,
                    {
                        "end_reason": "test completed",
                        "success": True
                    }
                )
                
                # Verify session was ended
                project_sessions = self.storage.get_sessions(test_project)
                for session_info in project_sessions:
                    if session_info["session_id"] == test_session:
                        logger.info(f"Session end time: {session_info['end_time']}")
            
            # Get storage statistics
            stats = self.storage.get_statistics()
            logger.info(f"Storage statistics: {json.dumps(stats, indent=2)}")
            
            # Let the cleanup thread run
            logger.info("Waiting for cleanup thread to run...")
            time.sleep(5)
            
            # Close storage
            self.storage.close()
            
            logger.info("ScalableBenchmarkStorage test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in ScalableBenchmarkStorage test: {str(e)}")
            return False
        finally:
            # Ensure storage is closed
            self.storage.close()

# Helper mock registry for testing
class MockSensorRegistry:
    """A mock sensor registry for testing."""
    
    def __init__(self):
        """Initialize the mock registry."""
        self.sensors = {}
    
    def register_sensor(self, sensor: ErrorSensor) -> bool:
        """Register a sensor."""
        self.sensors[sensor.sensor_id] = sensor
        return True
    
    def get_sensor(self, sensor_id: str) -> Optional[ErrorSensor]:
        """Get a sensor by ID."""
        return self.sensors.get(sensor_id)
    
    def get_all_sensor_ids(self) -> List[str]:
        """Get all sensor IDs."""
        return list(self.sensors.keys())

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SENSOR SYSTEM DESIGN TESTS")
    print("="*70 + "\n")
    
    # Dictionary to track test results
    results = {}
    
    # Test threading manager
    print("\n--- Testing Sensor Threading Manager ---\n")
    threading_test = TestSensorThreadingManager()
    results["SensorThreadingManager"] = threading_test.run_test()
    
    # Test error handler
    print("\n--- Testing Sensor Error Handler ---\n")
    error_handler_test = TestSensorErrorHandler()
    results["SensorErrorHandler"] = error_handler_test.run_test()
    
    # Test error aggregation system
    print("\n--- Testing Error Aggregation System ---\n")
    aggregation_test = TestErrorAggregationSystem()
    results["ErrorAggregationSystem"] = aggregation_test.run_test()
    
    # Test scalable benchmark storage
    print("\n--- Testing Scalable Benchmark Storage ---\n")
    storage_test = TestScalableBenchmarkStorage()
    results["ScalableBenchmarkStorage"] = storage_test.run_test()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
