#!/usr/bin/env python3
"""
FixWurx Auditor - Comprehensive Sensor Test Script

This script tests all implemented sensors to verify they are working correctly
with the auditor framework. It tests creation, monitoring, and status reporting
for each sensor.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any

# Import all sensors
from sensor_base import ErrorSensor
from error_report import ErrorReport
from sensor_registry import SensorRegistry, create_sensor_registry
from sensor_manager import SensorManager

# Core sensors
from obligation_ledger_sensor import ObligationLedgerSensor
from graph_database_sensor import GraphDatabaseSensor
from meta_awareness_sensor import MetaAwarenessSensor
from performance_benchmark_sensor import PerformanceBenchmarkSensor

# Additional implemented sensors
from energy_calculator_sensor import EnergyCalculatorSensor
from proof_metrics_sensor import ProofMetricsSensor
from time_series_database_sensor import TimeSeriesDatabaseSensor
from document_store_sensor import DocumentStoreSensor

# System sensors
from memory_monitor_sensor import MemoryMonitorSensor
from threading_safety_sensor import ThreadingSafetySensor
from auditor_agent_activity_sensor import AuditorAgentActivitySensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [SensorTest] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sensor_test')


class SensorTester:
    """Helper class to test sensor functionality."""
    
    def __init__(self):
        """Initialize the sensor tester."""
        # Create sensor registry and manager
        self.registry = create_sensor_registry()
        self.manager = SensorManager(self.registry)
        
        # Create data directory if it doesn't exist
        os.makedirs("auditor_data/test_results", exist_ok=True)
        
        # Sensor test results
        self.test_results = {}
    
    def test_sensor(self, sensor_class, config=None, test_data=None):
        """Test a single sensor's functionality."""
        sensor_name = sensor_class.__name__
        logger.info(f"Testing {sensor_name}...")
        
        try:
            # Create the sensor instance
            sensor = sensor_class(config=config)
            
            # Register the sensor
            self.registry.register_sensor(sensor)
            
            # Test 1: Basic creation and status
            sensor_id = sensor.sensor_id
            component_name = sensor.component_name
            logger.info(f"  Created {sensor_id} for {component_name}")
            
            # Test 2: Status reporting
            status = sensor.get_status()
            logger.info(f"  Status: health_score={status.get('health_score', 'N/A')}")
            
            # Test 3: Monitoring with no data
            try:
                # Some sensors might require data parameter
                reports1 = sensor.monitor()
            except TypeError:
                # If the sensor requires data, pass empty data
                reports1 = sensor.monitor({})
            
            logger.info(f"  Monitor (no data): {len(reports1)} reports")
            
            # Test 4: Monitoring with test data (if provided)
            reports2 = []
            if test_data:
                reports2 = sensor.monitor(test_data)
                logger.info(f"  Monitor (with data): {len(reports2)} reports")
            
            # Store test results
            self.test_results[sensor_name] = {
                "sensor_id": sensor_id,
                "component_name": component_name,
                "status": status,
                "reports_count_no_data": len(reports1),
                "reports_count_with_data": len(reports2),
                "success": True
            }
            
            logger.info(f"✅ {sensor_name} test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ {sensor_name} test failed: {str(e)}")
            
            # Store failure
            self.test_results[sensor_name] = {
                "success": False,
                "error": str(e)
            }
            
            return False
    
    def run_all_tests(self):
        """Run tests for all sensors."""
        # Test data for different sensor types
        test_data = {
            # Common mock data
            "common": {
                "timestamp": time.time(),
                "context": {"test": True}
            },
            
            # Graph database test data
            "graph": {
                "node_count": 500,
                "edge_count": 1200,
                "orphaned_nodes": 5,
                "circular_refs": 2,
                "query_time_ms": 120
            },
            
            # Energy calculation test data
            "energy": {
                "cpu_percent": 65.0,
                "memory_gb": 2.5,
                "io_mb_per_sec": 15.0,
                "bug_fixes": 10
            },
            
            # Proof metrics test data
            "proof": {
                "changes": {
                    "security_verification": {
                        "description": "Updated security checks",
                        "lines_changed": 25,
                        "criticality": "HIGH"
                    }
                }
            },
            
            # Time series database test data
            "time_series": {
                "query_time_ms": 150,
                "insertion_time_ms": 40,
                "index_health": 0.85,
                "data_point_count": 950
            },
            
            # Document store test data
            "document": {
                "query_time_ms": 120,
                "storage_time_ms": 180,
                "search_recall": 0.92,
                "total_docs": 15000,
                "total_size_mb": 2500
            }
        }
        
        # List of sensors to test with their specific test data
        sensors_to_test = [
            # Core sensors
            (ObligationLedgerSensor, None, test_data["common"]),
            (GraphDatabaseSensor, None, test_data["graph"]),
            (MetaAwarenessSensor, {"use_mock_llm": True}, None),
            (PerformanceBenchmarkSensor, None, None),
            
            # Component sensors
            (EnergyCalculatorSensor, None, test_data["energy"]),
            (ProofMetricsSensor, None, test_data["proof"]),
            
            # Database sensors
            (TimeSeriesDatabaseSensor, {"mock_mode": True}, test_data["time_series"]),
            (DocumentStoreSensor, {"mock_mode": True}, test_data["document"]),
            
            # System sensors
            (MemoryMonitorSensor, {"use_tracemalloc": False}, None),
            (ThreadingSafetySensor, {"lightweight_mode": True}, None),
            (AuditorAgentActivitySensor, {"mock_mode": True}, None)
        ]
        
        # Run tests for each sensor
        for sensor_class, config, data in sensors_to_test:
            self.test_sensor(sensor_class, config, data)
        
        # Save test results
        self._save_results()
        
        return self.test_results
    
    def _save_results(self):
        """Save test results to a file."""
        try:
            filename = f"auditor_data/test_results/sensor_tests_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(
                    {
                        "timestamp": time.time(),
                        "results": self.test_results,
                        "success_rate": f"{sum(1 for r in self.test_results.values() if r.get('success', False))}/{len(self.test_results)}"
                    }, 
                    f, 
                    indent=2
                )
            logger.info(f"Test results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")


def main():
    """Run the sensor tests."""
    print("\n" + "="*70)
    print("AUDITOR SENSOR SYSTEM TEST")
    print("="*70 + "\n")
    
    print("This script tests all implemented sensors to verify their functionality.\n")
    
    # Create and run the tester
    tester = SensorTester()
    results = tester.run_all_tests()
    
    # Print summary
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_count = len(results)
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {success_count}/{total_count} sensors passed")
    print("="*70)
    
    for sensor_name, result in results.items():
        status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
        print(f"{sensor_name}: {status}")
        
        if not result.get('success', False):
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit(main())
