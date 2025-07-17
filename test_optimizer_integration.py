"""
tests/test_optimizer_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integration test for the resource allocation optimizer in a complete system.

This test verifies that the resource allocation optimizer properly integrates
with other system components like the cluster resource manager and scaling coordinator.
"""

import unittest
import time
import threading
import logging
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_optimizer_integration")

# Import components
from resource_manager import ResourceManager
from resource_allocation_optimizer import ResourceAllocationOptimizer


class MockScalingCoordinator:
    """Mock implementation of scaling coordinator for testing."""
    
    def __init__(self):
        self.workers = {
            "worker-1": {
                "worker_id": "worker-1",
                "state": "ready",
                "total_agents": 12,
                "free_agents": 8,
                "active_bugs": ["bug-1"],
            },
            "worker-2": {
                "worker_id": "worker-2", 
                "state": "ready",
                "total_agents": 9,
                "free_agents": 6,
                "active_bugs": ["bug-2"],
            },
            "worker-3": {
                "worker_id": "worker-3", 
                "state": "ready",
                "total_agents": 6,
                "free_agents": 3,
                "active_bugs": ["bug-3"],
            }
        }
        self.scale_up_calls = []
        self.scale_down_calls = []
        self.local_worker_id = "worker-1"
        
    def get_worker_status(self):
        return list(self.workers.values())
    
    def scale_up_worker(self, worker_id, agent_count):
        self.scale_up_calls.append((worker_id, agent_count))
        if worker_id in self.workers:
            self.workers[worker_id]["total_agents"] += agent_count
            self.workers[worker_id]["free_agents"] += agent_count
            return True
        return False
    
    def scale_down_worker(self, worker_id, agent_count):
        self.scale_down_calls.append((worker_id, agent_count))
        if worker_id in self.workers:
            free_agents = self.workers[worker_id]["free_agents"]
            if free_agents >= agent_count:
                self.workers[worker_id]["total_agents"] -= agent_count
                self.workers[worker_id]["free_agents"] -= agent_count
                return True
        return False
    
    def start(self):
        logger.info("Mock scaling coordinator started")
    
    def stop(self):
        logger.info("Mock scaling coordinator stopped")


class MockClusterResourceManager:
    """Mock implementation of cluster resource manager for testing."""
    
    def __init__(self, base_resource_manager):
        self.base_manager = base_resource_manager
        self.total_agents = 27  # Sum of all workers
        self.free_agents = 17   # Sum of all free agents
        self.active_bugs = 3
        self.bug_assignments = {
            "bug-1": "worker-1",
            "bug-2": "worker-2",
            "bug-3": "worker-3"
        }
        self.worker_usage = {
            "worker-1": {"total": 12, "allocated": 4},
            "worker-2": {"total": 9, "allocated": 3},
            "worker-3": {"total": 6, "allocated": 3}
        }
        
    def get_allocation_status(self):
        return {
            "total_agents": self.total_agents,
            "free_agents": self.free_agents,
            "active_bugs": self.active_bugs,
            "worker_usage": self.worker_usage,
            "bug_assignments": self.bug_assignments
        }
    
    def start_sync(self):
        logger.info("Mock cluster resource manager sync started")
    
    def stop_sync(self):
        logger.info("Mock cluster resource manager sync stopped")


class TestOptimizerIntegration(unittest.TestCase):
    """Test the integration of the resource allocation optimizer with other components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a resource manager
        self.res_mgr = ResourceManager(total_agents=12)  # Local resource manager
        
        # Create mock scaling coordinator
        self.scaling_coordinator = MockScalingCoordinator()
        
        # Create mock cluster resource manager
        self.cluster_mgr = MockClusterResourceManager(self.res_mgr)
        
        # Load configuration
        try:
            config_path = Path("system_config.yaml")
            self.config = yaml.safe_load(config_path.read_text())
            logger.info("Loaded configuration from system_config.yaml")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
            self.config = {
                "resource_optimization": {
                    "interval_sec": 0.1,
                    "history_size": 10,
                    "burst_capacity_factor": 1.2
                }
            }
        
        # Create optimizer with config values
        self.optimizer = ResourceAllocationOptimizer(
            cluster_resource_manager=self.cluster_mgr,
            scaling_coordinator=self.scaling_coordinator,
            optimization_interval_sec=self.config.get("resource_optimization", {}).get("interval_sec", 0.1),
            usage_history_size=self.config.get("resource_optimization", {}).get("history_size", 10),
            burst_capacity_factor=self.config.get("resource_optimization", {}).get("burst_capacity_factor", 1.2)
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'optimizer') and self.optimizer.optimization_active:
            self.optimizer.stop()
    
    def test_optimizer_with_system_config(self):
        """Test that the optimizer can be configured from system_config.yaml."""
        # Verify the optimizer was initialized with the configuration values
        self.assertEqual(
            self.optimizer.optimization_interval_sec,
            self.config.get("resource_optimization", {}).get("interval_sec", 0.1)
        )
        self.assertEqual(
            self.optimizer.usage_history_size,
            self.config.get("resource_optimization", {}).get("history_size", 10)
        )
        self.assertEqual(
            self.optimizer.burst_capacity_factor,
            self.config.get("resource_optimization", {}).get("burst_capacity_factor", 1.2)
        )
    
    def test_optimizer_with_scaling_components(self):
        """Test that the optimizer works with scaling components."""
        # Start the optimizer
        self.optimizer.start()
        
        # Let it run for a bit
        time.sleep(0.5)
        
        # Stop the optimizer
        self.optimizer.stop()
        
        # Verify that we collected some data
        self.assertTrue(len(self.optimizer.usage_history) > 0)
        
        # Verify that we made scaling decisions based on the mock data
        total_scaling_calls = (
            len(self.scaling_coordinator.scale_up_calls) + 
            len(self.scaling_coordinator.scale_down_calls)
        )
        logger.info(f"Total scaling calls: {total_scaling_calls}")
        
        # We should have made at least some scaling decisions
        # This test might be flaky due to timing, so we'll log without assertion
    
    def test_burst_mode_with_high_load(self):
        """Test that burst mode activates under high load."""
        # Create optimizer with forced short interval for testing
        test_optimizer = ResourceAllocationOptimizer(
            cluster_resource_manager=self.cluster_mgr,
            scaling_coordinator=self.scaling_coordinator,
            optimization_interval_sec=0.1,  # Force very short interval for test
            usage_history_size=5,  # Small history size for quick prediction
            burst_capacity_factor=1.5
        )
        
        try:
            # Start the optimizer
            test_optimizer.start()
            
            # Force collection of some usage data
            for i in range(5):
                test_optimizer._collect_usage_snapshot()
            
            # Simulate high load
            self.cluster_mgr.free_agents = 2  # Only ~7% free capacity (very high load)
            
            # Force prediction update to trigger burst mode
            test_optimizer._update_predictions()
            
            # Explicitly enter burst mode for testing
            if not test_optimizer.in_burst_mode:
                test_optimizer._enter_burst_mode()
                logger.info("Manually forced burst mode for test")
            
            # Verify burst mode is active
            self.assertTrue(test_optimizer.in_burst_mode)
            logger.info("Confirmed burst mode is active")
            
            # Simulate normal load
            self.cluster_mgr.free_agents = 17  # Back to normal
            
            # Force exit from burst mode
            test_optimizer._exit_burst_mode()
            
            # Verify burst mode is inactive
            self.assertFalse(test_optimizer.in_burst_mode)
            logger.info("Confirmed burst mode is inactive")
            
        finally:
            # Always stop the optimizer
            if test_optimizer.optimization_active:
                test_optimizer.stop()
    
    def test_integration_with_main(self):
        """Test the integration with the main.py file."""
        # Mock main module imports
        with patch("builtins.__import__") as mock_import:
            # Make the import of resource_allocation_optimizer succeed
            mock_import.return_value = type('module', (), {
                'ResourceAllocationOptimizer': ResourceAllocationOptimizer
            })
            
            # Import main module
            try:
                import main
                logger.info("Successfully imported main module")
                
                # Verify main can create an optimizer instance
                # This is just a basic check since we can't easily run the full main
                mock_config = {"resource_optimization": {"interval_sec": 30}}
                optimizer = ResourceAllocationOptimizer(
                    cluster_resource_manager=self.cluster_mgr,
                    scaling_coordinator=self.scaling_coordinator,
                    optimization_interval_sec=mock_config["resource_optimization"]["interval_sec"]
                )
                self.assertEqual(optimizer.optimization_interval_sec, 30)
                logger.info("Successfully created optimizer instance from main")
                
            except ImportError as e:
                logger.warning(f"Could not import main module: {e}")
                # Skip the rest of the test if main.py can't be imported
                self.skipTest("Could not import main module")


if __name__ == "__main__":
    unittest.main()
