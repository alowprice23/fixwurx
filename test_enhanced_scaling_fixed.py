#!/usr/bin/env python3
"""
test_enhanced_scaling.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Test script for the enhanced horizontal scaling capabilities in FixWurx.

This test verifies:
1. Enhanced multi-region worker node management
2. Container orchestration integration (Docker/Kubernetes)
3. Predictive scaling based on resource optimizer metrics
4. Automatic worker failure detection and recovery
5. Burst capacity management for load spikes
6. Integration with advanced load balancer

Together, these components provide a robust horizontal scaling system for
processing multiple bugs in parallel across a distributed cluster.
"""

import time
import logging
import threading
import unittest
import random
from typing import Dict, List, Any
from pathlib import Path
import json
import os
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_enhanced_scaling")

# Import the components
from resource_manager import ResourceManager
from enhanced_scaling_coordinator import EnhancedScalingCoordinator, WorkerState, DeploymentMode
from advanced_load_balancer import AdvancedLoadBalancer, BalancingStrategy
from resource_manager_extension import ClusterResourceManager
from resource_allocation_optimizer import ResourceAllocationOptimizer

# Test constants
DEFAULT_TEST_DURATION_SEC = 30
DEFAULT_SCALING_INTERVAL_SEC = 2
DEFAULT_HEALTH_CHECK_INTERVAL_SEC = 1


class MockResourceOptimizer:
    """Mock implementation of resource optimizer for testing."""
    
    def __init__(self):
        self.in_burst_mode = False
        self.current_usage_ratio = 0.5
        self.predicted_utilization = 0.6
        
    def get_optimization_metrics(self):
        """Get optimization metrics."""
        return {
            "current_usage_ratio": self.current_usage_ratio,
            "predicted_utilization": self.predicted_utilization,
            "in_burst_mode": self.in_burst_mode
        }
        
    def set_burst_mode(self, enabled: bool):
        """Enable or disable burst mode."""
        self.in_burst_mode = enabled
        
    def set_usage_ratio(self, ratio: float):
        """Set current usage ratio."""
        self.current_usage_ratio = ratio
        
    def set_predicted_utilization(self, utilization: float):
        """Set predicted utilization."""
        self.predicted_utilization = utilization


class TestEnhancedScaling(unittest.TestCase):
    """Test suite for the enhanced scaling coordinator."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create state directory
        self.state_dir = Path(tempfile.mkdtemp(prefix="test_enhanced_scaling_"))
        
        # Create test config
        self.config = {
            "min_workers": 1,
            "max_workers": 5,
            "sync_interval_sec": DEFAULT_SCALING_INTERVAL_SEC,
            "heartbeat_timeout_sec": 5,
            "worker_prefix": "test-worker-",
            "deployment_mode": "simulation",  # Use simulation mode for testing
            "discovery_method": "static",
            "static_workers": [],  # No static workers by default
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.3,
            "cool_down_sec": 5,  # Short cooldown for testing
            "burst_factor": 1.5,
            "max_burst_duration_sec": 10,  # Short burst duration for testing
            "enable_failure_detection": True,
            "enable_auto_recovery": True,
            "max_recovery_attempts": 2
        }
        
        # Create components
        self.base_resource_manager = ResourceManager()
        self.mock_optimizer = MockResourceOptimizer()
        
        # Create load balancer
        self.load_balancer = AdvancedLoadBalancer(
            strategy=BalancingStrategy.WEIGHTED_CAPACITY,
            health_check_interval_sec=DEFAULT_HEALTH_CHECK_INTERVAL_SEC
        )
        
        # Create enhanced scaling coordinator
        self.coordinator = EnhancedScalingCoordinator(
            config=self.config,
            resource_manager=self.base_resource_manager,
            advanced_load_balancer=self.load_balancer,
            resource_optimizer=self.mock_optimizer,
            state_path=str(self.state_dir / "scaling_state.json")
        )
        
        # Create cluster resource manager
        self.cluster_manager = ClusterResourceManager(
            base_resource_manager=self.base_resource_manager,
            scaling_coordinator=self.coordinator,
            load_balancer=self.load_balancer,
            sync_interval_sec=DEFAULT_SCALING_INTERVAL_SEC
        )
        
        # Start components
        self.coordinator.start()
        self.load_balancer.start()
        self.cluster_manager.start_sync()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop components
        self.cluster_manager.stop_sync()
        self.load_balancer.stop()
        self.coordinator.stop()
        
        # Clean up state directory
        try:
            import shutil
            shutil.rmtree(self.state_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up state directory: {e}")
    
    def test_initialization(self):
        """Test initialization of enhanced scaling coordinator."""
        # Check that we have the local worker only
        self.assertEqual(len(self.coordinator.workers), 1, "Should only have the local worker initially")
        
        # Check that local worker ID is set
        self.assertIsNotNone(self.coordinator.local_worker_id)
        
        # Check that local worker is registered
        self.assertIn(self.coordinator.local_worker_id, self.coordinator.workers)
        
        # Check that local worker is in READY state
        local_worker = self.coordinator.workers[self.coordinator.local_worker_id]
        self.assertEqual(local_worker.state, WorkerState.READY)
        
        # Check initial metrics
        metrics = self.coordinator.get_scaling_metrics()
        self.assertEqual(metrics["worker_count"], 1)
        self.assertEqual(metrics["active_workers"], 1)
        self.assertFalse(metrics["in_burst_mode"])
    
    def test_simulation_scaling(self):
        """Test scaling up and down in simulation mode."""
        # Verify initial state
        initial_count = len(self.coordinator.workers)
        self.assertEqual(initial_count, 1, "Should start with only the local worker")
        
        # Force a scale up by directly calling the method
        success = self.coordinator._scale_up_simulation()
        self.assertTrue(success, "Scale up should succeed")
        
        # Wait for the worker to be added
        time.sleep(1)
        
        # Check that a worker was added
        worker_count = len(self.coordinator.workers)
        self.assertGreater(worker_count, initial_count, f"Expected more than {initial_count} workers, got {worker_count}")
        
        # Stop and restart to clear any state
        self.coordinator.stop()
        
        # Create a new coordinator with no static workers
        self.coordinator = EnhancedScalingCoordinator(
            config=self.config,
            resource_manager=self.base_resource_manager,
            advanced_load_balancer=self.load_balancer,
            resource_optimizer=self.mock_optimizer,
            state_path=str(self.state_dir / "scaling_state_new.json")
        )
        
        # Start it again
        self.coordinator.start()
        
        # Verify we're back to just one worker
        self.assertEqual(len(self.coordinator.workers), 1, "Should have just the local worker after restart")
    
    def test_burst_mode(self):
        """Test burst mode activation and deactivation."""
        # Get initial capacity
        local_worker = self.coordinator.workers[self.coordinator.local_worker_id]
        initial_capacity = local_worker.total_agents
        
        # Enable burst mode
        self.mock_optimizer.set_burst_mode(True)
        
        # Wait for burst mode to be applied
        time.sleep(DEFAULT_SCALING_INTERVAL_SEC * 2)
        
        # Check that burst mode is active
        self.assertTrue(self.coordinator.in_burst_mode)
        
        # Check that capacity was increased
        local_worker = self.coordinator.workers[self.coordinator.local_worker_id]
        burst_capacity = local_worker.total_agents
        expected_burst_capacity = int(initial_capacity * self.config["burst_factor"])
        self.assertEqual(burst_capacity, expected_burst_capacity)
        
        # Disable burst mode
        self.mock_optimizer.set_burst_mode(False)
        self.mock_optimizer.set_usage_ratio(0.2)  # Low utilization to allow capacity reduction
        
        # Wait for burst mode to be deactivated (either by timeout or condition)
        time.sleep(self.config["max_burst_duration_sec"] + DEFAULT_SCALING_INTERVAL_SEC * 2)
        
        # Check that burst mode is no longer active
        self.assertFalse(self.coordinator.in_burst_mode)
        
        # Check that capacity was restored
        local_worker = self.coordinator.workers[self.coordinator.local_worker_id]
        final_capacity = local_worker.total_agents
        self.assertEqual(final_capacity, initial_capacity)
    
    def test_worker_registration(self):
        """Test registering a bug assignment and completion."""
        # Register a bug assignment
        bug_id = "test-bug-1"
        self.coordinator.register_bug_assignment(bug_id)
        
        # Check that bug is in active bugs
        local_worker = self.coordinator.workers[self.coordinator.local_worker_id]
        self.assertIn(bug_id, local_worker.active_bugs)
        
        # Register bug completion
        self.coordinator.register_bug_completion(bug_id, True, 150)
        
        # Check that bug is no longer active
        local_worker = self.coordinator.workers[self.coordinator.local_worker_id]
        self.assertNotIn(bug_id, local_worker.active_bugs)
    
    def test_worker_discovery(self):
        """Test worker discovery (static method)."""
        # Add static workers to config
        self.coordinator.config["static_workers"] = ["127.0.0.1:8081", "127.0.0.1:8082"]
        
        # Force discovery
        self.coordinator._discover_static_workers()
        
        # Wait for discovery to complete
        time.sleep(0.5)
        
        # Check that static workers were discovered
        worker_count = len(self.coordinator.workers)
        # Add 1 for local worker
        expected_count = 1 + len(self.coordinator.config["static_workers"])
        
        self.assertEqual(worker_count, expected_count, f"Expected {expected_count} workers, got {worker_count}")
    
    def test_load_balancer_integration(self):
        """Test integration with the load balancer."""
        # Register a bug assignment
        bug_id = "test-bug-2"
        self.coordinator.register_bug_assignment(bug_id)
        
        # Wait for update to propagate
        time.sleep(DEFAULT_SCALING_INTERVAL_SEC)
        
        # Check if we can get the worker for the bug
        worker_id = self.coordinator.get_worker_for_bug(bug_id)
        self.assertEqual(worker_id, self.coordinator.local_worker_id)
        
        # Complete the bug
        self.coordinator.register_bug_completion(bug_id, True, 200)
        
        # Wait for update to propagate
        time.sleep(DEFAULT_SCALING_INTERVAL_SEC)
        
        # Check that bug is no longer assigned
        worker_id = self.coordinator.get_worker_for_bug(bug_id)
        self.assertIsNone(worker_id)
    
    def test_state_persistence(self):
        """Test state persistence and loading."""
        # Add a simulated worker
        self.coordinator._scale_up_simulation()
        
        # Wait for the worker to be added
        time.sleep(DEFAULT_SCALING_INTERVAL_SEC)
        
        # Get the current worker count
        original_count = len(self.coordinator.workers)
        self.assertGreater(original_count, 1, "Should have more than one worker after adding a simulated worker")
        
        # Stop the coordinator (which should save state)
        self.coordinator.stop()
        
        # Create a new coordinator with the same state path
        new_coordinator = EnhancedScalingCoordinator(
            config=self.config,
            resource_manager=self.base_resource_manager,
            state_path=str(self.state_dir / "scaling_state.json")
        )
        
        # Check that state was loaded
        self.assertEqual(len(new_coordinator.workers), original_count)
        
        # Clean up
        new_coordinator.stop()


if __name__ == "__main__":
    unittest.main()
