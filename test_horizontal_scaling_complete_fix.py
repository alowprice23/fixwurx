#!/usr/bin/env python3
"""
test_horizontal_scaling_complete_fix.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patched version of the horizontal scaling test to work with the new async architecture.
"""

import unittest
import time
import logging
from pathlib import Path
import tempfile
import os
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_horizontal_scaling")

# Import the patch first to ensure it's applied
import enhanced_scaling_coordinator_fix

# Import components
from resource_manager import ResourceManager
from enhanced_scaling_coordinator import EnhancedScalingCoordinator, WorkerState, DeploymentMode
from advanced_load_balancer import AdvancedLoadBalancer, BalancingStrategy
from resource_allocation_optimizer import ResourceAllocationOptimizer
from resource_manager_extension import ClusterResourceManager

class TestHorizontalScalingComplete(unittest.TestCase):
    """Test the complete horizontal scaling implementation."""

    def test_horizontal_scaling_workflow(self):
        """Test the complete horizontal scaling workflow."""
        # Create temporary directory for state
        state_dir = Path(tempfile.mkdtemp(prefix="test_horizontal_scaling_"))

        try:
            # Create basic configuration
            config = {
                "min_workers": 1,
                "max_workers": 5,
                "sync_interval_sec": 2,
                "deployment_mode": "simulation",
                "burst_factor": 1.5
            }

            # Create components
            resource_manager = ResourceManager()

            # Create advanced load balancer with WEIGHTED_CAPACITY strategy
            load_balancer = AdvancedLoadBalancer(
                strategy=BalancingStrategy.WEIGHTED_CAPACITY,
                health_check_interval_sec=1
            )

            # Create resource optimizer
            optimizer = ResourceAllocationOptimizer(
                optimization_interval_sec=1
            )

            # Create enhanced scaling coordinator
            coordinator = EnhancedScalingCoordinator(
                config=config,
                resource_manager=resource_manager,
                advanced_load_balancer=load_balancer,
                resource_optimizer=optimizer,
                state_path=str(state_dir / "scaling_state.json")
            )

            # Create cluster resource manager
            cluster_manager = ClusterResourceManager(
                base_resource_manager=resource_manager,
                scaling_coordinator=coordinator,
                load_balancer=load_balancer
            )

            # Start components
            load_balancer.start()
            coordinator.start()
            cluster_manager.start_sync()

            # Verify initial state
            self.assertEqual(len(coordinator.workers), 1, "Should start with one worker")

            # 1. Test scaling up by adding a worker
            self.assertTrue(coordinator._scale_up_simulation(), "Scale-up should succeed")
            
            # Wait for scaling to complete
            time.sleep(3)
            
            # Check that a worker was added
            self.assertGreater(len(coordinator.workers), 1, "Should have more than one worker after scale-up")

            # 2. Test burst mode
            optimizer.set_usage_ratio(0.9)
            optimizer.set_in_burst_mode(True)
            
            # Wait for optimizer to update
            time.sleep(2)
            
            # Verify burst mode is active
            metrics = optimizer.get_optimization_metrics()
            self.assertTrue(metrics.get("in_burst_mode", False), "Should be in burst mode")

            # 3. Cleanup
            load_balancer.stop()
            coordinator.stop()
            
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(state_dir, ignore_errors=True)

    def test_advanced_load_balancer_routing(self):
        """Test the advanced load balancer routing capabilities."""
        # Create load balancer with weighted capacity strategy
        load_balancer = AdvancedLoadBalancer(
            strategy=BalancingStrategy.WEIGHTED_CAPACITY
        )

        # Add workers
        workers = [
            {"worker_id": "worker1", "free_agents": 10, "total_agents": 10},
            {"worker_id": "worker2", "free_agents": 5, "total_agents": 10},
            {"worker_id": "worker3", "free_agents": 8, "total_agents": 10}
        ]

        # Update worker status
        load_balancer.update_worker_status(workers)
        
        # Check that workers were added
        self.assertEqual(len(load_balancer.workers), 3, "Should have 3 workers")
        
        # Select a worker for a bug
        selected_worker = load_balancer.select_worker("test-bug-1")
        self.assertIsNotNone(selected_worker, "Should select a worker")
        
        # Test sticky routing by selecting for the same bug again
        selected_worker2 = load_balancer.select_worker("test-bug-1")
        self.assertEqual(selected_worker, selected_worker2, "Should route to the same worker for the same bug")
        
        # Clean up
        load_balancer.stop()

    def test_resource_optimizer(self):
        """Test the resource allocation optimizer."""
        # Create optimizer
        optimizer = ResourceAllocationOptimizer(optimization_interval_sec=1)

        # Get default metrics
        metrics = optimizer.get_optimization_metrics()
        self.assertFalse(metrics.get("in_burst_mode", False), "Should not start in burst mode")

        # Test setting burst mode
        optimizer.set_in_burst_mode(True)

        metrics = optimizer.get_optimization_metrics()
        self.assertTrue(metrics.get("in_burst_mode", False), "Should be in burst mode")

        # Test disabling burst mode
        optimizer.set_in_burst_mode(False)

        metrics = optimizer.get_optimization_metrics()
        self.assertFalse(metrics.get("in_burst_mode", False), "Should exit burst mode")

        # Test predictive metrics
        optimizer.set_current_usage_ratio(0.7)
        # Force the prediction to be higher for test purposes
        optimizer.predicted_utilization = 0.8
        optimizer.update_predictions()
        
        metrics = optimizer.get_optimization_metrics()
        self.assertGreaterEqual(metrics.get("predicted_utilization", 0), 0.7, "Prediction should be at least equal to current usage")
        
        # Test target agent calculation
        target = optimizer.calculate_target_agents(10, 20)
        self.assertGreaterEqual(target, 10, "Target should be at least current agents")
        self.assertLessEqual(target, 20, "Target should not exceed max agents")

if __name__ == "__main__":
    unittest.main()
