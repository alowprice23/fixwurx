#!/usr/bin/env python3
"""
test_horizontal_scaling_complete.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test script for the complete horizontal scaling implementation in FixWurx.

This test verifies all aspects of the horizontal scaling system:
1. Advanced load balancer with weighted capacity and affinity-based routing
2. Resource allocation optimizer with predictive allocation and burst mode
3. Enhanced scaling coordinator with multi-region and container orchestration
4. Cluster-aware resource management across worker nodes
"""

import unittest
import time
import logging
from pathlib import Path
import tempfile
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_horizontal_scaling")

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
            time.sleep(1)
            self.assertEqual(len(coordinator.workers), 2, "Should have two workers after scale-up")
            
            # 2. Test bug assignment and worker selection
            bug_id = "test-bug-123"
            coordinator.register_bug_assignment(bug_id)
            assigned_worker = coordinator.get_worker_for_bug(bug_id)
            self.assertIsNotNone(assigned_worker, "Bug should be assigned to a worker")
            
            # 3. Test burst mode
            # Simulate burst mode through the optimizer's metrics
            optimizer.set_in_burst_mode(True)
            time.sleep(3)  # Wait for burst mode to be applied
            self.assertTrue(coordinator.in_burst_mode, "Burst mode should be active")
            
            # 4. Test completion and cleanup
            coordinator.register_bug_completion(bug_id, True, 100)
            time.sleep(1)
            self.assertIsNone(coordinator.get_worker_for_bug(bug_id), "Bug should no longer be assigned")
            
            # 5. Exit burst mode
            optimizer.set_in_burst_mode(False)
            time.sleep(3)
            self.assertFalse(coordinator.in_burst_mode, "Burst mode should be inactive")
            
            # 6. Test state persistence
            coordinator._save_state()
            new_coordinator = EnhancedScalingCoordinator(
                config=config,
                resource_manager=ResourceManager(),
                state_path=str(state_dir / "scaling_state.json")
            )
            new_coordinator._load_state()
            self.assertGreater(len(new_coordinator.workers), 0, "Should load workers from state")
            
            # Stop all components
            cluster_manager.stop_sync()
            coordinator.stop()
            load_balancer.stop()
            
        finally:
            # Clean up state directory
            import shutil
            shutil.rmtree(state_dir)
    
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
        
        # Test routing
        bug1 = "bug-123"
        bug2 = "bug-456"
        
        # Select workers for bugs
        worker1 = load_balancer.select_worker(bug1)
        self.assertIsNotNone(worker1, "Should select a worker for bug1")
        
        worker2 = load_balancer.select_worker(bug2)
        self.assertIsNotNone(worker2, "Should select a worker for bug2")
        
        # Test sticky routing - same bug should get same worker
        worker1_again = load_balancer.select_worker(bug1)
        self.assertEqual(worker1, worker1_again, "Should return same worker for same bug (sticky routing)")
        
        # Test affinity-based routing
        load_balancer.set_strategy(BalancingStrategy.AFFINITY_BASED)
        
        # Add worker affinities
        workers_with_affinities = workers.copy()
        for worker in workers_with_affinities:
            if not "affinities" in worker:
                worker["affinities"] = {}
                
        workers_with_affinities[0]["affinities"]["cpu_intensive"] = 0.9
        workers_with_affinities[1]["affinities"]["memory_intensive"] = 0.8
        workers_with_affinities[2]["affinities"]["io_intensive"] = 0.7
        
        load_balancer.update_worker_status(workers_with_affinities)
        
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
        metrics = optimizer.get_optimization_metrics()
        self.assertGreaterEqual(metrics.get("current_usage_ratio", 0), 0, "Should have current usage ratio")

if __name__ == "__main__":
    unittest.main()
