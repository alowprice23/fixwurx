#!/usr/bin/env python3
"""
test_async_load_balancer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test suite for the async load balancer component.
"""

import unittest
import asyncio
import time
from typing import Dict, Any, Optional

from async_load_balancer import (
    AsyncLoadBalancer,
    BalancingStrategy,
    LoadBalancerNotRunningError,
    CircuitBreakerOpenError
)

class TestAsyncLoadBalancer(unittest.TestCase):
    """Test cases for AsyncLoadBalancer."""
    
    def setUp(self):
        """Set up test fixtures for each test method."""
        self.config = {
            "hash_replicas": 10,
            "sticky_bugs": True,
            "enable_affinity_routing": True,
            "enable_predictive_routing": True,
            "auto_strategy_selection": False,
            "default_strategy": BalancingStrategy.ROUND_ROBIN,
            "circuit_breaker_failure_threshold": 3,
            "circuit_breaker_reset_timeout_sec": 1
        }
        
        # Create balancer with test_mode=True to ensure deterministic behavior
        self.balancer = AsyncLoadBalancer(config=self.config, test_mode=True)
        
        # Configure test workers
        self.test_workers = [
            {"id": "worker-1", "capacity": 10},
            {"id": "worker-2", "capacity": 8},
            {"id": "worker-3", "capacity": 5}
        ]
    
    def test_initialization(self):
        """Test initialization of async load balancer."""
        self.assertEqual(self.balancer.config["hash_replicas"], 10)
        self.assertTrue(self.balancer.config["sticky_bugs"])
        self.assertTrue(self.balancer.config["enable_affinity_routing"])
        self.assertTrue(self.balancer.config["enable_predictive_routing"])
        self.assertTrue(self.balancer.test_mode)
    
    def test_start_and_stop(self):
        """Test starting and stopping the load balancer."""
        async def run_test():
            # Start the balancer
            await self.balancer.start()
            self.assertEqual(self.balancer.state_manager.current_state.name, "RUNNING")
            
            # Stop the balancer
            await self.balancer.stop()
            self.assertEqual(self.balancer.state_manager.current_state.name, "STOPPED")
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_add_and_remove_worker(self):
        """Test adding and removing workers."""
        async def run_test():
            # Start the balancer
            await self.balancer.start()
            
            # Add workers
            for worker in self.test_workers:
                await self.balancer.add_worker(worker["id"], worker["capacity"])
            
            # Check that workers were added
            workers = await self.balancer.worker_registry.get_all_workers()
            self.assertEqual(len(workers), 3)
            
            # Remove a worker
            await self.balancer.remove_worker("worker-2")
            
            # Check that worker was removed
            workers = await self.balancer.worker_registry.get_all_workers()
            self.assertEqual(len(workers), 2)
            worker_ids = [w.id for w in workers]
            self.assertIn("worker-1", worker_ids)
            self.assertIn("worker-3", worker_ids)
            self.assertNotIn("worker-2", worker_ids)
            
            # Stop the balancer
            await self.balancer.stop()
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_select_worker(self):
        """Test selecting a worker for a task."""
        async def run_test():
            # Start the balancer
            await self.balancer.start()
            
            # Add workers
            for worker in self.test_workers:
                await self.balancer.add_worker(worker["id"], worker["capacity"])
            
            # Select a worker for a task
            worker_id = await self.balancer.select_worker("task-1")
            self.assertIn(worker_id, ["worker-1", "worker-2", "worker-3"])
            
            # Select again for same task (should be sticky)
            worker_id2 = await self.balancer.select_worker("task-1")
            self.assertEqual(worker_id, worker_id2)
            
            # Select with bypass_sticky
            worker_id3 = await self.balancer.select_worker("task-1", {"bypass_sticky": True})
            # Note: This might still return the same worker by chance
            
            # Check that bug routing was recorded
            self.assertIn("task-1", self.balancer.bug_routing)
            
            # Stop the balancer
            await self.balancer.stop()
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_task_completion(self):
        """Test registering task completion."""
        async def run_test():
            # Start the balancer
            await self.balancer.start()
            
            # Add workers
            for worker in self.test_workers:
                await self.balancer.add_worker(worker["id"], worker["capacity"])
            
            # Select a worker for a task
            worker_id = await self.balancer.select_worker("task-1")
            
            # Register successful completion
            await self.balancer.register_task_completion("task-1", worker_id, True, 100)
            
            # Check that success was recorded
            self.assertEqual(self.balancer.bug_routing["task-1"].success_count, 1)
            self.assertEqual(self.balancer.bug_routing["task-1"].failure_count, 0)
            self.assertEqual(self.balancer.bug_routing["task-1"].total_processing_time_ms, 100)
            
            # Register failed completion
            await self.balancer.register_task_completion("task-1", worker_id, False, 150)
            
            # Check that failure was recorded
            self.assertEqual(self.balancer.bug_routing["task-1"].success_count, 1)
            self.assertEqual(self.balancer.bug_routing["task-1"].failure_count, 1)
            self.assertEqual(self.balancer.bug_routing["task-1"].total_processing_time_ms, 250)
            
            # Stop the balancer
            await self.balancer.stop()
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        async def run_test():
            # Start the balancer
            await self.balancer.start()
            
            # Add workers
            for worker in self.test_workers:
                await self.balancer.add_worker(worker["id"], worker["capacity"])
            
            # Record multiple failures to trip the circuit breaker
            for i in range(3):
                await self.balancer.circuit_breaker.record_failure()
            
            # Circuit breaker should be open now
            self.assertTrue(self.balancer.circuit_breaker.is_open())
            
            # Selecting a worker should now fail
            with self.assertRaises(CircuitBreakerOpenError):
                await self.balancer.select_worker("task-1")
            
            # Wait for reset timeout
            await asyncio.sleep(1.1)
            
            # Let circuit breaker check state
            await self.balancer.circuit_breaker.check_state()
            
            # Circuit breaker should be half-open now
            self.assertEqual(self.balancer.circuit_breaker.state, "HALF_OPEN")
            
            # Record a success to close the circuit breaker
            await self.balancer.circuit_breaker.record_success()
            
            # Circuit breaker should be closed now
            self.assertEqual(self.balancer.circuit_breaker.state, "CLOSED")
            
            # Stop the balancer
            await self.balancer.stop()
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_metrics(self):
        """Test metrics collection."""
        async def run_test():
            # Start the balancer
            await self.balancer.start()
            
            # Add workers
            for worker in self.test_workers:
                await self.balancer.add_worker(worker["id"], worker["capacity"])
            
            # Select a worker for a task (records metrics)
            worker_id = await self.balancer.select_worker("task-1")
            
            # Get metrics
            metrics = await self.balancer.get_metrics()
            
            # Check metrics structure
            self.assertIn("state", metrics)
            self.assertIn("worker_count", metrics)
            self.assertIn("healthy_worker_count", metrics)
            self.assertIn("active_tasks_count", metrics)
            self.assertIn("current_strategy", metrics)
            self.assertIn("circuit_breaker_state", metrics)
            self.assertIn("test_mode", metrics)
            
            # Check metrics values
            self.assertEqual(metrics["state"], "RUNNING")
            self.assertEqual(metrics["worker_count"], 3)
            self.assertEqual(metrics["healthy_worker_count"], 3)
            self.assertEqual(metrics["active_tasks_count"], 1)
            self.assertEqual(metrics["current_strategy"], "ROUND_ROBIN")
            self.assertEqual(metrics["circuit_breaker_state"], "CLOSED")
            self.assertTrue(metrics["test_mode"])
            
            # Stop the balancer
            await self.balancer.stop()
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_not_running_error(self):
        """Test error when balancer is not running."""
        async def run_test():
            # Don't start the balancer
            
            # Selecting a worker should fail
            with self.assertRaises(LoadBalancerNotRunningError):
                await self.balancer.select_worker("task-1")
        
        # Run the async test
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
