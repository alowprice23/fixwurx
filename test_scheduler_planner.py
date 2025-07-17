#!/usr/bin/env python3
"""
test_scheduler_planner.py
─────────────────────────
Unit tests for the enhanced scheduler with planner integration.
Tests focus on:

1. Path-aware scheduling
2. Multi-criteria prioritization
3. Planner integration
4. Secure state verification
5. Adaptive fallback strategies
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import time
import json
import hashlib
from pathlib import Path
import asyncio

from scheduler import (
    Scheduler, 
    SchedulingStrategy, 
    EntropyMetrics, 
    PathExecutionMetrics
)


class TestPathExecutionMetrics(unittest.TestCase):
    """Test the PathExecutionMetrics class."""
    
    def test_init(self):
        """Test initialization with defaults."""
        metrics = PathExecutionMetrics("path-123", "bug-456", complexity=0.75)
        
        # Check attributes
        self.assertEqual(metrics.path_id, "path-123")
        self.assertEqual(metrics.bug_id, "bug-456")
        self.assertEqual(metrics.complexity, 0.75)
        self.assertEqual(metrics.progress, 0.0)
        self.assertEqual(metrics.steps_completed, 0)
        self.assertEqual(metrics.total_steps, 0)
        self.assertFalse(metrics.success)
        self.assertFalse(metrics.fallback_used)
        
        # Check execution ID format (should be 16 chars)
        self.assertEqual(len(metrics.execution_id), 16)
    
    def test_update_progress(self):
        """Test progress updates."""
        metrics = PathExecutionMetrics("path-123", "bug-456")
        
        # Update progress
        progress = metrics.update_progress(5, 10)
        
        # Check values
        self.assertEqual(metrics.steps_completed, 5)
        self.assertEqual(metrics.total_steps, 10)
        self.assertEqual(metrics.progress, 0.5)
        self.assertEqual(progress, 0.5)
        
        # Check history recording
        self.assertEqual(len(metrics.history), 1)
        
        # Update to 100%
        metrics.update_progress(10, 10)
        self.assertEqual(metrics.progress, 1.0)
        self.assertEqual(len(metrics.history), 2)
    
    def test_mark_complete(self):
        """Test marking completion."""
        metrics = PathExecutionMetrics("path-123", "bug-456")
        metrics.update_progress(5, 10)  # 50% complete
        
        # Mark successful completion
        metrics.mark_complete(success=True)
        
        # Check values
        self.assertTrue(metrics.success)
        self.assertFalse(metrics.fallback_used)
        self.assertEqual(metrics.progress, 1.0)
        self.assertEqual(metrics.steps_completed, metrics.total_steps)
        
        # Check history
        self.assertEqual(len(metrics.history), 2)
        
        # Test with fallback
        metrics = PathExecutionMetrics("path-123", "bug-456")
        metrics.mark_complete(success=False, fallback_used=True)
        self.assertFalse(metrics.success)
        self.assertTrue(metrics.fallback_used)
    
    def test_estimate_remaining_time(self):
        """Test remaining time estimation."""
        metrics = PathExecutionMetrics("path-123", "bug-456")
        
        # No history yet
        self.assertEqual(metrics.estimate_remaining_time(), float('inf'))
        
        # Add progress history with delay to simulate progress rate
        metrics.update_progress(2, 10)  # 20%
        time.sleep(0.01)  # Small delay
        metrics.update_progress(4, 10)  # 40%
        
        # Should have a finite estimate now
        estimate = metrics.estimate_remaining_time()
        self.assertLess(estimate, float('inf'))
        self.assertGreater(estimate, 0)
        
        # Complete the progress
        metrics.mark_complete(success=True)
        self.assertEqual(metrics.estimate_remaining_time(), 0.0)
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = PathExecutionMetrics("path-123", "bug-456", complexity=0.75)
        metrics.update_progress(5, 10)
        
        # Convert to dict
        data = metrics.to_dict()
        
        # Check key fields
        self.assertEqual(data["path_id"], "path-123")
        self.assertEqual(data["bug_id"], "bug-456")
        self.assertEqual(data["complexity"], 0.75)
        self.assertEqual(data["progress"], 0.5)
        self.assertEqual(data["steps_completed"], 5)
        self.assertEqual(data["total_steps"], 10)
        self.assertIn("estimated_remaining_time", data)


class TestEntropyMetrics(unittest.TestCase):
    """Test the EntropyMetrics class."""
    
    def test_init(self):
        """Test initialization with defaults."""
        metrics = EntropyMetrics(initial_entropy=10.0, info_gain=0.5)
        
        # Check attributes
        self.assertEqual(metrics.current_entropy, 10.0)
        self.assertEqual(metrics.initial_entropy, 10.0)
        self.assertEqual(metrics.info_gain, 0.5)
        self.assertEqual(metrics.attempts, 0)
        self.assertEqual(metrics.progress_rate, 0.0)
        
        # Check planner integration attributes
        self.assertEqual(metrics.paths_attempted, 0)
        self.assertEqual(metrics.paths_completed, 0)
        self.assertEqual(metrics.successful_paths, 0)
        self.assertEqual(metrics.failed_paths, 0)
        self.assertIsNone(metrics.current_path_id)
    
    def test_update(self):
        """Test entropy updates."""
        metrics = EntropyMetrics(initial_entropy=10.0)
        
        # Update entropy
        reduction = metrics.update(8.0)
        
        # Check values
        self.assertEqual(metrics.current_entropy, 8.0)
        self.assertEqual(reduction, 2.0)
        self.assertEqual(metrics.attempts, 1)
        
        # Check history recording
        self.assertEqual(len(metrics.history), 1)
        
        # Update again
        metrics.update(5.0)
        self.assertEqual(metrics.current_entropy, 5.0)
        self.assertEqual(metrics.attempts, 2)
        self.assertEqual(len(metrics.history), 2)
    
    def test_estimate_remaining_time(self):
        """Test remaining time estimation."""
        metrics = EntropyMetrics(initial_entropy=10.0)
        
        # No progress rate yet
        self.assertEqual(metrics.estimate_remaining_time(), float('inf'))
        
        # Set a progress rate manually
        metrics.progress_rate = 2.0  # 2 bits per hour
        self.assertEqual(metrics.estimate_remaining_time(), 5.0)  # 10/2
        
        # Update entropy to check time estimate
        metrics.update(6.0)
        # Progress rate may have changed since we can't control time precisely in tests
        # So we'll manually set it again for testing
        metrics.progress_rate = 2.0
        self.assertEqual(metrics.estimate_remaining_time(), 3.0)  # 6/2
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = EntropyMetrics(initial_entropy=10.0, info_gain=0.75)
        metrics.update(8.0)
        
        # Convert to dict
        data = metrics.to_dict()
        
        # Check key fields
        self.assertEqual(data["current_entropy"], 8.0)
        self.assertEqual(data["initial_entropy"], 10.0)
        self.assertEqual(data["info_gain"], 0.75)
        self.assertEqual(data["attempts"], 1)
        self.assertIn("estimated_remaining_time", data)
        self.assertIn("explanation", data)


class TestScheduler(unittest.TestCase):
    """Test the Scheduler class with planner integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock engine and planner
        self.mock_engine = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_resource_manager = MagicMock()
        self.mock_engine.resource_manager = self.mock_resource_manager
        
        # Create scheduler
        self.scheduler = Scheduler(
            engine=self.mock_engine,
            strategy=SchedulingStrategy.PLANNER,
            planner_agent=self.mock_planner
        )
        
        # Create test bug states
        self.bug1 = MagicMock()
        self.bug1.bug_id = "bug-123"
        self.bug1.entropy = 8.0
        
        self.bug2 = MagicMock()
        self.bug2.bug_id = "bug-456"
        self.bug2.entropy = 5.0
    
    def test_initialize_bug_metrics(self):
        """Test initializing bug metrics."""
        self.scheduler.initialize_bug_metrics("bug-test", 7.5, 1.2)
        
        # Check metrics were created
        self.assertIn("bug-test", self.scheduler.bug_metrics)
        metrics = self.scheduler.bug_metrics["bug-test"]
        self.assertEqual(metrics.current_entropy, 7.5)
        self.assertEqual(metrics.info_gain, 1.2)
        
        # Check priority queue
        self.assertEqual(len(self.scheduler.priority_queue), 1)
        
        # Check failure tracking
        self.assertIn("bug-test", self.scheduler.path_failures_by_bug)
        self.assertEqual(self.scheduler.path_failures_by_bug["bug-test"], 0)
    
    def test_update_bug_entropy(self):
        """Test updating bug entropy."""
        # Initialize metrics
        self.scheduler.initialize_bug_metrics("bug-test", 10.0)
        
        # Update entropy
        reduction = self.scheduler.update_bug_entropy("bug-test", 7.0)
        
        # Check results
        self.assertEqual(reduction, 3.0)
        self.assertEqual(self.scheduler.bug_metrics["bug-test"].current_entropy, 7.0)
        self.assertEqual(self.scheduler.bug_metrics["bug-test"].attempts, 1)
        
        # Check priority queue update
        self.assertEqual(len(self.scheduler.priority_queue), 2)  # Original + update
    
    def test_planner_strategy_detection(self):
        """Test planner strategy detection."""
        # Test with planner strategy
        self.assertTrue(self.scheduler._is_planner_strategy())
        
        # Test with non-planner strategy
        scheduler = Scheduler(
            engine=self.mock_engine,
            strategy=SchedulingStrategy.ENTROPY
        )
        self.assertFalse(scheduler._is_planner_strategy())
        
        # Test with multi-criteria
        scheduler = Scheduler(
            engine=self.mock_engine,
            strategy=SchedulingStrategy.MULTI_CRITERIA
        )
        self.assertTrue(scheduler._is_planner_strategy())
    
    def test_prioritize_backlog_planner(self):
        """Test backlog prioritization with planner strategy."""
        # Add bugs to backlog
        self.scheduler.backlog = [self.bug1, self.bug2]
        
        # Initialize metrics
        self.scheduler.initialize_bug_metrics("bug-123", 8.0)
        self.scheduler.initialize_bug_metrics("bug-456", 5.0)
        
        # Set up paths for bug1
        self.scheduler.bug_to_paths["bug-123"] = ["path-1"]
        self.scheduler.active_paths["path-1"] = MagicMock()
        self.scheduler.active_paths["path-1"].metadata = {"priority": 0.8}
        
        # Set up paths for bug2
        self.scheduler.bug_to_paths["bug-456"] = ["path-2"]
        self.scheduler.active_paths["path-2"] = MagicMock()
        self.scheduler.active_paths["path-2"].metadata = {"priority": 0.5}
        
        # Prioritize backlog
        self.scheduler.prioritize_backlog()
        
        # Check order - bug1 should be first due to higher path priority
        self.assertEqual(self.scheduler.backlog[0].bug_id, "bug-123")
        self.assertEqual(self.scheduler.backlog[1].bug_id, "bug-456")
    
    def test_prioritize_backlog_multi_criteria(self):
        """Test backlog prioritization with multi-criteria strategy."""
        # Set up multi-criteria scheduler
        scheduler = Scheduler(
            engine=self.mock_engine,
            strategy=SchedulingStrategy.MULTI_CRITERIA,
            planner_agent=self.mock_planner,
            criteria_weights={
                "entropy": 0.7,           # More heavily weighted to entropy
                "age": 0.1,
                "path_complexity": 0.1,
                "progress_rate": 0.05,
                "resources_required": 0.05
            }
        )
        
        # Add bugs to backlog
        scheduler.backlog = [self.bug1, self.bug2]
        
        # Initialize metrics
        scheduler.initialize_bug_metrics("bug-123", 8.0)
        scheduler.initialize_bug_metrics("bug-456", 5.0)
        
        # Prioritize backlog
        scheduler.prioritize_backlog()
        
        # Check order - bug2 should be first due to lower entropy
        self.assertEqual(scheduler.backlog[0].bug_id, "bug-456")
        self.assertEqual(scheduler.backlog[1].bug_id, "bug-123")
    
    def test_select_path_for_bug(self):
        """Test path selection for a bug."""
        # Set up paths for a bug
        self.scheduler.bug_to_paths["bug-123"] = ["path-1", "path-2", "path-3"]
        
        # Set up path metadata
        self.scheduler.active_paths["path-1"] = MagicMock()
        self.scheduler.active_paths["path-1"].metadata = {"priority": 0.3, "complexity": 0.8}
        
        self.scheduler.active_paths["path-2"] = MagicMock()
        self.scheduler.active_paths["path-2"].metadata = {"priority": 0.9, "complexity": 0.5}
        
        self.scheduler.active_paths["path-3"] = MagicMock()
        self.scheduler.active_paths["path-3"].metadata = {"priority": 0.6, "complexity": 0.2}
        
        # Test priority strategy (default)
        selected = self.scheduler._select_path_for_bug("bug-123")
        self.assertEqual(selected, "path-2")  # Highest priority
        
        # Test complexity strategy
        self.scheduler.path_selection_strategy = "complexity"
        selected = self.scheduler._select_path_for_bug("bug-123")
        self.assertEqual(selected, "path-3")  # Lowest complexity
        
        # Test with failed paths
        self.scheduler.failed_paths.add("path-2")
        self.scheduler.path_selection_strategy = "priority"
        selected = self.scheduler._select_path_for_bug("bug-123")
        self.assertEqual(selected, "path-3")  # Second highest priority since path-2 failed
    
    def test_create_engine_for_bug(self):
        """Test creating an engine for a bug."""
        # Set up paths for a bug
        self.scheduler.bug_to_paths["bug-123"] = ["path-1", "path-2"]
        
        # Set up path metadata
        self.scheduler.active_paths["path-1"] = MagicMock()
        self.scheduler.active_paths["path-1"].metadata = {"priority": 0.8, "complexity": 0.5}
        self.scheduler.active_paths["path-2"] = MagicMock()
        self.scheduler.active_paths["path-2"].metadata = {"priority": 0.5, "complexity": 0.3}
        
        # Initialize metrics
        self.scheduler.initialize_bug_metrics("bug-123", 8.0)
        
        # Create engine
        result = self.scheduler.create_engine_for_bug(self.bug1)
        
        # Check engine was created properly
        self.assertEqual(result, self.mock_engine.return_value)
        
        # Check path execution metrics were created
        self.assertIn("path-1", self.scheduler.executing_paths)
        
        # Check path ID was stored in bug metrics
        self.assertEqual(self.scheduler.bug_metrics["bug-123"].current_path_id, "path-1")
    
    def test_record_path_failure(self):
        """Test recording path failures."""
        # Set up test path and metrics
        self.scheduler.bug_to_paths["bug-123"] = ["path-1", "path-2"]
        self.scheduler.active_paths["path-1"] = MagicMock()
        self.scheduler.active_paths["path-1"].metadata = {"priority": 0.8}
        
        # Initialize metrics
        self.scheduler.initialize_bug_metrics("bug-123", 8.0)
        self.scheduler.bug_metrics["bug-123"].current_path_id = "path-1"
        
        # Create execution metrics
        self.scheduler.executing_paths["path-1"] = PathExecutionMetrics(
            path_id="path-1",
            bug_id="bug-123",
            complexity=0.5
        )
        
        # Record failure
        self.scheduler.record_path_failure("bug-123", "path-1")
        
        # Check failure was recorded
        self.assertIn("path-1", self.scheduler.failed_paths)
        self.assertEqual(self.scheduler.path_failures_by_bug["bug-123"], 1)
        
        # Check execution metrics were moved to history
        self.assertIn("path-1", self.scheduler.path_history)
        self.assertNotIn("path-1", self.scheduler.executing_paths)
        
        # Check bug metrics were updated
        self.assertEqual(self.scheduler.bug_metrics["bug-123"].failed_paths, 1)
        self.assertIsNone(self.scheduler.bug_metrics["bug-123"].current_path_id)
        
        # Check planner was notified
        self.mock_planner.record_path_result.assert_called_once()
    
    def test_adaptive_fallback(self):
        """Test adaptive fallback when multiple paths fail."""
        # Set up scheduler with adaptive strategy
        scheduler = Scheduler(
            engine=self.mock_engine,
            strategy=SchedulingStrategy.PLANNER_ADAPTIVE,
            planner_agent=self.mock_planner
        )
        
        # Add bugs to backlog
        scheduler.backlog = [self.bug1, self.bug2]
        
        # Initialize metrics
        scheduler.initialize_bug_metrics("bug-123", 8.0)
        scheduler.initialize_bug_metrics("bug-456", 5.0)
        
        # Record failures for bug1
        scheduler.path_failures_by_bug["bug-123"] = 3  # Reached threshold
        
        # Record minimal failures for bug2
        scheduler.path_failures_by_bug["bug-456"] = 1
        
        # Prioritize backlog
        scheduler.prioritize_backlog()
        
        # Check order - bug2 should be first due to fewer failures
        self.assertEqual(scheduler.backlog[0].bug_id, "bug-456")
        self.assertEqual(scheduler.backlog[1].bug_id, "bug-123")
    
    def test_state_integrity(self):
        """Test state integrity verification."""
        # Initialize some state
        self.scheduler.initialize_bug_metrics("bug-123", 8.0)
        self.scheduler.bug_to_paths["bug-123"] = ["path-1"]
        self.scheduler.active_paths["path-1"] = MagicMock()
        
        # Get initial hash
        initial_hash = self.scheduler.last_state_hash
        
        # Verify integrity
        result = self.scheduler.verify_state_integrity()
        
        # Check result
        self.assertTrue(result["verified"])
        self.assertEqual(result["hash"], initial_hash)
        
        # Make a state change
        self.scheduler.update_bug_entropy("bug-123", 7.0)
        
        # Hash should have changed
        self.assertNotEqual(self.scheduler.last_state_hash, initial_hash)
        
        # Verify again
        result = self.scheduler.verify_state_integrity()
        self.assertTrue(result["verified"])


class TestSchedulerAsync(unittest.IsolatedAsyncioTestCase):
    """Test the async methods of the Scheduler class."""
    
    async def asyncSetUp(self):
        """Set up test fixtures."""
        # Mock engine and planner
        self.mock_engine = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_resource_manager = MagicMock()
        self.mock_engine.resource_manager = self.mock_resource_manager
        
        # Set free resources
        self.mock_resource_manager.free = 9  # Enough for 3 bugs (3 agents each)
        
        # Create scheduler
        self.scheduler = Scheduler(
            engine=self.mock_engine,
            strategy=SchedulingStrategy.HYBRID,
            planner_agent=self.mock_planner
        )
        
        # Create test bug states
        self.bug1 = MagicMock()
        self.bug1.bug_id = "bug-123"
        self.bug1.entropy = 8.0
        
        self.bug2 = MagicMock()
        self.bug2.bug_id = "bug-456"
        self.bug2.entropy = 5.0
        
        self.bug3 = MagicMock()
        self.bug3.bug_id = "bug-789"
        self.bug3.entropy = 7.0
        
        # Add bugs to backlog
        self.scheduler.backlog = [self.bug1, self.bug2, self.bug3]
        
        # Initialize metrics
        self.scheduler.initialize_bug_metrics("bug-123", 8.0)
        self.scheduler.initialize_bug_metrics("bug-456", 5.0)
        self.scheduler.initialize_bug_metrics("bug-789", 7.0)
    
    async def test_get_active_bugs(self):
        """Test getting active bugs asynchronously."""
        # Get active bugs
        active_bugs = await self.scheduler.get_active_bugs()
        
        # Check results - should get all 3 bugs from backlog
        self.assertEqual(len(active_bugs), 3)
        
        # Check they were moved from backlog to active
        self.assertEqual(len(self.scheduler.backlog), 0)
        self.assertEqual(len(self.scheduler.active_bugs), 3)
        
        # Check ordering based on hybrid strategy (entropy + age)
        # bug2 (lowest entropy) should be first
        self.assertEqual(active_bugs[0].bug_id, "bug-456")
    
    async def test_get_active_bugs_with_resource_constraints(self):
        """Test getting active bugs with limited resources."""
        # Reduce available resources
        self.mock_resource_manager.free = 3  # Only enough for 1 bug
        
        # Get active bugs
        active_bugs = await self.scheduler.get_active_bugs()
        
        # Check results - should only get 1 bug
        self.assertEqual(len(active_bugs), 1)
        
        # Check ordering - should get bug2 (lowest entropy)
        self.assertEqual(active_bugs[0].bug_id, "bug-456")
        
        # 2 bugs should remain in backlog
        self.assertEqual(len(self.scheduler.backlog), 2)
        self.assertEqual(len(self.scheduler.active_bugs), 1)


if __name__ == "__main__":
    unittest.main()
