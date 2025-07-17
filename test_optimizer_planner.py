#!/usr/bin/env python3
"""
test_optimizer_planner.py
─────────────────────────
Unit tests for the enhanced optimizer with planner integration.
Tests focus on:

1. Path complexity awareness
2. Planner callback integration
3. Security features with secure hashing
4. Parameter adaptation based on planner metrics
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import json
import time
from pathlib import Path

from optimizer import EnhancedAdaptiveOptimizer, PlannerPathData
from planner_agent import PlannerAgent


class TestPlannerPathData(unittest.TestCase):
    """Test the PlannerPathData class."""
    
    def test_from_path(self):
        """Test creating a PlannerPathData object from a path dictionary."""
        # Create a test path with actions and dependencies
        path = {
            "path_id": "test-path-123",
            "bug_id": "bug-456",
            "actions": [
                {"type": "analyze", "agent": "observer"},
                {"type": "patch", "agent": "analyst"},
                {"type": "verify", "agent": "verifier"}
            ],
            "dependencies": ["dep1", "dep2"],
            "success": True,
            "fallback_used": False,
            "execution_time": 10.5
        }
        
        # Create PlannerPathData from the path
        path_data = PlannerPathData.from_path(path)
        
        # Verify attributes
        self.assertEqual(path_data.path_id, "test-path-123")
        self.assertEqual(path_data.bug_id, "bug-456")
        self.assertTrue(path_data.success)
        self.assertFalse(path_data.fallback_used)
        self.assertEqual(path_data.execution_time, 10.5)
        self.assertEqual(path_data.num_actions, 3)
        self.assertEqual(path_data.num_dependencies, 2)
        
        # Verify complexity calculation
        # action_complexity = min(3 / 10.0, 1.0) = 0.3
        # dependency_complexity = min(2 / 5.0, 1.0) = 0.4
        # complexity = (0.3 + 0.4) / 2.0 = 0.35
        self.assertAlmostEqual(path_data.complexity, 0.35)
    
    def test_from_path_empty(self):
        """Test creating a PlannerPathData object from an empty path."""
        # Create a minimal path
        path = {
            "path_id": "empty-path",
            "bug_id": "bug-empty"
        }
        
        # Create PlannerPathData from the path
        path_data = PlannerPathData.from_path(path)
        
        # Verify attributes
        self.assertEqual(path_data.path_id, "empty-path")
        self.assertEqual(path_data.bug_id, "bug-empty")
        self.assertFalse(path_data.success)
        self.assertFalse(path_data.fallback_used)
        self.assertEqual(path_data.execution_time, 0.0)
        self.assertEqual(path_data.num_actions, 0)
        self.assertEqual(path_data.num_dependencies, 0)
        self.assertEqual(path_data.complexity, 0.0)


class TestOptimizerPlannerIntegration(unittest.TestCase):
    """Test integration between the optimizer and planner."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for optimizer state
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temp_dir.name) / "optimizer_state.json"
        
        # Create mock objects
        self.mock_engine = MagicMock()
        self.mock_planner = MagicMock(spec=PlannerAgent)
        self.mock_scheduler = MagicMock()
        
        # Set up planner metrics
        self.mock_planner.get_metrics.return_value = {
            "paths_generated": 10,
            "fallbacks_used": 2,
            "successful_fixes": 7,
            "failed_fixes": 3
        }
        
        # Create the optimizer
        self.optimizer = EnhancedAdaptiveOptimizer(
            engine=self.mock_engine,
            planner=self.mock_planner,
            scheduler=self.mock_scheduler,
            storage_path=self.state_file,
            update_freq=5  # Lower for testing
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_planner_integration(self):
        """Test basic planner integration."""
        # Verify planner was set
        self.assertEqual(self.optimizer.planner, self.mock_planner)
        
        # Verify exploration_rate parameter was created
        self.assertIn("exploration_rate", self.optimizer.parameters)
        param_state = self.optimizer.parameters["exploration_rate"]
        self.assertEqual(param_state.target_object, self.mock_planner)
    
    def test_handle_path_completion(self):
        """Test handling path completion events."""
        # Create a test path
        path_data = {
            "path_id": "test-path-123",
            "bug_id": "bug-456",
            "actions": [{"type": "action"} for _ in range(5)],
            "dependencies": ["dep1", "dep2"],
            "success": True,
            "fallback_used": False
        }
        
        # Handle path completion
        self.optimizer.handle_path_completion(path_data)
        
        # Verify path was added to history
        self.assertIn("test-path-123", self.optimizer.path_history)
        self.assertIn("bug-456", self.optimizer.bug_paths)
        self.assertIn("test-path-123", self.optimizer.bug_paths["bug-456"])
        
        # Verify complexity stats were updated
        self.assertEqual(self.optimizer.path_complexity_stats["total_paths"], 1)
        self.assertGreater(self.optimizer.path_complexity_stats["mean"], 0.0)
        
        # Add a second path for the same bug
        path_data2 = {
            "path_id": "test-path-456",
            "bug_id": "bug-456",
            "actions": [{"type": "action"} for _ in range(10)],  # More complex
            "dependencies": ["dep1", "dep2", "dep3", "dep4"],
            "success": False,
            "fallback_used": True
        }
        
        # Handle second path completion
        self.optimizer.handle_path_completion(path_data2)
        
        # Verify second path was added
        self.assertIn("test-path-456", self.optimizer.path_history)
        self.assertEqual(len(self.optimizer.bug_paths["bug-456"]), 2)
        
        # Verify complexity stats were updated
        self.assertEqual(self.optimizer.path_complexity_stats["total_paths"], 2)
        
        # The second path should have increased the mean complexity
        expected_complexity1 = (min(5 / 10.0, 1.0) + min(2 / 5.0, 1.0)) / 2.0  # First path
        expected_complexity2 = (min(10 / 10.0, 1.0) + min(4 / 5.0, 1.0)) / 2.0  # Second path
        expected_mean = (expected_complexity1 + expected_complexity2) / 2.0
        
        self.assertAlmostEqual(self.optimizer.path_complexity_stats["mean"], expected_mean, places=3)
    
    def test_reward_calculation_with_planner(self):
        """Test reward calculation with planner metrics."""
        # Add a path to history
        path_data = {
            "path_id": "test-path-123",
            "bug_id": "bug-456",
            "actions": [{"type": "action"} for _ in range(5)],
            "dependencies": ["dep1", "dep2"],
            "success": True,
            "fallback_used": False
        }
        
        # Create path data and add to history
        path_obj = PlannerPathData.from_path(path_data)
        self.optimizer.path_history["test-path-123"] = path_obj
        
        # Create a metric with planner data
        metric = {
            "success": True,
            "mean_tokens": 1000,
            "path_id": "test-path-123",
            "bug_id": "bug-456",
            "planner": {
                "successful_fixes": 8,
                "failed_fixes": 2,
                "fallbacks_used": 1,
                "paths_generated": 10
            }
        }
        
        # Calculate reward
        reward = self.optimizer._calculate_reward(metric, "bug-456")
        
        # Base reward for success with tokens < 1500 is 1.0
        # Planner reward includes:
        # - Path complexity bonus (path_obj.complexity * PATH_COMPLEXITY_WEIGHT)
        # - Success rate bonus (8 / 10 * 0.2)
        # - Fallback rate penalty (1 / 10 * 0.2)
        base_reward = 1.0
        expected_complexity_bonus = path_obj.complexity * 0.3  # PATH_COMPLEXITY_WEIGHT is 0.3
        expected_success_bonus = 0.8 * 0.2
        expected_fallback_penalty = 0.1 * 0.2
        
        expected_reward = base_reward + expected_complexity_bonus + expected_success_bonus - expected_fallback_penalty
        
        self.assertAlmostEqual(reward, expected_reward, places=3)
    
    def test_parameter_adaptation(self):
        """Test parameter adaptation based on planner metrics."""
        # Set up a high fallback rate scenario
        self.mock_planner.get_metrics.return_value = {
            "paths_generated": 10,
            "fallbacks_used": 5,  # 50% fallback rate should increase exploration
            "successful_fixes": 4,
            "failed_fixes": 6
        }
        
        # Add some paths with high complexity
        for i in range(10):
            path_data = {
                "path_id": f"complex-path-{i}",
                "bug_id": f"bug-{i}",
                "actions": [{"type": "action"} for _ in range(10)],  # Max complexity
                "dependencies": ["dep1", "dep2", "dep3", "dep4", "dep5"],  # Max complexity
                "success": i % 2 == 0,  # 50% success rate
                "fallback_used": i % 3 == 0  # 33% fallback rate
            }
            self.optimizer.handle_path_completion(path_data)
        
        # Add metrics to trigger parameter update
        for i in range(6):  # update_freq + 1 to ensure update happens
            self.optimizer.push_metric({
                "success": i % 2 == 0,
                "mean_tokens": 1000,
                "bug_id": f"bug-{i}"
            })
        
        # Verify exploration rate was increased due to high fallback rate and complexity
        exploration_param = self.optimizer.parameters["exploration_rate"]
        self.assertGreater(exploration_param.current_value, 0.1)  # Initial value was 0.1
        
        # Verify the mock planner's exploration_rate attribute was updated
        expected_value = exploration_param.current_value
        self.mock_planner.exploration_rate = expected_value
    
    def test_callbacks(self):
        """Test callback registration and execution."""
        # Create a mock callback
        callback_mock = MagicMock()
        
        # Register the callback
        result = self.optimizer.register_planner_callback("on_parameters_updated", callback_mock)
        self.assertTrue(result)
        
        # Add metrics to trigger parameter update
        for i in range(6):  # update_freq + 1 to ensure update happens
            self.optimizer.push_metric({
                "success": True,
                "mean_tokens": 1000,
                "bug_id": f"bug-{i}"
            })
        
        # Manually trigger parameter update to test callback
        self.optimizer._update_parameters()
        
        # Verify callback was called
        self.assertTrue(callback_mock.called)
    
    def test_state_integrity(self):
        """Test state integrity verification."""
        # Add some data to create a valid state
        path_data = {
            "path_id": "test-path-123",
            "bug_id": "bug-456",
            "actions": [{"type": "action"} for _ in range(5)],
            "dependencies": ["dep1", "dep2"],
            "success": True,
            "fallback_used": False
        }
        self.optimizer.handle_path_completion(path_data)
        
        # Verify integrity
        integrity_report = self.optimizer.verify_state_integrity()
        
        # Check the report
        self.assertTrue(integrity_report["verified"])
        self.assertIn("full_hash", integrity_report)
        self.assertIn("details", integrity_report)
        
        # Test summary includes integrity
        summary = self.optimizer.summary()
        self.assertIn("integrity", summary)
        self.assertTrue(summary["integrity"])


if __name__ == "__main__":
    unittest.main()
