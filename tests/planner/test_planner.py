"""
tests/planner/test_planner.py
────────────────────────────
Test suite for the planner agent functionality.

Tests:
1. Solution path generation
2. Path selection
3. Fallback mechanisms
4. Integration with the triangulation engine
5. Metrics collection
"""

import os
import pytest
import unittest
import json
from unittest.mock import MagicMock, patch
import tempfile
import time

# Import components to test
from planner_agent import PlannerAgent
from triangulation_engine import TriangulationEngine
from agent_memory import AgentMemory
from data_structures import BugState, FamilyTree


@pytest.mark.planner
class TestPlannerCore(unittest.TestCase):
    """Test core planner functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the family tree
        self.temp_dir = tempfile.TemporaryDirectory()
        self.family_tree_path = os.path.join(self.temp_dir.name, "family_tree.json")
        
        # Create a minimal config
        self.config = {
            "planner": {
                "enabled": True,
                "family-tree-path": self.family_tree_path,
                "solutions-per-bug": 3,
                "max-path-depth": 5,
                "fallback-threshold": 0.3,
                "prompts": {
                    "system": "You are the Planner Agent.",
                    "task": "Generate a solution path for: {bug_description}",
                    "fallback": "Generate a fallback for bug: {bug_id}"
                }
            }
        }
        
        # Initialize agent memory
        self.agent_memory = AgentMemory()
        
        # Initialize planner agent
        self.planner = PlannerAgent(self.config, self.agent_memory)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test planner initialization."""
        self.assertIsNotNone(self.planner)
        self.assertEqual(self.planner.config, self.config)
        self.assertEqual(self.planner.agent_memory, self.agent_memory)
        self.assertEqual(self.planner.family_tree_path, self.family_tree_path)
    
    def test_generate_solution_paths(self):
        """Test solution path generation."""
        # Create a bug state
        bug_state = BugState(
            bug_id="bug-123",
            title="Test Bug",
            description="This is a test bug",
            severity="medium",
            status="new"
        )
        
        # Mock the LLM response for solution paths
        with patch.object(self.planner, '_generate_path_via_llm', return_value={
            "path_id": "path-1",
            "bug_id": "bug-123",
            "actions": [
                {"type": "analyze", "agent": "observer", "description": "Analyze the bug"},
                {"type": "fix", "agent": "analyst", "description": "Fix the bug"},
                {"type": "verify", "agent": "verifier", "description": "Verify the fix"}
            ],
            "success_probability": 0.8,
            "complexity": 0.5,
            "fallback_strategy": "retry_with_different_approach"
        }):
            paths = self.planner.generate_solution_paths(bug_state)
            
            # Check that paths were generated
            self.assertIsNotNone(paths)
            self.assertEqual(len(paths), 1)  # Since we're mocking a single response
            
            # Check path structure
            path = paths[0]
            self.assertEqual(path["path_id"], "path-1")
            self.assertEqual(path["bug_id"], "bug-123")
            self.assertEqual(len(path["actions"]), 3)
            self.assertEqual(path["success_probability"], 0.8)
            self.assertEqual(path["complexity"], 0.5)
    
    def test_select_best_path(self):
        """Test path selection logic."""
        # Create paths with different probabilities
        paths = [
            {
                "path_id": "path-1",
                "bug_id": "bug-123",
                "success_probability": 0.5,
                "complexity": 0.7
            },
            {
                "path_id": "path-2",
                "bug_id": "bug-123",
                "success_probability": 0.8,  # Highest probability
                "complexity": 0.5
            },
            {
                "path_id": "path-3",
                "bug_id": "bug-123",
                "success_probability": 0.6,
                "complexity": 0.3
            }
        ]
        
        # Mock the stored paths
        self.planner.bug_paths = {"bug-123": paths}
        
        # Select best path
        best_path = self.planner.select_best_path("bug-123")
        
        # Should select path-2 with highest success probability
        self.assertEqual(best_path["path_id"], "path-2")
    
    def test_activate_fallback(self):
        """Test fallback activation."""
        # Create paths including a fallback
        paths = [
            {
                "path_id": "path-1",
                "bug_id": "bug-123",
                "success_probability": 0.8,
                "complexity": 0.5,
                "is_fallback": False
            },
            {
                "path_id": "path-2",
                "bug_id": "bug-123",
                "success_probability": 0.6,
                "complexity": 0.3,
                "is_fallback": True  # This is a fallback path
            }
        ]
        
        # Mock the stored paths
        self.planner.bug_paths = {"bug-123": paths}
        
        # Activate fallback for path-1
        fallback = self.planner.activate_fallback("path-1")
        
        # Should select path-2 as the fallback
        self.assertEqual(fallback["path_id"], "path-2")
    
    def test_record_path_result(self):
        """Test recording path execution results."""
        # Create a test path
        path_id = "path-1"
        metrics = {
            "duration": 10.5,
            "actions_completed": 3,
            "resources_used": 2
        }
        
        # Record a successful result
        self.planner.record_path_result(path_id, True, metrics)
        
        # Check that the result was recorded
        self.assertIn(path_id, self.planner.path_results)
        result = self.planner.path_results[path_id]
        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"], metrics)
        
        # Record a failed result
        self.planner.record_path_result(path_id, False, metrics)
        
        # Check that the result was updated
        self.assertFalse(self.planner.path_results[path_id]["success"])
    
    def test_get_metrics(self):
        """Test metrics collection."""
        # Add some test paths and results
        self.planner.bug_paths = {
            "bug-1": [{"path_id": "path-1"}, {"path_id": "path-2"}],
            "bug-2": [{"path_id": "path-3"}]
        }
        
        self.planner.path_results = {
            "path-1": {"success": True, "metrics": {"duration": 5}},
            "path-2": {"success": False, "metrics": {"duration": 10}},
            "path-3": {"success": True, "metrics": {"duration": 7}}
        }
        
        # Get metrics
        metrics = self.planner.get_metrics()
        
        # Check metrics
        self.assertEqual(metrics["total_paths"], 3)
        self.assertEqual(metrics["successful_paths"], 2)
        self.assertEqual(metrics["failed_paths"], 1)
        self.assertAlmostEqual(metrics["success_rate"], 2/3)


@pytest.mark.planner
class TestPlannerTriangulationIntegration(unittest.TestCase):
    """Test integration between planner and triangulation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the family tree
        self.temp_dir = tempfile.TemporaryDirectory()
        self.family_tree_path = os.path.join(self.temp_dir.name, "family_tree.json")
        
        # Create a minimal config
        self.config = {
            "planner": {
                "enabled": True,
                "family-tree-path": self.family_tree_path,
                "solutions-per-bug": 2,
                "max-path-depth": 3,
                "fallback-threshold": 0.3
            }
        }
        
        # Initialize triangulation engine with planner support
        self.engine = TriangulationEngine(config=self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_engine_has_planner(self):
        """Test that the engine has a planner."""
        self.assertIsNotNone(self.engine.planner)
        self.assertTrue(self.engine.planner_enabled)
    
    def test_solution_path_assignment(self):
        """Test assigning a solution path to a bug."""
        # Create a test path
        test_path = {
            "path_id": "test-path-1",
            "actions": [
                {"type": "analyze", "agent": "observer", "description": "Analyze"},
                {"type": "fix", "agent": "analyst", "description": "Fix"}
            ],
            "success_probability": 0.9,
            "complexity": 0.4
        }
        
        # Set the path in the engine
        success = self.engine.set_solution_path(test_path)
        
        # Check that the path was set
        self.assertTrue(success)
        self.assertEqual(self.engine.current_solution_path, test_path)
        
        # Check that metrics were initialized
        self.assertIn("test-path-1", self.engine.path_metrics)
        metrics = self.engine.path_metrics["test-path-1"]
        self.assertEqual(metrics["total_actions"], 2)
        self.assertEqual(metrics["success_probability"], 0.9)
        self.assertEqual(metrics["complexity"], 0.4)
    
    def test_path_execution_tracking(self):
        """Test tracking path execution."""
        # This would require more complex mocking of the state machine
        # and execution flow, which is beyond the scope of this test.
        # For now, we'll just verify the interfaces exist.
        self.assertTrue(hasattr(self.engine, 'set_solution_path'))
        self.assertTrue(hasattr(self.engine, 'get_solution_paths'))
        self.assertTrue(hasattr(self.engine, 'get_path_metrics'))
        self.assertTrue(hasattr(self.engine, 'get_all_path_metrics'))
        self.assertTrue(hasattr(self.engine, 'get_agent_assignments'))


if __name__ == "__main__":
    unittest.main()
