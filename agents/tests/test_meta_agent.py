"""
tests/test_meta_agent.py
───────────────────────
Unit tests for the meta_agent.py module with focus on planner integration
and prompt-weight adaptation features.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import time
from typing import Dict, Any, List

# Import the MetaAgent class
from agents.core.meta_agent import MetaAgent
from data_structures import BugState, PlannerPath, FamilyTree, HistoryEntry

class TestMetaAgent(unittest.TestCase):
    """Tests for the MetaAgent class."""
    
    def setUp(self):
        """Set up test fixtures for each test method."""
        # Create mock agents
        self.mock_planner = MagicMock()
        self.mock_observer = MagicMock()
        self.mock_analyst = MagicMock()
        self.mock_verifier = MagicMock()
        self.mock_optimizer = MagicMock()
        
        # Setup LLM configs for agents
        for agent in [self.mock_observer, self.mock_analyst, self.mock_verifier]:
            agent.llm_config = {
                "config_list": [
                    {
                        "temperature": 0.2,
                        "max_tokens": 2048
                    }
                ]
            }
        
        # Setup active paths in planner
        self.mock_planner.active_paths = {}
        
        # Create a test path
        self.test_path_id = "test-path-123"
        self.test_path = PlannerPath(
            path_id=self.test_path_id,
            bug_id="bug-123",
            actions=[
                {"type": "analyze", "agent": "observer"},
                {"type": "patch", "agent": "analyst"},
                {"type": "verify", "agent": "verifier"}
            ],
            dependencies=["dep1", "dep2"],
            fallbacks=[{"type": "simplify"}],
            metadata={"priority": 0.8}
        )
        self.mock_planner.active_paths[self.test_path_id] = self.test_path
        
        # Setup planner metrics return value
        self.mock_planner.get_metrics.return_value = {
            "paths_generated": 10,
            "fallbacks_used": 2,
            "successful_fixes": 5,
            "failed_fixes": 3
        }
        
        # Create MetaAgent instance with planner
        self.meta_agent_with_planner = MetaAgent(
            planner=self.mock_planner,
            observer=self.mock_observer,
            analyst=self.mock_analyst,
            verifier=self.mock_verifier,
            optimiser_cb=self.mock_optimizer
        )
        
        # Create MetaAgent instance without planner
        self.meta_agent_without_planner = MetaAgent(
            planner=None,
            observer=self.mock_observer,
            analyst=self.mock_analyst,
            verifier=self.mock_verifier,
            optimiser_cb=self.mock_optimizer
        )
    
    def test_initialization(self):
        """Test MetaAgent initialization with and without planner."""
        # Test with planner
        self.assertEqual(self.meta_agent_with_planner.planner, self.mock_planner)
        self.assertEqual(self.meta_agent_with_planner.observer, self.mock_observer)
        self.assertEqual(self.meta_agent_with_planner.analyst, self.mock_analyst)
        self.assertEqual(self.meta_agent_with_planner.verifier, self.mock_verifier)
        self.assertEqual(self.meta_agent_with_planner.optimiser_cb, self.mock_optimizer)
        
        # Test without planner
        self.assertIsNone(self.meta_agent_without_planner.planner)
        
        # Test internal state
        self.assertEqual(self.meta_agent_with_planner._bugs_seen, 0)
        self.assertEqual(len(self.meta_agent_with_planner._hist), 0)
        self.assertEqual(self.meta_agent_with_planner._last_adaptation, 0)
        self.assertEqual(len(self.meta_agent_with_planner._path_histories), 0)
    
    def test_record_result_without_path(self):
        """Test recording a result without path information."""
        # Record a result without path_id
        self.meta_agent_with_planner.record_result(
            bug_id="bug-123",
            success=True,
            tokens_used=1000
        )
        
        # Check that history was updated
        self.assertEqual(self.meta_agent_with_planner._bugs_seen, 1)
        self.assertEqual(len(self.meta_agent_with_planner._hist), 1)
        
        # Check the recorded history entry
        entry = self.meta_agent_with_planner._hist[0]
        self.assertTrue(entry.success)
        self.assertEqual(entry.tokens, 1000)
        self.assertEqual(entry.path_complexity, 0.0)
        self.assertIsNone(entry.path_id)
        
        # Verify planner was not called
        self.mock_planner.record_path_result.assert_not_called()
    
    def test_record_result_with_path(self):
        """Test recording a result with path information."""
        # Record a result with path_id
        self.meta_agent_with_planner.record_result(
            bug_id="bug-123",
            success=True,
            tokens_used=1000,
            path_id=self.test_path_id
        )
        
        # Check that history was updated
        self.assertEqual(self.meta_agent_with_planner._bugs_seen, 1)
        self.assertEqual(len(self.meta_agent_with_planner._hist), 1)
        
        # Check the recorded history entry
        entry = self.meta_agent_with_planner._hist[0]
        self.assertTrue(entry.success)
        self.assertEqual(entry.tokens, 1000)
        self.assertGreater(entry.path_complexity, 0.0)  # Should be calculated from path
        self.assertEqual(entry.path_id, self.test_path_id)
        
        # Verify path history was created
        self.assertIn(self.test_path_id, self.meta_agent_with_planner._path_histories)
        self.assertEqual(len(self.meta_agent_with_planner._path_histories[self.test_path_id]), 1)
        
        # Verify planner was called
        self.mock_planner.record_path_result.assert_called_once_with(
            self.test_path_id, 
            True,
            {"tokens_used": 1000}
        )
    
    def test_path_complexity_calculation(self):
        """Test calculation of path complexity."""
        # Record a result with path_id
        self.meta_agent_with_planner.record_result(
            bug_id="bug-123",
            success=True,
            tokens_used=1000,
            path_id=self.test_path_id
        )
        
        # Get the calculated complexity
        entry = self.meta_agent_with_planner._hist[0]
        
        # The formula is (action_complexity + dependency_complexity) / 2.0 where:
        # action_complexity = min(len(path.actions) / 10.0, 1.0)
        # dependency_complexity = min(len(path.dependencies) / 5.0, 1.0)
        expected_action_complexity = min(len(self.test_path.actions) / 10.0, 1.0)
        expected_dependency_complexity = min(len(self.test_path.dependencies) / 5.0, 1.0)
        expected_complexity = (expected_action_complexity + expected_dependency_complexity) / 2.0
        
        # Verify the complexity is calculated correctly
        self.assertAlmostEqual(entry.path_complexity, expected_complexity, places=5)
    
    def test_maybe_update_not_enough_history(self):
        """Test that update is skipped when not enough history."""
        # Record just one result, which is less than WINDOW // 2
        self.meta_agent_with_planner.record_result(
            bug_id="bug-123",
            success=True,
            tokens_used=1000
        )
        
        # Try to update
        self.meta_agent_with_planner.maybe_update()
        
        # Verify no changes were made to agent configs
        for agent in [self.mock_observer, self.mock_analyst, self.mock_verifier]:
            self.assertEqual(agent.llm_config["config_list"][0]["temperature"], 0.2)
            self.assertEqual(agent.llm_config["config_list"][0]["max_tokens"], 2048)
        
        # Verify optimizer was not called
        self.mock_optimizer.assert_not_called()
    
    def test_adaptation_frequency(self):
        """Test that adaptation respects frequency setting."""
        # Record enough results to meet history requirement
        for i in range(15):  # More than WINDOW // 2
            self.meta_agent_with_planner.record_result(
                bug_id=f"bug-{i}",
                success=True,
                tokens_used=1000
            )
        
        # Set last_adaptation to skip update
        self.meta_agent_with_planner._last_adaptation = self.meta_agent_with_planner._bugs_seen
        
        # Try to update
        self.meta_agent_with_planner.maybe_update()
        
        # Verify no changes were made to agent configs (adaptation should be skipped)
        for agent in [self.mock_observer, self.mock_analyst, self.mock_verifier]:
            self.assertEqual(agent.llm_config["config_list"][0]["temperature"], 0.2)
            self.assertEqual(agent.llm_config["config_list"][0]["max_tokens"], 2048)
    
    def test_path_aware_adaptation(self):
        """Test adaptation that incorporates path complexity."""
        # Create a complex path
        complex_path_id = "complex-path-123"
        complex_path = PlannerPath(
            path_id=complex_path_id,
            bug_id="bug-456",
            actions=[{"type": "action"} for _ in range(10)],  # 10 actions
            dependencies=["dep1", "dep2", "dep3", "dep4", "dep5"],  # 5 dependencies
            fallbacks=[],
            metadata={}
        )
        self.mock_planner.active_paths[complex_path_id] = complex_path
        
        # Record results with complex path to meet history and adaptation requirements
        for i in range(20):  # Fill the history window
            self.meta_agent_with_planner.record_result(
                bug_id=f"bug-{i}",
                success=False,  # Low success rate to trigger temperature increase
                tokens_used=1000,
                path_id=complex_path_id
            )
        
        # Force update
        self.meta_agent_with_planner._last_adaptation = 0
        self.meta_agent_with_planner.maybe_update()
        
        # Verify changes were made to agent configs
        # 1. Temperature should increase due to low success rate
        # 2. Additional increase for planner due to path complexity
        for agent in [self.mock_observer, self.mock_analyst, self.mock_verifier]:
            self.assertGreater(agent.llm_config["config_list"][0]["temperature"], 0.2)
        
        # Planner doesn't have llm_config, but check the metrics were sent to optimizer
        self.mock_optimizer.assert_called_once()
        metrics_payload = self.mock_optimizer.call_args[0][0]
        
        # Verify metrics include complexity and planner information
        self.assertIn("mean_complexity", metrics_payload)
        self.assertGreater(metrics_payload["mean_complexity"], 0.0)
        self.assertIn("planner", metrics_payload)
    
    def test_get_path_performance(self):
        """Test getting performance metrics for a specific path."""
        # Record multiple results for the same path
        for i in range(3):
            self.meta_agent_with_planner.record_result(
                bug_id="bug-123",
                success=i % 2 == 0,  # Alternate success/failure
                tokens_used=1000 + i * 100,
                path_id=self.test_path_id
            )
        
        # Get path performance
        performance = self.meta_agent_with_planner.get_path_performance(self.test_path_id)
        
        # Verify performance metrics
        self.assertEqual(performance["path_id"], self.test_path_id)
        self.assertEqual(performance["attempts"], 3)
        self.assertAlmostEqual(performance["success_rate"], 2/3, places=5)
        self.assertAlmostEqual(performance["avg_tokens"], (1000 + 1100 + 1200) / 3, places=5)
        self.assertIn("last_attempt", performance)
        self.assertIn("complexity", performance)
        
        # Test with unknown path
        unknown_performance = self.meta_agent_with_planner.get_path_performance("unknown-path")
        self.assertIn("error", unknown_performance)
    
    def test_get_metrics(self):
        """Test getting all metrics from the meta agent."""
        # Record some results
        for i in range(5):
            self.meta_agent_with_planner.record_result(
                bug_id=f"bug-{i}",
                success=i % 2 == 0,
                tokens_used=1000 + i * 100,
                path_id=self.test_path_id if i % 2 == 0 else None
            )
        
        # Get metrics
        metrics = self.meta_agent_with_planner.get_metrics()
        
        # Verify basic metrics
        self.assertEqual(metrics["bugs_seen"], 5)
        self.assertEqual(metrics["history_size"], 5)
        self.assertEqual(metrics["paths_tracked"], 1)
        
        # Verify success rate and token metrics
        self.assertIn("success_rate", metrics)
        self.assertIn("avg_tokens", metrics)
        self.assertIn("avg_complexity", metrics)
        
        # Verify planner metrics
        self.assertIn("planner", metrics)
        self.assertEqual(metrics["planner"], self.mock_planner.get_metrics.return_value)
        
        # Verify agent configs
        self.assertIn("agent_configs", metrics)
        self.assertEqual(len(metrics["agent_configs"]), 3)  # observer, analyst, verifier
        for agent_name in ["observer", "analyst", "verifier"]:
            self.assertIn(agent_name, metrics["agent_configs"])
            self.assertIn("temperature", metrics["agent_configs"][agent_name])
            self.assertIn("max_tokens", metrics["agent_configs"][agent_name])
    
    def test_repr(self):
        """Test the string representation of MetaAgent."""
        # Record some results
        for i in range(3):
            self.meta_agent_with_planner.record_result(
                bug_id=f"bug-{i}",
                success=True,
                tokens_used=1000
            )
        
        # Get string representation
        repr_str = repr(self.meta_agent_with_planner)
        
        # Verify it contains expected information
        self.assertIn("bugs=3", repr_str)
        self.assertIn("recent_succ=1.00", repr_str)
        self.assertIn("history=3", repr_str)
        self.assertIn("planner_enabled=True", repr_str)
        
        # Test without planner
        repr_str_no_planner = repr(self.meta_agent_without_planner)
        self.assertIn("planner_enabled=False", repr_str_no_planner)


if __name__ == "__main__":
    unittest.main()
