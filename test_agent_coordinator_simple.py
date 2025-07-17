#!/usr/bin/env python3
"""
test_agent_coordinator_simple.py
────────────────────────────────
Simplified test suite for the AgentCoordinator module.
"""

import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import textwrap

from agent_coordinator import _Artefacts, HandoffStatus
from state_machine import Phase
from data_structures import BugState


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name="mock"):
        self.name = name
        self.prompts = []
        self.responses = {}
        
        # Default responses
        self.responses["INITIALIZATION"] = "INITIALIZED"
        self.responses["Initialize the family tree"] = json.dumps({"status": "success"})
    
    async def ask(self, prompt):
        """Mock ask method."""
        self.prompts.append(prompt)
        
        # Return predefined response if available
        for key, response in self.responses.items():
            if key in prompt:
                return response
        
        # Default response with JSON status
        if "HANDOFF:" in prompt:
            return json.dumps({"status": "SUCCESS", "message": "Handoff processed"})
        
        return "OK"


class MockBug:
    """Mock bug for testing."""
    
    def __init__(self, id="test-bug-123"):
        self.id = id
        self.phase = Phase.REPRO
        self.timer = 0


class MockEngine:
    """Mock triangulation engine for testing."""
    
    def __init__(self, bug=None):
        self.bugs = [bug or MockBug()]


class TestArtefacts(unittest.TestCase):
    """Test case for the _Artefacts class."""
    
    def test_record_action_time(self):
        """Test recording action times."""
        art = _Artefacts()
        
        # Record start
        art.record_action_start("test")
        self.assertIn("test_start", art.action_times)
        
        # Patch time.time to return a later time when called the second time
        original_time = art.action_times["test_start"]
        
        # Manually override the time calculation
        art.action_times["test_start"] = original_time - 1.0  # Make it 1 second earlier
        
        # Record end
        art.record_action_end("test")
        self.assertIn("test_duration", art.action_times)
        self.assertGreater(art.action_times["test_duration"], 0)
    
    def test_record_handoff(self):
        """Test recording handoffs."""
        art = _Artefacts()
        
        # Record handoff
        art.record_handoff("agent1", "agent2", "success")
        
        # Check handoff was recorded
        self.assertIn("agent1_to_agent2", art.handoff_counts)
        self.assertEqual(art.handoff_counts["agent1_to_agent2"], 1)
        
        # Check history entry
        self.assertEqual(len(art.execution_history), 1)
        self.assertEqual(art.execution_history[0]["type"], "handoff")
        self.assertEqual(art.execution_history[0]["from_agent"], "agent1")
        self.assertEqual(art.execution_history[0]["to_agent"], "agent2")
        self.assertEqual(art.execution_history[0]["status"], "success")
    
    def test_record_error(self):
        """Test recording errors."""
        art = _Artefacts()
        
        # Record error
        art.record_error("test_error", "Test error details")
        
        # Check error was recorded
        self.assertIn("test_error", art.error_counts)
        self.assertEqual(art.error_counts["test_error"], 1)
        
        # Check history entry
        self.assertEqual(len(art.execution_history), 1)
        self.assertEqual(art.execution_history[0]["type"], "error")
        self.assertEqual(art.execution_history[0]["error_type"], "test_error")
        self.assertEqual(art.execution_history[0]["details"], "Test error details")
    
    def test_get_metrics(self):
        """Test getting metrics."""
        art = _Artefacts()
        
        # Add some metrics
        art.record_action_start("test")
        art.record_action_end("test")
        art.record_handoff("agent1", "agent2", "success")
        art.record_error("test_error", "Test error details")
        
        # Get metrics
        metrics = art.get_metrics()
        
        # Check metrics
        self.assertIn("total_duration", metrics)
        self.assertIn("action_times", metrics)
        self.assertIn("handoff_counts", metrics)
        self.assertIn("error_counts", metrics)
        self.assertIn("test_duration", metrics["action_times"])
        self.assertEqual(metrics["handoff_counts"]["agent1_to_agent2"], 1)
        self.assertEqual(metrics["error_counts"]["test_error"], 1)
    
    def test_get_current_action(self):
        """Test getting the current action."""
        art = _Artefacts()
        
        # No action when no paths
        self.assertIsNone(art.get_current_action())
        
        # Add a path
        art.solution_paths = [
            {
                "path_id": "test-path-1",
                "actions": [
                    {"type": "test", "description": "Test action"}
                ]
            }
        ]
        art.current_path_id = "test-path-1"
        
        # Get action
        action = art.get_current_action()
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "test")
        self.assertEqual(action["description"], "Test action")
        
        # Advance to next action (which doesn't exist)
        art.advance_to_next_action()
        self.assertIsNone(art.get_current_action())
    
    def test_context(self):
        """Test context management."""
        art = _Artefacts()
        
        # Add to context
        art.add_to_context("test_key", "test_value")
        
        # Get from context
        value = art.get_from_context("test_key")
        self.assertEqual(value, "test_value")
        
        # Get with default
        value = art.get_from_context("missing_key", "default_value")
        self.assertEqual(value, "default_value")


# Create a simplified test case without actual agent initialization
class TestAgentCoordinatorSimple(unittest.TestCase):
    """Simplified test case for the AgentCoordinator class that doesn't need real agents."""
    
    def test_artefacts_integration(self):
        """Test integration between artefacts and metrics."""
        # Create artefacts
        art = _Artefacts()
        
        # Add metrics data
        art.record_handoff("observer", "analyst", "success")
        art.record_error("test_error", "Testing error handling")
        art.add_to_context("test_key", "test_value")
        
        # Add a path
        art.solution_paths = [
            {
                "path_id": "test-path-1",
                "actions": [
                    {"type": "analyze", "agent": "observer", "description": "Test analysis"}
                ]
            }
        ]
        art.current_path_id = "test-path-1"
        
        # Check current action
        action = art.get_current_action()
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "analyze")
        
        # Advance action and check metrics
        art.advance_to_next_action()
        metrics = art.get_metrics()
        
        # Verify metrics contain expected data
        self.assertEqual(metrics["handoff_counts"]["observer_to_analyst"], 1)
        self.assertEqual(metrics["error_counts"]["test_error"], 1)
        self.assertEqual(art.get_from_context("test_key"), "test_value")


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    print_header("AGENT COORDINATOR SIMPLE TEST SUITE")
    unittest.main(verbosity=2)
    print_header("TEST SUITE COMPLETE")
