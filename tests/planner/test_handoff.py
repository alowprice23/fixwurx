"""
tests/planner/test_handoff.py
─────────────────────────────
Test suite for the agent handoff functionality.

Tests:
1. Agent handoff protocol implementation
2. Handoff context preservation
3. Task transition between agents
4. Integration with planner agent
"""

import os
import pytest
import unittest
import json
import tempfile
from unittest.mock import MagicMock, patch

# Import components to test
from agent_handoff import AgentHandoff
from planner_agent import PlannerAgent
from agent_memory import AgentMemory
from triangulation_engine import TriangulationEngine
from data_structures import BugState


@pytest.mark.handoff
class TestAgentHandoffCore(unittest.TestCase):
    """Test core agent handoff functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create handoff manager
        self.handoff = AgentHandoff()
        
        # Create test agents
        self.agent_ids = ["observer-1", "analyst-1", "verifier-1"]
    
    def test_initialization(self):
        """Test handoff initialization."""
        self.assertIsNotNone(self.handoff)
        self.assertEqual(len(self.handoff.active_handoffs), 0)
    
    def test_create_handoff(self):
        """Test creating a handoff between agents."""
        # Create a handoff context
        context = {
            "bug_id": "bug-123",
            "description": "Test bug description",
            "files": ["file1.py", "file2.py"],
            "steps_to_reproduce": ["Step 1", "Step 2"]
        }
        
        # Create a handoff
        handoff_id = self.handoff.create_handoff(
            source_id=self.agent_ids[0],
            target_id=self.agent_ids[1],
            context=context,
            task="analyze_bug"
        )
        
        # Check that the handoff was created
        self.assertIsNotNone(handoff_id)
        self.assertIn(handoff_id, self.handoff.active_handoffs)
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["source_id"], self.agent_ids[0])
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["target_id"], self.agent_ids[1])
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["task"], "analyze_bug")
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["context"], context)
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["status"], "pending")
    
    def test_accept_handoff(self):
        """Test accepting a handoff."""
        # Create a handoff
        context = {"bug_id": "bug-123"}
        handoff_id = self.handoff.create_handoff(
            source_id=self.agent_ids[0],
            target_id=self.agent_ids[1],
            context=context,
            task="analyze_bug"
        )
        
        # Accept the handoff
        result = self.handoff.accept_handoff(handoff_id, self.agent_ids[1])
        
        # Check that the handoff was accepted
        self.assertTrue(result)
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["status"], "accepted")
    
    def test_complete_handoff(self):
        """Test completing a handoff."""
        # Create a handoff
        context = {"bug_id": "bug-123"}
        handoff_id = self.handoff.create_handoff(
            source_id=self.agent_ids[0],
            target_id=self.agent_ids[1],
            context=context,
            task="analyze_bug"
        )
        
        # Accept the handoff
        self.handoff.accept_handoff(handoff_id, self.agent_ids[1])
        
        # Complete the handoff with results
        results = {
            "analysis": "Bug caused by null reference",
            "fix_suggestion": "Add null check"
        }
        result = self.handoff.complete_handoff(handoff_id, self.agent_ids[1], results)
        
        # Check that the handoff was completed
        self.assertTrue(result)
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["status"], "completed")
        self.assertEqual(self.handoff.active_handoffs[handoff_id]["results"], results)
    
    def test_get_handoff_status(self):
        """Test getting handoff status."""
        # Create a handoff
        context = {"bug_id": "bug-123"}
        handoff_id = self.handoff.create_handoff(
            source_id=self.agent_ids[0],
            target_id=self.agent_ids[1],
            context=context,
            task="analyze_bug"
        )
        
        # Get the status
        status = self.handoff.get_handoff_status(handoff_id)
        
        # Check the status
        self.assertEqual(status, "pending")
        
        # Accept and check status again
        self.handoff.accept_handoff(handoff_id, self.agent_ids[1])
        status = self.handoff.get_handoff_status(handoff_id)
        self.assertEqual(status, "accepted")
        
        # Complete and check status again
        self.handoff.complete_handoff(handoff_id, self.agent_ids[1], {"result": "success"})
        status = self.handoff.get_handoff_status(handoff_id)
        self.assertEqual(status, "completed")
    
    def test_get_handoff_results(self):
        """Test getting handoff results."""
        # Create a handoff
        context = {"bug_id": "bug-123"}
        handoff_id = self.handoff.create_handoff(
            source_id=self.agent_ids[0],
            target_id=self.agent_ids[1],
            context=context,
            task="analyze_bug"
        )
        
        # Accept and complete the handoff
        self.handoff.accept_handoff(handoff_id, self.agent_ids[1])
        results = {"analysis": "Bug analysis complete"}
        self.handoff.complete_handoff(handoff_id, self.agent_ids[1], results)
        
        # Get the results
        handoff_results = self.handoff.get_handoff_results(handoff_id)
        
        # Check the results
        self.assertEqual(handoff_results, results)
    
    def test_chain_handoffs(self):
        """Test chaining multiple handoffs."""
        # Create first handoff (observer -> analyst)
        context1 = {"bug_id": "bug-123", "description": "Test bug"}
        handoff_id1 = self.handoff.create_handoff(
            source_id=self.agent_ids[0],  # observer
            target_id=self.agent_ids[1],  # analyst
            context=context1,
            task="analyze_bug"
        )
        
        # Accept and complete first handoff
        self.handoff.accept_handoff(handoff_id1, self.agent_ids[1])
        results1 = {"analysis": "Bug analysis complete", "fix_suggestion": "Add null check"}
        self.handoff.complete_handoff(handoff_id1, self.agent_ids[1], results1)
        
        # Create second handoff (analyst -> verifier) with results from first
        context2 = {
            "bug_id": "bug-123",
            "analysis": results1["analysis"],
            "fix": results1["fix_suggestion"]
        }
        handoff_id2 = self.handoff.create_handoff(
            source_id=self.agent_ids[1],  # analyst
            target_id=self.agent_ids[2],  # verifier
            context=context2,
            task="verify_fix"
        )
        
        # Accept and complete second handoff
        self.handoff.accept_handoff(handoff_id2, self.agent_ids[2])
        results2 = {"verification": "Fix verified successfully"}
        self.handoff.complete_handoff(handoff_id2, self.agent_ids[2], results2)
        
        # Check that both handoffs completed successfully
        self.assertEqual(self.handoff.get_handoff_status(handoff_id1), "completed")
        self.assertEqual(self.handoff.get_handoff_status(handoff_id2), "completed")
        
        # Check that the context was properly chained
        self.assertIn("fix", self.handoff.active_handoffs[handoff_id2]["context"])
        self.assertEqual(
            self.handoff.active_handoffs[handoff_id2]["context"]["fix"], 
            results1["fix_suggestion"]
        )


@pytest.mark.handoff
class TestPlannerHandoffIntegration(unittest.TestCase):
    """Test integration between planner and agent handoff system."""
    
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
                "fallback-threshold": 0.3
            }
        }
        
        # Initialize agent memory
        self.agent_memory = AgentMemory()
        
        # Initialize handoff manager
        self.handoff = AgentHandoff()
        
        # Initialize planner agent with handoff support
        self.planner = PlannerAgent(self.config, self.agent_memory, handoff_manager=self.handoff)
        
        # Initialize triangulation engine
        self.engine = TriangulationEngine(config=self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_planner_creates_handoff(self):
        """Test that the planner can create a handoff."""
        # Register agents
        self.planner.register_agent("planner-1", "planner")
        self.planner.register_agent("observer-1", "observer", parent_id="planner-1")
        
        # Create a bug state
        bug_state = BugState(
            bug_id="bug-123",
            title="Test Bug",
            description="This is a test bug",
            severity="medium",
            status="new"
        )
        
        # Create a path with an action that requires handoff
        path = {
            "path_id": "path-1",
            "bug_id": "bug-123",
            "actions": [
                {
                    "type": "observe",
                    "agent": "observer-1",
                    "description": "Observe the bug",
                    "requires_handoff": True
                }
            ]
        }
        
        # Mock the planner to return this path
        with patch.object(self.planner, 'select_best_path', return_value=path):
            # Create a handoff for the action
            handoff_id = self.planner.create_action_handoff(
                path["path_id"],
                path["actions"][0],
                bug_state
            )
            
            # Check that the handoff was created
            self.assertIsNotNone(handoff_id)
            self.assertIn(handoff_id, self.handoff.active_handoffs)
            self.assertEqual(self.handoff.active_handoffs[handoff_id]["source_id"], "planner-1")
            self.assertEqual(self.handoff.active_handoffs[handoff_id]["target_id"], "observer-1")
            self.assertEqual(self.handoff.active_handoffs[handoff_id]["status"], "pending")
            
            # Check that the context includes the bug details
            self.assertIn("bug_id", self.handoff.active_handoffs[handoff_id]["context"])
            self.assertEqual(
                self.handoff.active_handoffs[handoff_id]["context"]["bug_id"],
                bug_state.bug_id
            )
    
    def test_planner_handles_handoff_results(self):
        """Test that the planner can handle handoff results."""
        # Register agents
        self.planner.register_agent("planner-1", "planner")
        self.planner.register_agent("observer-1", "observer", parent_id="planner-1")
        
        # Create a bug state
        bug_state = BugState(
            bug_id="bug-123",
            title="Test Bug",
            description="This is a test bug",
            severity="medium",
            status="new"
        )
        
        # Create a path with an action that requires handoff
        path = {
            "path_id": "path-1",
            "bug_id": "bug-123",
            "actions": [
                {
                    "type": "observe",
                    "agent": "observer-1",
                    "description": "Observe the bug",
                    "requires_handoff": True
                }
            ]
        }
        
        # Create a handoff
        handoff_id = self.handoff.create_handoff(
            source_id="planner-1",
            target_id="observer-1",
            context={"bug_id": bug_state.bug_id, "description": bug_state.description},
            task="observe_bug"
        )
        
        # Accept and complete the handoff
        self.handoff.accept_handoff(handoff_id, "observer-1")
        results = {"observation": "Bug observed successfully", "files_affected": ["file1.py", "file2.py"]}
        self.handoff.complete_handoff(handoff_id, "observer-1", results)
        
        # Have the planner process the results
        with patch.object(self.planner, 'select_best_path', return_value=path):
            processed = self.planner.process_handoff_result(handoff_id, path["path_id"])
            
            # Check that the results were processed
            self.assertTrue(processed)
            
            # Check that the results were stored
            self.assertIn(handoff_id, self.planner.handoff_results)
            self.assertEqual(self.planner.handoff_results[handoff_id], results)


if __name__ == "__main__":
    unittest.main()
