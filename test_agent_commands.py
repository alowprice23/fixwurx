#!/usr/bin/env python3
"""
test_agent_commands.py
─────────────────────
Test suite for the agent_commands module.

This test verifies:
1. Agent system initialization
2. Command registration
3. Basic command functionality
4. Bug management
5. Integration with planner agent
"""

import os
import json
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import agent_commands
from data_structures import BugState

class TestAgentCommands(unittest.TestCase):
    """Test case for the agent_commands module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a test config
        self.config = {
            "agent_system": {
                "enabled": True,
                "solutions-per-bug": 2,
                "max-path-depth": 3
            }
        }
        
        # Reset global variables in agent_commands
        agent_commands._planner = None
        agent_commands._memory = None
        agent_commands._meta = None
        agent_commands._config = {}
        agent_commands._bugs = {}
        agent_commands._initialized = False
    
    def tearDown(self):
        """Clean up after test."""
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('agent_commands.PlannerAgent')
    @patch('agent_commands.AgentMemory')
    @patch('agent_commands.MetaAgent')
    def test_initialization(self, mock_meta, mock_memory, mock_planner):
        """Test that the agent system initializes correctly."""
        # Set up mocks
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_planner_instance = MagicMock()
        mock_planner_instance.get_family_relationships.return_value = {"relationships": {}}
        mock_planner.return_value = mock_planner_instance
        
        mock_meta_instance = MagicMock()
        mock_meta.return_value = mock_meta_instance
        
        # Initialize
        result = agent_commands.initialize(self.config)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(agent_commands._initialized)
        
        # Check that memory, planner, and meta were created
        mock_memory.assert_called_once()
        mock_planner.assert_called_once()
        mock_meta.assert_called_once()
        
        # Check that planner agent was initialized with config
        planner_config = mock_planner.call_args[0][0]
        self.assertIn("planner", planner_config)
        self.assertEqual(planner_config["planner"]["enabled"], True)
        
        # Check that meta agent was initialized with config
        meta_config = mock_meta.call_args[0][0]
        self.assertIn("enabled", meta_config)
        self.assertEqual(meta_config["enabled"], True)
        
        # Check that meta agent oversight was started
        mock_meta_instance.start_oversight.assert_called_once()
    
    def test_register(self):
        """Test command registration."""
        commands = agent_commands.register()
        
        # Check that all expected commands are registered
        self.assertIn("agent", commands)
        self.assertIn("plan", commands)
        self.assertIn("observe", commands)
        self.assertIn("analyze", commands)
        self.assertIn("verify", commands)
        self.assertIn("bug", commands)
        self.assertIn("meta", commands)
        
        # Check that commands are callable
        for cmd in commands.values():
            self.assertTrue(callable(cmd))
    
    @patch('agent_commands._initialized', True)
    @patch('agent_commands._planner')
    def test_agent_status_command(self, mock_planner):
        """Test agent status command."""
        # Set up mock
        mock_planner.get_metrics.return_value = {
            "enabled": True,
            "active_bugs": 2,
            "active_paths": 3,
            "family_tree_size": 4,
            "paths_generated": 5,
            "successful_fixes": 1,
            "failed_fixes": 0
        }
        
        # Create test bugs
        agent_commands._bugs = {
            "bug-1": BugState("bug-1", title="Test Bug 1", status="new"),
            "bug-2": BugState("bug-2", title="Test Bug 2", status="analyzed")
        }
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.agent_status_command("")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check output
        self.assertIn("Agent System Status:", output)
        self.assertIn("Active Bugs: 2", output)
        self.assertIn("Active Paths: 3", output)
        self.assertIn("Family Tree Size: 4", output)
        self.assertIn("Paths Generated: 5", output)
        self.assertIn("Successful Fixes: 1", output)
        self.assertIn("Failed Fixes: 0", output)
        self.assertIn("bug-1", output)
        self.assertIn("bug-2", output)
    
    @patch('agent_commands._initialized', True)
    def test_bug_create_command(self):
        """Test bug creation command."""
        # Clear bugs dictionary
        agent_commands._bugs = {}
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.bug_create_command("test-bug-1 Test Bug Title")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check bug was created
        self.assertIn("test-bug-1", agent_commands._bugs)
        bug = agent_commands._bugs["test-bug-1"]
        self.assertEqual(bug.title, "Test Bug Title")
        self.assertEqual(bug.status, "new")
        
        # Check output
        self.assertIn("Bug test-bug-1 created", output)
    
    @patch('agent_commands._initialized', True)
    def test_bug_list_command(self):
        """Test bug list command."""
        # Create test bugs
        agent_commands._bugs = {
            "bug-1": BugState("bug-1", title="Test Bug 1", status="new"),
            "bug-2": BugState("bug-2", title="Test Bug 2", status="analyzed"),
            "bug-3": BugState("bug-3", title="Test Bug 3", status="patched")
        }
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        # Test listing all bugs
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.bug_list_command("")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check output
        self.assertIn("bug-1", output)
        self.assertIn("bug-2", output)
        self.assertIn("bug-3", output)
        
        # Test filtering by status
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.bug_list_command("analyzed")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check output
        self.assertNotIn("bug-1", output)
        self.assertIn("bug-2", output)
        self.assertNotIn("bug-3", output)
    
    @patch('agent_commands._initialized', True)
    @patch('agent_commands._planner')
    def test_plan_generate_command(self, mock_planner):
        """Test plan generation command."""
        # Create a test bug
        bug = BugState("test-bug", title="Test Bug", status="new")
        agent_commands._bugs = {"test-bug": bug}
        
        # Set up mock
        mock_path1 = MagicMock()
        mock_path1.path_id = "path-1"
        mock_path1.metadata = {"priority": 0.8, "entropy": 0.5, "estimated_time": 10}
        mock_path1.actions = [
            {"type": "analyze", "description": "Analyze bug"},
            {"type": "patch", "description": "Generate patch"}
        ]
        mock_path1.fallbacks = []
        
        mock_path2 = MagicMock()
        mock_path2.path_id = "path-2"
        mock_path2.metadata = {"priority": 0.6, "entropy": 0.3, "estimated_time": 8}
        mock_path2.actions = [
            {"type": "analyze", "description": "Quick analysis"},
            {"type": "patch", "description": "Simple patch"}
        ]
        mock_path2.fallbacks = [
            {"type": "simplify", "description": "Simplified approach"}
        ]
        
        mock_planner.generate_solution_paths.return_value = [mock_path1, mock_path2]
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.plan_generate_command("test-bug")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check planner was called
        mock_planner.generate_solution_paths.assert_called_once_with(bug)
        
        # Check output
        self.assertIn("Generated 2 solution paths", output)
        self.assertIn("path-1", output)
        self.assertIn("path-2", output)
        self.assertIn("Priority: 0.80", output)
        self.assertIn("Priority: 0.60", output)
        self.assertIn("Analyze bug", output)
        self.assertIn("Simplified approach", output)
    
    @patch('agent_commands._initialized', True)
    @patch('agent_commands.specialized_agents.ObserverAgent')
    def test_observe_command(self, mock_observer):
        """Test observer command."""
        # Create observer instance
        mock_observer_instance = MagicMock()
        mock_observer.return_value = mock_observer_instance
        
        # Create a test bug
        bug = BugState("test-bug", title="Test Bug", status="new")
        agent_commands._bugs = {"test-bug": bug}
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.observe_command("analyze test-bug")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check output
        self.assertIn("Analyzing bug test-bug", output)
        self.assertIn("Observer agent activated", output)
        
        # Check bug status was updated
        self.assertEqual(bug.status, "analyzed")
        self.assertEqual(len(bug.phase_history), 1)
        self.assertEqual(bug.phase_history[0]["name"], "ANALYZE")
    
    @patch('agent_commands._initialized', True)
    @patch('agent_commands._meta')
    def test_meta_status_command(self, mock_meta):
        """Test meta status command."""
        # Set up mock
        mock_meta.get_metrics.return_value = {
            "agent_count": 4,
            "coordination_events": 5,
            "conflict_resolutions": 2,
            "optimizations": 3,
            "oversight_cycles": 10,
            "meta_insights_generated": 2
        }
        mock_meta.enabled = True
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.meta_status_command("")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check output
        self.assertIn("Meta Agent Status:", output)
        self.assertIn("Enabled: True", output)
        self.assertIn("Agent Count: 4", output)
        self.assertIn("Coordination Events: 5", output)
        self.assertIn("Conflict Resolutions: 2", output)
        self.assertIn("Optimizations: 3", output)
        self.assertIn("Oversight Cycles: 10", output)
        self.assertIn("Meta Insights Generated: 2", output)
    
    @patch('agent_commands._initialized', True)
    @patch('agent_commands._meta')
    def test_agent_network_command(self, mock_meta):
        """Test agent network command."""
        # Set up mock
        mock_meta.get_agent_network.return_value = {
            "nodes": [
                {"id": "planner", "type": "planner", "status": "active", "last_activity": time.time()},
                {"id": "observer", "type": "observer", "status": "active", "last_activity": time.time()},
                {"id": "analyst", "type": "analyst", "status": "active", "last_activity": time.time()},
                {"id": "verifier", "type": "verifier", "status": "active", "last_activity": time.time()}
            ],
            "edges": [
                {"source": "planner", "target": "observer", "strength": 0.5},
                {"source": "planner", "target": "analyst", "strength": 0.7},
                {"source": "planner", "target": "verifier", "strength": 0.3}
            ],
            "timestamp": time.time()
        }
        mock_meta.visualize_agent_network.return_value = ".triangulum/meta/agent_network_test.json"
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.agent_network_command("")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check output
        self.assertIn("Agent Network Visualization:", output)
        self.assertIn("Nodes: 4", output)
        self.assertIn("Edges: 3", output)
        self.assertIn("planner (planner)", output)
        self.assertIn("observer (observer)", output)
        self.assertIn("analyst (analyst)", output)
        self.assertIn("verifier (verifier)", output)
        self.assertIn("planner <--> observer", output)
        self.assertIn("planner <--> analyst", output)
        self.assertIn("planner <--> verifier", output)
    
    @patch('agent_commands._initialized', True)
    @patch('agent_commands._meta')
    def test_meta_insights_command(self, mock_meta):
        """Test meta insights command."""
        # Set up mock
        mock_meta.generate_insight.return_value = {
            "id": "insight-123456789",
            "type": "resource_optimization",
            "timestamp": time.time(),
            "confidence": 0.8,
            "description": "Simulated resource_optimization insight",
            "details": {
                "agent_count": 4,
                "activity_count": 0,
                "coordination_events": 5,
                "conflict_resolutions": 2
            },
            "recommendations": [
                {
                    "action": "optimize_resources",
                    "confidence": 0.8,
                    "expected_impact": 0.6
                },
                {
                    "action": "increase_coordination",
                    "confidence": 0.7,
                    "expected_impact": 0.5
                }
            ]
        }
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.meta_insights_command("")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check output
        self.assertIn("New Meta Agent Insight:", output)
        self.assertIn("Type: resource_optimization", output)
        self.assertIn("Confidence: 0.80", output)
        self.assertIn("Simulated resource_optimization insight", output)
        self.assertIn("optimize_resources", output)
        self.assertIn("increase_coordination", output)
    
    @patch('agent_commands._initialized', True)
    @patch('agent_commands._meta')
    def test_meta_coordinate_command(self, mock_meta):
        """Test meta coordinate command."""
        # Set up mock
        mock_meta.coordinate_agents.return_value = {
            "success": True,
            "plan_id": "task-123",
            "agent_count": 3,
            "step_count": 3
        }
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        with io.StringIO() as buf, redirect_stdout(buf):
            result = agent_commands.meta_coordinate_command("observer,analyst,verifier task-123 bug-fix")
            output = buf.getvalue()
        
        # Check result
        self.assertEqual(result, 0)
        
        # Check that meta agent was called with correct arguments
        mock_meta.coordinate_agents.assert_called_with(
            ["observer", "analyst", "verifier"], 
            "task-123", 
            "bug-fix"
        )
        
        # Check output
        self.assertIn("Created coordination plan for task task-123", output)
        self.assertIn("Agents: 3", output)
        self.assertIn("Steps: 3", output)


if __name__ == "__main__":
    unittest.main()
