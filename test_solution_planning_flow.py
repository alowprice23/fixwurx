#!/usr/bin/env python3
"""
Test module for solution_planning_flow.py

This module tests the integration between the solution planning flow
and the LLM agent system.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys
from datetime import datetime

# Import the module to test
from solution_planning_flow import SolutionPlanningFlow

class TestSolutionPlanningFlow(unittest.TestCase):
    """Test cases for SolutionPlanningFlow class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock agent system
        self.mock_agent_system = MagicMock()
        self.mock_agent_system.initialized = True
        self.mock_agent_system.initialize.return_value = True
        
        # Create sample bug detection results
        self.sample_bug_detection = {
            "success": True,
            "priority_results": {
                "priority_levels": {
                    "high": [
                        {
                            "id": "bug-001",
                            "file": "example.py",
                            "issue": {
                                "description": "Null pointer exception",
                                "severity": "high",
                                "type": "null_check",
                                "details": "Variable x is used without checking for None"
                            }
                        }
                    ],
                    "medium": [
                        {
                            "id": "bug-002",
                            "file": "utils.py",
                            "issue": {
                                "description": "Resource leak",
                                "severity": "medium",
                                "type": "resource_management",
                                "details": "File is not properly closed"
                            }
                        }
                    ]
                }
            }
        }
        
        # Create a mock solution path
        self.mock_solution_path = {
            "path_id": "path-001",
            "bug_id": "bug-001",
            "actions": [
                {"type": "analyze", "agent": "observer", "description": "Analyze bug bug-001"},
                {"type": "patch", "agent": "analyst", "description": "Patch bug bug-001"},
                {"type": "verify", "agent": "verifier", "description": "Verify bug bug-001"}
            ]
        }
        
        # Set up mock agent system behavior
        self.mock_agent_system.create_bug.return_value = {"bug_id": "bug-001"}
        self.mock_agent_system.generate_solution_paths.return_value = [self.mock_solution_path]
        self.mock_agent_system.select_solution_path.return_value = self.mock_solution_path
        
        # Create planner agent mock
        self.mock_planner = MagicMock()
        self.mock_planner.get_bug.return_value = {"bug_id": "bug-001"}
        self.mock_agent_system.get_agent.return_value = self.mock_planner
        
        # Setup mock for fix_bug
        self.mock_agent_system.fix_bug.return_value = {"success": True, "bug_id": "bug-001"}

    @patch('solution_planning_flow.get_agent_system')
    def test_initialization(self, mock_get_agent_system):
        """Test initialization of SolutionPlanningFlow."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Check that agent system was initialized
        mock_get_agent_system.assert_called_once()
        self.assertEqual(flow.agent_system, self.mock_agent_system)
        self.assertEqual(flow.processed_bugs, {})

    @patch('solution_planning_flow.get_agent_system')
    def test_run_planning_flow(self, mock_get_agent_system):
        """Test running the planning flow."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Run planning flow
        result = flow.run_planning_flow(self.sample_bug_detection)
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["bug_count"], 2)
        self.assertEqual(result["plan_count"], 2)
        self.assertEqual(len(result["plans"]), 2)
        
        # Check that agent system methods were called
        self.mock_agent_system.create_bug.assert_called()
        self.assertEqual(self.mock_agent_system.create_bug.call_count, 2)
        self.mock_agent_system.generate_solution_paths.assert_called()
        self.assertEqual(self.mock_agent_system.generate_solution_paths.call_count, 2)
        self.mock_agent_system.select_solution_path.assert_called()
        self.assertEqual(self.mock_agent_system.select_solution_path.call_count, 2)
        
        # Check processed bugs
        self.assertEqual(len(flow.processed_bugs), 2)
        self.assertIn("bug-001", flow.processed_bugs)
        self.assertIn("bug-002", flow.processed_bugs)

    @patch('solution_planning_flow.get_agent_system')
    def test_get_solution_plan(self, mock_get_agent_system):
        """Test getting a solution plan for a specific bug."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Add a processed bug
        bug_id = "bug-001"
        flow.processed_bugs[bug_id] = {"bug_id": bug_id, "solution_paths": [self.mock_solution_path]}
        
        # Get solution plan for existing bug
        result = flow.get_solution_plan(bug_id)
        self.assertIsNotNone(result)
        self.assertEqual(result["bug_id"], bug_id)
        
        # Get solution plan for new bug
        new_bug_id = "bug-003"
        result = flow.get_solution_plan(new_bug_id)
        self.assertIsNotNone(result)
        self.mock_agent_system.get_agent.assert_called_with("planner")
        self.mock_planner.get_bug.assert_called_with(new_bug_id)
        self.mock_agent_system.generate_solution_paths.assert_called_with(new_bug_id)
        self.mock_agent_system.select_solution_path.assert_called_with(new_bug_id)

    @patch('solution_planning_flow.get_agent_system')
    def test_implement_solution(self, mock_get_agent_system):
        """Test implementing a solution for a bug."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Implement solution
        bug_id = "bug-001"
        result = flow.implement_solution(bug_id)
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["bug_id"], bug_id)
        
        # Check that agent system methods were called
        self.mock_agent_system.get_agent.assert_called_with("planner")
        self.mock_planner.get_bug.assert_called_with(bug_id)
        self.mock_agent_system.fix_bug.assert_called_with(bug_id)

    @patch('solution_planning_flow.get_agent_system')
    def test_extract_bugs_from_detection_results(self, mock_get_agent_system):
        """Test extracting bugs from detection results."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Extract bugs
        bugs = flow._extract_bugs_from_detection_results(self.sample_bug_detection)
        
        # Verify results
        self.assertEqual(len(bugs), 2)
        self.assertIn("bug-001", bugs)
        self.assertIn("bug-002", bugs)
        
        # Check bug details
        bug_001 = bugs["bug-001"]
        self.assertEqual(bug_001["severity"], "high")
        self.assertEqual(bug_001["file"], "example.py")
        self.assertIn("high", bug_001["tags"])
        
        bug_002 = bugs["bug-002"]
        self.assertEqual(bug_002["severity"], "medium")
        self.assertEqual(bug_002["file"], "utils.py")
        self.assertIn("medium", bug_002["tags"])

    @patch('solution_planning_flow.get_agent_system')
    def test_map_severity(self, mock_get_agent_system):
        """Test mapping severity levels."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Test mapping different combinations
        self.assertEqual(flow._map_severity("critical", "high"), "critical")
        self.assertEqual(flow._map_severity("high", "medium"), "high")
        self.assertEqual(flow._map_severity("unknown", "low"), "low")
        self.assertEqual(flow._map_severity("CRITICAL", "medium"), "critical")  # Case insensitive

    @patch('solution_planning_flow.get_agent_system')
    def test_generate_summary(self, mock_get_agent_system):
        """Test generating summary of planning results."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Create sample results
        results = [
            {
                "bug_id": "bug-001",
                "bug_info": {"severity": "high"},
                "solution_paths": [{"path_id": "path-001"}, {"path_id": "path-002"}]
            },
            {
                "bug_id": "bug-002",
                "bug_info": {"severity": "medium"},
                "solution_paths": [{"path_id": "path-003"}]
            }
        ]
        
        # Generate summary
        summary = flow._generate_summary(results)
        
        # Verify summary
        self.assertEqual(summary["bug_count"], 2)
        self.assertEqual(summary["plan_count"], 3)
        self.assertEqual(summary["severity_counts"], {"high": 1, "medium": 1})
        self.assertEqual(summary["avg_paths_per_bug"], 1.5)

    @patch('solution_planning_flow.get_agent_system')
    def test_empty_bug_detection(self, mock_get_agent_system):
        """Test handling empty bug detection results."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Create empty bug detection results
        empty_detection = {
            "success": True,
            "priority_results": {
                "priority_levels": {}
            }
        }
        
        # Run planning flow
        result = flow.run_planning_flow(empty_detection)
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "No bugs found, no solution plans needed")
        self.assertEqual(result["plans"], [])

    @patch('solution_planning_flow.get_agent_system')
    def test_invalid_bug_detection(self, mock_get_agent_system):
        """Test handling invalid bug detection results."""
        mock_get_agent_system.return_value = self.mock_agent_system
        
        # Initialize SolutionPlanningFlow
        flow = SolutionPlanningFlow()
        
        # Create invalid bug detection results
        invalid_detection = {
            "success": False,
            "error": "Failed to detect bugs"
        }
        
        # Run planning flow
        result = flow.run_planning_flow(invalid_detection)
        
        # Verify results
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Invalid bug detection results")
        self.assertEqual(result["stage"], "input_validation")

if __name__ == '__main__':
    unittest.main()
