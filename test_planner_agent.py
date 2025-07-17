#!/usr/bin/env python3
"""
test_planner_agent.py
─────────────────────
Test suite for the PlannerAgent module.

This test verifies:
1. Solution path generation for bugs
2. Path selection and prioritization
3. Family tree management
4. Fallback strategy handling
5. Integration with agent_memory
6. Metrics reporting
"""

import os
import json
import time
import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from planner_agent import PlannerAgent
from agent_memory import AgentMemory
from data_structures import BugState, PlannerPath, FamilyTree


class TestPlannerAgent(unittest.TestCase):
    """Test case for the PlannerAgent class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for the family tree
        self.temp_dir = Path(tempfile.mkdtemp())
        self.family_tree_path = self.temp_dir / "family_tree.json"
        
        # Create a test config
        self.config = {
            "planner": {
                "enabled": True,
                "family-tree-path": str(self.family_tree_path),
                "solutions-per-bug": 3,
                "max-path-depth": 5,
                "fallback-threshold": 0.3
            }
        }
        
        # Create memory with test path
        self.memory = AgentMemory(
            mem_path=self.temp_dir / "memory.json",
            kv_path=self.temp_dir / "kv_store.json",
            compressed_path=self.temp_dir / "compressed_store.json",
            family_tree_path=self.family_tree_path
        )
        
        # Create the planner agent
        self.planner = PlannerAgent(self.config, self.memory)
        
        # Create a test bug
        self.test_bug = BugState(
            bug_id="test-bug-123",
            title="Test Bug",
            description="This is a test bug for the planner agent",
            severity="high",
            status="new",
            metadata={"entropy": 0.7, "complexity": "high"},
            tags={"test", "planner"}
        )
    
    def tearDown(self):
        """Clean up after test."""
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that the planner agent initializes correctly."""
        self.assertTrue(self.planner.enabled)
        self.assertEqual(self.planner.family_tree_path, str(self.family_tree_path))
        self.assertEqual(self.planner.solutions_per_bug, 3)
        self.assertEqual(self.planner.max_path_depth, 5)
        self.assertEqual(self.planner.fallback_threshold, 0.3)
        self.assertIsInstance(self.planner.family_tree, FamilyTree)
    
    def test_solution_path_generation(self):
        """Test the generation of solution paths for a bug."""
        # Generate paths
        paths = self.planner.generate_solution_paths(self.test_bug)
        
        # Check paths were created
        self.assertEqual(len(paths), 3)
        self.assertIsInstance(paths[0], PlannerPath)
        
        # Check paths have expected properties
        for path in paths:
            self.assertEqual(path.bug_id, "test-bug-123")
            self.assertGreater(len(path.actions), 0)
            self.assertIn("priority", path.metadata)
            self.assertIn("entropy", path.metadata)
            self.assertIn("estimated_time", path.metadata)
        
        # Check the bug was updated with solution paths
        self.assertEqual(len(self.test_bug.planner_solutions), 3)
        
        # Check the metrics were updated
        self.assertEqual(self.planner.metrics["paths_generated"], 3)
        self.assertGreater(self.planner.metrics["avg_path_length"], 0)
    
    def test_best_path_selection(self):
        """Test the selection of the best solution path."""
        # Generate paths
        self.planner.generate_solution_paths(self.test_bug)
        
        # Get the best path
        best_path = self.planner.select_best_path(self.test_bug.bug_id)
        
        # Check that a path was selected
        self.assertIsNotNone(best_path)
        
        # Check it's a valid PlannerPath
        self.assertIsInstance(best_path, PlannerPath)
        self.assertEqual(best_path.bug_id, self.test_bug.bug_id)
        
        # Check that the best path has the highest priority
        for path_id, path in self.planner.active_paths.items():
            if path_id != best_path.path_id:
                self.assertLessEqual(
                    path.metadata.get("priority", 0),
                    best_path.metadata.get("priority", 0)
                )
    
    def test_agent_registration(self):
        """Test agent registration in the family tree."""
        # Register agents
        result1 = self.planner.register_agent("observer-1", "observer")
        result2 = self.planner.register_agent("analyst-1", "analyst")
        result3 = self.planner.register_agent("verifier-1", "verifier")
        
        # Check registration succeeded
        self.assertTrue(result1)
        self.assertTrue(result2)
        self.assertTrue(result3)
        
        # Check the family tree
        tree_data = self.planner.get_family_relationships()
        
        # Check agents exist in the tree
        relationships = tree_data.get("relationships", {})
        self.assertIn("observer-1", relationships)
        self.assertIn("analyst-1", relationships)
        self.assertIn("verifier-1", relationships)
        
        # Check they are children of the planner
        planner_children = relationships.get("planner", {}).get("children", [])
        self.assertIn("observer-1", planner_children)
        self.assertIn("analyst-1", planner_children)
        self.assertIn("verifier-1", planner_children)
        
        # Check the parent relationships
        self.assertEqual(relationships.get("observer-1", {}).get("parent"), "planner")
        self.assertEqual(relationships.get("analyst-1", {}).get("parent"), "planner")
        self.assertEqual(relationships.get("verifier-1", {}).get("parent"), "planner")
    
    def test_path_result_recording(self):
        """Test recording results of path execution."""
        # Generate paths
        paths = self.planner.generate_solution_paths(self.test_bug)
        path_id = paths[0].path_id
        
        # Record success for a path
        success_metrics = {
            "execution_time": 10.5,
            "steps_completed": 5,
            "resources_used": {
                "memory": "120MB",
                "tokens": 2500
            }
        }
        self.planner.record_path_result(path_id, True, success_metrics)
        
        # Check metrics were updated
        self.assertEqual(self.planner.metrics["successful_fixes"], 1)
        self.assertEqual(self.planner.metrics["failed_fixes"], 0)
        
        # Check the path was updated
        path = self.planner.active_paths[path_id]
        self.assertTrue(path.metadata.get("executed", False))
        self.assertTrue(path.metadata.get("success", False))
        self.assertEqual(
            path.metadata.get("execution_metrics", {}).get("execution_time"),
            10.5
        )
        
        # Check the bug was updated
        bug = self.planner.active_bugs[self.test_bug.bug_id]
        self.assertIn("fixed", bug.tags)
        self.assertIn(f"fixed_by_path_{path_id}", bug.tags)
        
        # Record failure for another path
        if len(paths) > 1:
            path_id2 = paths[1].path_id
            failure_metrics = {
                "execution_time": 5.2,
                "error": "Verification failed",
                "failure_point": "verify_step"
            }
            self.planner.record_path_result(path_id2, False, failure_metrics)
            
            # Check metrics were updated
            self.assertEqual(self.planner.metrics["successful_fixes"], 1)
            self.assertEqual(self.planner.metrics["failed_fixes"], 1)
            
            # Check the path was updated
            path2 = self.planner.active_paths[path_id2]
            self.assertTrue(path2.metadata.get("executed", False))
            self.assertFalse(path2.metadata.get("success", True))
    
    def test_fallback_activation(self):
        """Test fallback strategy activation."""
        # Generate paths with high entropy to get fallbacks
        high_entropy_bug = BugState(
            bug_id="test-bug-fallback",
            title="Test Bug with Fallbacks",
            description="This is a complex test bug requiring fallbacks",
            severity="critical",
            status="new",
            metadata={"entropy": 0.9, "complexity": "very high"},
            tags={"test", "fallback"}
        )
        
        # Generate paths
        paths = self.planner.generate_solution_paths(high_entropy_bug)
        
        # Find a path with fallbacks
        path_with_fallbacks = None
        for path in paths:
            if path.fallbacks:
                path_with_fallbacks = path
                break
        
        # If we found a path with fallbacks, test activation
        if path_with_fallbacks:
            path_id = path_with_fallbacks.path_id
            original_fallback_count = len(path_with_fallbacks.fallbacks)
            
            # Activate fallback
            fallback = self.planner.activate_fallback(path_id)
            
            # Check fallback was returned
            self.assertIsNotNone(fallback)
            
            # Check fallback was removed from the path
            self.assertEqual(
                len(self.planner.active_paths[path_id].fallbacks),
                original_fallback_count - 1
            )
            
            # Check metrics were updated
            self.assertEqual(self.planner.metrics["fallbacks_used"], 1)
    
    def test_family_tree_relationships(self):
        """Test retrieving family relationships."""
        # Register agents in a specific hierarchy
        self.planner.register_agent("observer-2", "observer")
        self.planner.register_agent("analyst-2", "analyst")
        self.planner.register_agent("verifier-2", "verifier")
        
        # Get relationships for an agent
        observer_rels = self.planner.get_family_relationships("observer-2")
        
        # Check the relationships
        self.assertEqual(observer_rels["agent_id"], "observer-2")
        self.assertEqual(observer_rels["parent"], "planner")
        self.assertEqual(observer_rels["children"], [])
        self.assertEqual(observer_rels["ancestors"], ["planner"])
        
        # Get the full tree
        full_tree = self.planner.get_family_relationships()
        
        # Check it has the expected structure
        self.assertIn("relationships", full_tree)
        relationships = full_tree["relationships"]
        
        self.assertIn("planner", relationships)
        self.assertIn("observer-2", relationships)
        self.assertIn("analyst-2", relationships)
        self.assertIn("verifier-2", relationships)
    
    def test_metrics_reporting(self):
        """Test metrics reporting."""
        # Generate some activity
        self.planner.generate_solution_paths(self.test_bug)
        self.planner.register_agent("observer-3", "observer")
        
        # Get metrics
        metrics = self.planner.get_metrics()
        
        # Check metrics exist
        self.assertIn("paths_generated", metrics)
        self.assertIn("fallbacks_used", metrics)
        self.assertIn("successful_fixes", metrics)
        self.assertIn("failed_fixes", metrics)
        self.assertIn("avg_path_length", metrics)
        self.assertIn("active_bugs", metrics)
        self.assertIn("active_paths", metrics)
        self.assertIn("family_tree_size", metrics)
        self.assertIn("enabled", metrics)
        
        # Check values are sensible
        self.assertEqual(metrics["paths_generated"], 3)
        self.assertEqual(metrics["active_bugs"], 1)
        self.assertEqual(metrics["active_paths"], 3)
        self.assertGreater(metrics["family_tree_size"], 1)
        self.assertTrue(metrics["enabled"])


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_test_result(name, success):
    """Print test result."""
    result = "✅ PASSED" if success else "❌ FAILED"
    print(f"{result} - {name}")


def manual_test():
    """Run manual tests to show functionality."""
    print_header("PLANNER AGENT MANUAL TEST")
    
    # Create temp dir
    temp_dir = Path(tempfile.mkdtemp())
    family_tree_path = temp_dir / "family_tree.json"
    
    try:
        # Create a test config
        config = {
            "planner": {
                "enabled": True,
                "family-tree-path": str(family_tree_path),
                "solutions-per-bug": 2,
                "max-path-depth": 3,
                "fallback-threshold": 0.3
            }
        }
        
        # Create memory with test path
        memory = AgentMemory(
            mem_path=temp_dir / "memory.json",
            kv_path=temp_dir / "kv_store.json",
            compressed_path=temp_dir / "compressed_store.json",
            family_tree_path=family_tree_path
        )
        
        # Create the planner agent
        planner = PlannerAgent(config, memory)
        
        # Create a test bug
        bug = BugState(
            bug_id="manual-test-bug",
            title="Manual Test Bug",
            description="This is a manual test bug for the planner agent demonstration",
            severity="high",
            status="new",
            metadata={"entropy": 0.7, "complexity": "high"},
            tags={"manual", "demo"}
        )
        
        # Test 1: Generate solution paths
        paths = planner.generate_solution_paths(bug)
        print_test_result(
            "Generated solution paths",
            len(paths) == 2 and all(isinstance(p, PlannerPath) for p in paths)
        )
        
        # Print path details
        print("\nGenerated Paths:")
        for i, path in enumerate(paths):
            print(f"\nPath {i+1}:")
            print(f"  ID: {path.path_id}")
            print(f"  Priority: {path.metadata.get('priority', 'unknown')}")
            print(f"  Actions: {len(path.actions)}")
            print(f"  Fallbacks: {len(path.fallbacks)}")
            
            print("\n  Actions Details:")
            for j, action in enumerate(path.actions):
                print(f"    {j+1}. {action.get('type')} - {action.get('description')}")
        
        # Test 2: Register agents
        observer_id = "observer-demo"
        analyst_id = "analyst-demo"
        verifier_id = "verifier-demo"
        
        observer_reg = planner.register_agent(observer_id, "observer")
        analyst_reg = planner.register_agent(analyst_id, "analyst")
        verifier_reg = planner.register_agent(verifier_id, "verifier")
        
        print_test_result(
            "Registered agents in family tree",
            observer_reg and analyst_reg and verifier_reg
        )
        
        # Test 3: Get family tree
        tree = planner.get_family_relationships()
        
        print("\nFamily Tree Structure:")
        for agent_id, data in tree.get("relationships", {}).items():
            children = data.get("children", [])
            parent = data.get("parent", "none")
            agent_type = data.get("metadata", {}).get("type", "unknown")
            print(f"  Agent: {agent_id} (Type: {agent_type})")
            print(f"    Parent: {parent}")
            print(f"    Children: {', '.join(children) if children else 'none'}")
        
        # Test 4: Select best path
        best_path = planner.select_best_path(bug.bug_id)
        print_test_result(
            "Selected best path",
            best_path is not None and isinstance(best_path, PlannerPath)
        )
        
        if best_path:
            print(f"\nBest Path: {best_path.path_id}")
            print(f"  Priority: {best_path.metadata.get('priority', 'unknown')}")
        
        # Test 5: Record path result
        if paths:
            path_id = paths[0].path_id
            planner.record_path_result(
                path_id,
                True,
                {"execution_time": 8.2, "steps_completed": 3}
            )
            print_test_result(
                "Recorded successful path result",
                planner.metrics["successful_fixes"] == 1
            )
        
        # Test 6: Get metrics
        metrics = planner.get_metrics()
        print("\nPlanner Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print_test_result(
            "Retrieved metrics",
            isinstance(metrics, dict) and len(metrics) > 5
        )
        
    finally:
        # Clean up
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up: {e}")
    
    print_header("MANUAL TEST COMPLETE")


if __name__ == "__main__":
    # Run either unit tests or manual demonstration
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        manual_test()
    else:
        unittest.main()
