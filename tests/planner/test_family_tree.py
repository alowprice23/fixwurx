"""
tests/planner/test_family_tree.py
─────────────────────────────────
Test suite for the agent family tree functionality.

Tests:
1. Family tree creation and persistence
2. Agent registration and relationship tracking
3. Tree traversal and querying
4. Integration with planner agent
"""

import os
import pytest
import unittest
import json
import tempfile
from unittest.mock import MagicMock, patch

# Import components to test
from data_structures import FamilyTree
from planner_agent import PlannerAgent
from agent_memory import AgentMemory


@pytest.mark.family_tree
class TestFamilyTreeCore(unittest.TestCase):
    """Test core family tree functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the family tree
        self.temp_dir = tempfile.TemporaryDirectory()
        self.family_tree_path = os.path.join(self.temp_dir.name, "family_tree.json")
        
        # Initialize family tree
        self.family_tree = FamilyTree()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test family tree initialization."""
        self.assertIsNotNone(self.family_tree)
        self.assertEqual(len(self.family_tree.agents), 0)
        self.assertEqual(len(self.family_tree.relationships), 0)
    
    def test_add_agent(self):
        """Test adding an agent to the family tree."""
        # Add an agent
        self.family_tree.add_agent("agent-1", "observer")
        
        # Check that the agent was added
        self.assertIn("agent-1", self.family_tree.agents)
        self.assertEqual(self.family_tree.agents["agent-1"]["type"], "observer")
        self.assertEqual(self.family_tree.agents["agent-1"]["status"], "active")
        
        # Add another agent
        self.family_tree.add_agent("agent-2", "analyst", parent_id="agent-1")
        
        # Check that the agent was added
        self.assertIn("agent-2", self.family_tree.agents)
        self.assertEqual(self.family_tree.agents["agent-2"]["type"], "analyst")
        self.assertEqual(self.family_tree.agents["agent-2"]["parent_id"], "agent-1")
    
    def test_add_relationship(self):
        """Test adding a relationship between agents."""
        # Add agents
        self.family_tree.add_agent("agent-1", "observer")
        self.family_tree.add_agent("agent-2", "analyst")
        
        # Add relationship
        self.family_tree.add_relationship("agent-1", "agent-2", "parent")
        
        # Check that the relationship was added
        self.assertIn(("agent-1", "agent-2"), self.family_tree.relationships)
        self.assertEqual(self.family_tree.relationships[("agent-1", "agent-2")], "parent")
    
    def test_get_children(self):
        """Test getting children of an agent."""
        # Add agents with parent-child relationships
        self.family_tree.add_agent("parent", "planner")
        self.family_tree.add_agent("child1", "observer", parent_id="parent")
        self.family_tree.add_agent("child2", "analyst", parent_id="parent")
        self.family_tree.add_agent("grandchild", "verifier", parent_id="child1")
        
        # Add relationships
        self.family_tree.add_relationship("parent", "child1", "parent")
        self.family_tree.add_relationship("parent", "child2", "parent")
        self.family_tree.add_relationship("child1", "grandchild", "parent")
        
        # Get children of parent
        children = self.family_tree.get_children("parent")
        
        # Check children
        self.assertEqual(len(children), 2)
        self.assertIn("child1", children)
        self.assertIn("child2", children)
        
        # Get children of child1
        children = self.family_tree.get_children("child1")
        
        # Check children
        self.assertEqual(len(children), 1)
        self.assertIn("grandchild", children)
    
    def test_get_parent(self):
        """Test getting the parent of an agent."""
        # Add agents with parent-child relationships
        self.family_tree.add_agent("parent", "planner")
        self.family_tree.add_agent("child", "observer", parent_id="parent")
        
        # Add relationship
        self.family_tree.add_relationship("parent", "child", "parent")
        
        # Get parent of child
        parent = self.family_tree.get_parent("child")
        
        # Check parent
        self.assertEqual(parent, "parent")
        
        # Get parent of parent (should be None)
        parent = self.family_tree.get_parent("parent")
        self.assertIsNone(parent)
    
    def test_save_and_load(self):
        """Test saving and loading the family tree."""
        # Add some agents and relationships
        self.family_tree.add_agent("agent-1", "planner")
        self.family_tree.add_agent("agent-2", "observer", parent_id="agent-1")
        self.family_tree.add_agent("agent-3", "analyst", parent_id="agent-1")
        
        self.family_tree.add_relationship("agent-1", "agent-2", "parent")
        self.family_tree.add_relationship("agent-1", "agent-3", "parent")
        
        # Save the family tree
        self.family_tree.save_to_file(self.family_tree_path)
        
        # Create a new family tree and load it
        new_tree = FamilyTree()
        new_tree.load_from_file(self.family_tree_path)
        
        # Check that the trees match
        self.assertEqual(len(new_tree.agents), len(self.family_tree.agents))
        self.assertEqual(len(new_tree.relationships), len(self.family_tree.relationships))
        
        for agent_id, agent_data in self.family_tree.agents.items():
            self.assertIn(agent_id, new_tree.agents)
            self.assertEqual(new_tree.agents[agent_id]["type"], agent_data["type"])
            
        for relation, relation_type in self.family_tree.relationships.items():
            self.assertIn(relation, new_tree.relationships)
            self.assertEqual(new_tree.relationships[relation], relation_type)


@pytest.mark.family_tree
class TestFamilyTreePlanner(unittest.TestCase):
    """Test family tree integration with planner agent."""
    
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
        
        # Initialize planner agent
        self.planner = PlannerAgent(self.config, self.agent_memory)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_register_agent(self):
        """Test registering an agent with the planner."""
        # Register an agent
        success = self.planner.register_agent("observer-1", "observer")
        
        # Check that the agent was registered
        self.assertTrue(success)
        
        # Check that the agent is in the family tree
        family_tree = self.planner.get_family_tree()
        self.assertIn("observer-1", family_tree.agents)
        self.assertEqual(family_tree.agents["observer-1"]["type"], "observer")
    
    def test_agent_hierarchy(self):
        """Test creating an agent hierarchy."""
        # Register a root planner agent
        self.planner.register_agent("planner-1", "planner")
        
        # Register child agents
        self.planner.register_agent("observer-1", "observer", parent_id="planner-1")
        self.planner.register_agent("analyst-1", "analyst", parent_id="planner-1")
        
        # Register a grandchild
        self.planner.register_agent("verifier-1", "verifier", parent_id="analyst-1")
        
        # Get the family tree
        family_tree = self.planner.get_family_tree()
        
        # Check the hierarchy
        self.assertEqual(family_tree.get_parent("observer-1"), "planner-1")
        self.assertEqual(family_tree.get_parent("analyst-1"), "planner-1")
        self.assertEqual(family_tree.get_parent("verifier-1"), "analyst-1")
        
        # Check the children
        children = family_tree.get_children("planner-1")
        self.assertEqual(len(children), 2)
        self.assertIn("observer-1", children)
        self.assertIn("analyst-1", children)
        
        children = family_tree.get_children("analyst-1")
        self.assertEqual(len(children), 1)
        self.assertIn("verifier-1", children)
    
    def test_get_family_relationships(self):
        """Test getting family relationships from the planner."""
        # Register agents in a hierarchy
        self.planner.register_agent("planner-1", "planner")
        self.planner.register_agent("observer-1", "observer", parent_id="planner-1")
        self.planner.register_agent("analyst-1", "analyst", parent_id="planner-1")
        
        # Get the family relationships
        relationships = self.planner.get_family_relationships()
        
        # Check the relationships
        self.assertIn("agents", relationships)
        self.assertEqual(len(relationships["agents"]), 3)
        
        # Check that each agent has the expected attributes
        for agent in relationships["agents"]:
            self.assertIn("id", agent)
            self.assertIn("type", agent)
            
            if agent["id"] == "planner-1":
                self.assertEqual(agent["type"], "planner")
                self.assertIsNone(agent.get("parent_id"))
            else:
                self.assertEqual(agent.get("parent_id"), "planner-1")


if __name__ == "__main__":
    unittest.main()
