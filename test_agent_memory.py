#!/usr/bin/env python3
"""
test_agent_memory.py
────────────────────
Test suite for the AgentMemory module.

This script verifies:
1. Basic operations (add, query, store, retrieve)
2. Compression functionality
3. Family tree storage and retrieval
4. Solution path versioning
5. Cross-session learning storage
6. Performance metrics
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import unittest

from agent_memory import AgentMemory
from compress import Compressor


class TestAgentMemory(unittest.TestCase):
    """Test cases for the AgentMemory class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Setup paths for test files
        self.mem_path = self.test_dir / "memory.json"
        self.kv_path = self.test_dir / "kv_store.json"
        self.compressed_path = self.test_dir / "compressed_store.json"
        self.family_tree_path = self.test_dir / "family_tree.json"
        
        # Create memory instance
        self.memory = AgentMemory(
            mem_path=self.mem_path,
            kv_path=self.kv_path,
            compressed_path=self.compressed_path,
            family_tree_path=self.family_tree_path
        )
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            shutil.rmtree(self.test_dir)
        except:
            print(f"Warning: Failed to remove test directory {self.test_dir}")
    
    def test_bug_storage(self):
        """Test bug storage and retrieval."""
        # Add a test bug
        self.memory.add_entry(
            "BUG-123",
            "Fix division by zero in calculation module",
            "Add null check before division operation",
            {"severity": "high", "component": "calculator"}
        )
        
        # Query similar bugs
        similar = self.memory.query_similar("division zero calculation")
        
        # Verify the bug is found
        self.assertTrue(len(similar) > 0)
        self.assertEqual(similar[0][0], "BUG-123")
        
        # Verify similarity score is reasonable
        self.assertGreater(similar[0][1], 0.1)
        
        # Verify persistence
        self.assertTrue(self.mem_path.exists())
        
        # Create a new memory instance with the same path
        new_memory = AgentMemory(
            mem_path=self.mem_path,
            kv_path=self.kv_path,
            compressed_path=self.compressed_path,
            family_tree_path=self.family_tree_path
        )
        
        # Verify bug is loaded
        similar = new_memory.query_similar("division zero calculation")
        self.assertTrue(len(similar) > 0)
        self.assertEqual(similar[0][0], "BUG-123")
    
    def test_key_value_storage(self):
        """Test key-value storage functionality."""
        # Store some values
        self.memory.store("test:key1", "value1")
        self.memory.store("test:key2", {"nested": "value2"})
        self.memory.store("other:key", [1, 2, 3])
        
        # Retrieve values
        self.assertEqual(self.memory.retrieve("test:key1"), "value1")
        self.assertEqual(self.memory.retrieve("test:key2")["nested"], "value2")
        self.assertEqual(self.memory.retrieve("other:key"), [1, 2, 3])
        
        # Verify default value
        self.assertEqual(self.memory.retrieve("nonexistent", "default"), "default")
        
        # List keys with prefix
        test_keys = self.memory.list_keys("test:")
        self.assertEqual(len(test_keys), 2)
        self.assertIn("test:key1", test_keys)
        self.assertIn("test:key2", test_keys)
        
        # Verify persistence
        self.assertTrue(self.kv_path.exists())
        
        # Create a new memory instance with the same path
        new_memory = AgentMemory(
            mem_path=self.mem_path,
            kv_path=self.kv_path,
            compressed_path=self.compressed_path,
            family_tree_path=self.family_tree_path
        )
        
        # Verify data is loaded
        self.assertEqual(new_memory.retrieve("test:key1"), "value1")
    
    def test_compression(self):
        """Test compression functionality."""
        # Create a large text
        large_text = "This is a sample of repeated text for compression testing. " * 100
        
        # Store with compression
        stats = self.memory.store_compressed("test:large", large_text)
        
        # Verify compression stats
        self.assertIn("original_size", stats)
        self.assertIn("compressed_size", stats)
        self.assertIn("compression_ratio", stats)
        
        # Retrieve compressed data
        retrieved = self.memory.retrieve_compressed("test:large")
        self.assertEqual(retrieved, large_text)
        
        # Get overall compression stats
        overall_stats = self.memory.get_compression_stats()
        self.assertEqual(overall_stats["count"], 1)
        
        # Verify persistence
        self.assertTrue(self.compressed_path.exists())
        
        # Create a new memory instance with the same path
        new_memory = AgentMemory(
            mem_path=self.mem_path,
            kv_path=self.kv_path,
            compressed_path=self.compressed_path,
            family_tree_path=self.family_tree_path
        )
        
        # Verify data is loaded
        retrieved = new_memory.retrieve_compressed("test:large")
        self.assertEqual(retrieved, large_text)
    
    def test_family_tree(self):
        """Test family tree storage and retrieval."""
        # Create a sample family tree
        family_tree = {
            "root": "planner",
            "children": {
                "observer": {
                    "role": "analysis",
                    "children": {}
                },
                "analyst": {
                    "role": "solution",
                    "children": {}
                },
                "verifier": {
                    "role": "validation",
                    "children": {}
                }
            },
            "metadata": {
                "created_at": time.time(),
                "version": "1.0.0"
            }
        }
        
        # Store the family tree
        self.memory.store_family_tree(family_tree)
        
        # Verify it's stored correctly
        retrieved = self.memory.get_family_tree()
        self.assertEqual(retrieved["root"], "planner")
        self.assertEqual(len(retrieved["children"]), 3)
        
        # Verify file is created
        self.assertTrue(self.family_tree_path.exists())
        
        # Load the file directly to check format
        with self.family_tree_path.open("r") as f:
            tree_json = json.load(f)
        
        self.assertEqual(tree_json["root"], "planner")
    
    def test_solution_path_versioning(self):
        """Test solution path versioning functionality."""
        # First, create a solution path with minimal content
        # to test the basic version control functionality
        path_id = "test-path-1"
        
        # Create initial version with bare minimum content
        initial_path = {
            "bug_id": "BUG-42",
            "priority": 0.8
        }
        
        # Store initial version
        result = self.memory.store_solution_path(
            path_id, 
            initial_path, 
            {"creator": "planner", "description": "Initial path"}
        )
        
        # Verify storage result
        self.assertEqual(result["path_id"], path_id)
        self.assertEqual(result["revision_number"], 0)
        
        # Create an updated version with a different priority
        updated_path = {
            "bug_id": "BUG-42",
            "priority": 0.9
        }
        
        # Store updated version
        result = self.memory.store_solution_path(
            path_id, 
            updated_path, 
            {"creator": "planner", "description": "Updated priority"}
        )
        
        # Verify it's a new revision
        self.assertEqual(result["revision_number"], 1)
        
        # Get current revision (should be the latest)
        current = self.memory.get_solution_path(path_id)
        self.assertEqual(current["priority"], 0.9)
        
        # Get specific revision
        first_revision = self.memory.get_solution_path(path_id, 0)
        self.assertEqual(first_revision["priority"], 0.8)
        
        # Get revision history
        history = self.memory.get_revision_history(path_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["description"], "Initial path")
        self.assertEqual(history[1]["description"], "Updated priority")
        
        # Rollback to first revision
        rollback = self.memory.rollback_solution_path(path_id, 0)
        self.assertEqual(rollback["priority"], 0.8)
        
        # Verify current is now the first revision
        current = self.memory.get_solution_path(path_id)
        self.assertEqual(current["priority"], 0.8)
        
        # List all solution paths
        paths = self.memory.list_solution_paths()
        self.assertIn(path_id, paths)
    
    def test_learning_storage(self):
        """Test cross-session learning storage."""
        # Store learning data for a model
        model_id = "gpt-4"
        learning_data = {
            "successful_patterns": [
                {"pattern": "check null before division", "success_rate": 0.95},
                {"pattern": "validate input parameters", "success_rate": 0.87}
            ],
            "failure_patterns": [
                {"pattern": "ignore edge cases", "failure_rate": 0.75}
            ],
            "metadata": {
                "training_iterations": 100,
                "last_updated": time.time()
            }
        }
        
        self.memory.store_learning_data(model_id, learning_data)
        
        # Retrieve learning data
        retrieved = self.memory.get_learning_data(model_id)
        self.assertEqual(len(retrieved["successful_patterns"]), 2)
        self.assertEqual(len(retrieved["failure_patterns"]), 1)
        
        # Store learning data for another model
        self.memory.store_learning_data("claude-3", {"test": "data"})
        
        # List learning models
        models = self.memory.list_learning_models()
        self.assertIn("gpt-4", models)
        self.assertIn("claude-3", models)
    
    def test_persistence_and_stats(self):
        """Test persistence and statistics."""
        # Add some data
        self.memory.add_entry("BUG-1", "Test bug", "Test patch")
        self.memory.store("key1", "value1")
        self.memory.store_compressed("compressed1", "value" * 100)
        
        # Get statistics
        stats = self.memory.stats()
        
        # Verify stats
        self.assertEqual(stats["bugs"]["count"], 1)
        self.assertEqual(stats["kv_store"]["count"], 1)
        self.assertEqual(stats["compressed_store"]["count"], 1)
        
        # Verify string representation
        repr_str = repr(self.memory)
        self.assertIn("bugs=1", repr_str)
        self.assertIn("kv_entries=1", repr_str)
        self.assertIn("compressed_entries=1", repr_str)
        
        # Save all data
        self.memory.save_all()
        
        # Verify all files exist
        self.assertTrue(self.mem_path.exists())
        self.assertTrue(self.kv_path.exists())
        self.assertTrue(self.compressed_path.exists())

        # Create a new memory instance with the same path
        new_memory = AgentMemory(
            mem_path=self.mem_path,
            kv_path=self.kv_path,
            compressed_path=self.compressed_path,
            family_tree_path=self.family_tree_path
        )
        
        # Verify all data is loaded
        self.assertEqual(len(new_memory), 1)
        self.assertEqual(new_memory.retrieve("key1"), "value1")
        self.assertIsNotNone(new_memory.retrieve_compressed("compressed1"))


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def main():
    """Run the test suite."""
    print_header("AGENT MEMORY TEST SUITE")
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
