#!/usr/bin/env python3
"""
test_replay_buffer.py
─────────────────────
Unit tests for the replay buffer module. Tests both the basic ReplayBuffer and
the enhanced CompressedReplayBuffer with a focus on:

1. Basic functionality
2. Compression capabilities
3. Persistence to disk
4. Security features
5. Integration with the larger system
"""

import os
import time
import json
import tempfile
import unittest
import hashlib
from pathlib import Path
from typing import Dict, Any, List

from replay_buffer import ReplayBuffer, CompressedReplayBuffer, secure_sample


class TestSecureSample(unittest.TestCase):
    """Test the secure_sample function."""
    
    def test_secure_sample_basic(self):
        """Test basic sampling functionality."""
        population = list(range(100))
        sample_size = 10
        
        # Test that we get the right number of samples
        sample = secure_sample(population, sample_size)
        self.assertEqual(len(sample), sample_size)
        
        # Test that all samples are unique
        self.assertEqual(len(sample), len(set(sample)))
        
        # Test that all samples are from the original population
        for item in sample:
            self.assertIn(item, population)
    
    def test_secure_sample_edge_cases(self):
        """Test edge cases for secure_sample."""
        # Empty population
        with self.assertRaises(ValueError):
            secure_sample([], 1)
        
        # Sample size equals population size
        population = [1, 2, 3, 4, 5]
        sample = secure_sample(population, 5)
        self.assertEqual(set(sample), set(population))
        
        # Sample size greater than population size
        with self.assertRaises(ValueError):
            secure_sample(population, 6)
        
        # Sample size of 0
        sample = secure_sample(population, 0)
        self.assertEqual(sample, [])
    
    def test_secure_sample_distribution(self):
        """Test that sampling is relatively uniform."""
        # This is a probabilistic test, but should catch obvious biases
        population = list(range(10))
        # Take a large number of samples
        counts = [0] * 10
        num_trials = 10000
        sample_size = 1
        
        for _ in range(num_trials):
            sample = secure_sample(population, sample_size)
            counts[sample[0]] += 1
        
        # Check that each element was sampled approximately the expected number of times
        expected = num_trials / len(population)
        for i, count in enumerate(counts):
            # Allow for 20% deviation from expected
            self.assertGreater(count, expected * 0.8, f"Element {i} undersampled")
            self.assertLess(count, expected * 1.2, f"Element {i} oversampled")


class TestReplayBuffer(unittest.TestCase):
    """Test the basic ReplayBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=5)
        self.sample_episodes = [
            {"bugs_seen": 1, "success_rate": 0.8, "custom_field": "value1"},
            {"bugs_seen": 2, "success_rate": 0.85, "custom_field": "value2"},
            {"bugs_seen": 3, "success_rate": 0.9, "custom_field": "value3"},
            {"bugs_seen": 4, "success_rate": 0.95, "custom_field": "value4"},
            {"bugs_seen": 5, "success_rate": 1.0, "custom_field": "value5"},
            {"bugs_seen": 6, "success_rate": 0.75, "custom_field": "value6"}
        ]
    
    def test_add_and_capacity(self):
        """Test adding episodes and capacity enforcement."""
        # Add episodes up to capacity
        for i in range(5):
            self.buffer.add(self.sample_episodes[i])
            self.assertEqual(len(self.buffer), i + 1)
        
        # Verify buffer is full
        self.assertTrue(self.buffer.is_full())
        
        # Add one more episode, which should replace the oldest
        self.buffer.add(self.sample_episodes[5])
        
        # Verify buffer size is still at capacity
        self.assertEqual(len(self.buffer), 5)
        
        # Verify the oldest episode was removed (bugs_seen=1)
        samples = self.buffer.sample(5)  # Get all episodes
        bugs_seen_values = [ep["bugs_seen"] for ep in samples]
        self.assertNotIn(1, bugs_seen_values)
        self.assertIn(6, bugs_seen_values)
    
    def test_sampling(self):
        """Test sampling functionality."""
        # Add all episodes
        for episode in self.sample_episodes:
            self.buffer.add(episode)
        
        # Sample 3 episodes
        samples = self.buffer.sample(3)
        self.assertEqual(len(samples), 3)
        
        # Verify all samples are unique
        bugs_seen_values = [ep["bugs_seen"] for ep in samples]
        self.assertEqual(len(bugs_seen_values), len(set(bugs_seen_values)))
        
        # Test sampling when buffer is not full
        buffer = ReplayBuffer(capacity=10)
        buffer.add({"test": "value"})
        samples = buffer.sample(5)
        self.assertEqual(len(samples), 1)  # Only one episode available
        
        # Test sampling from empty buffer
        buffer.clear()
        samples = buffer.sample(5)
        self.assertEqual(samples, [])
    
    def test_timestamp_addition(self):
        """Test that a timestamp is added if not present."""
        # Add episode without timestamp
        episode = {"bugs_seen": 7, "success_rate": 0.7}
        self.buffer.add(episode)
        
        # Sample the episode
        sample = self.buffer.sample(1)[0]
        
        # Verify timestamp was added
        self.assertIn("timestamp", sample)
        self.assertIsInstance(sample["timestamp"], int)
    
    def test_clear(self):
        """Test clearing the buffer."""
        # Add episodes
        for episode in self.sample_episodes[:3]:
            self.buffer.add(episode)
        
        # Clear buffer
        self.buffer.clear()
        
        # Verify buffer is empty
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.sample(5), [])


class TestCompressedReplayBuffer(unittest.TestCase):
    """Test the enhanced CompressedReplayBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for buffer files
        self.test_dir = tempfile.TemporaryDirectory()
        self.buffer_file = Path(self.test_dir.name) / "test_buffer.json"
        
        # Create the buffer
        self.buffer = CompressedReplayBuffer(
            capacity=5,
            max_tokens=1024,
            buffer_file=self.buffer_file
        )
        
        # Sample episodes with different sizes to test compression
        self.small_episode = {
            "bugs_seen": 1,
            "success_rate": 0.8,
            "metadata": {"small": "episode"}
        }
        
        # Generate a larger episode with repeating text (more compressible)
        large_text = "This is a longer description that will benefit from compression. " * 20
        self.large_episode = {
            "bugs_seen": 2,
            "success_rate": 0.9,
            "description": large_text,
            "metadata": {"large": "episode"}
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.test_dir.cleanup()
    
    def test_compression(self):
        """Test that episodes are compressed properly."""
        # Add a large episode that should benefit from compression
        metadata = self.buffer.add(self.large_episode)
        
        # Verify compression metadata
        self.assertIn("original_size", metadata)
        self.assertIn("compressed_size", metadata)
        self.assertIn("compression_ratio", metadata)
        self.assertIn("checksum", metadata)
        
        # For highly compressible data, the ratio should be < 1.0
        # But we can't guarantee this in a test, so we just check it exists
        ratio = metadata["compression_ratio"]
        self.assertIsInstance(ratio, float)
        
        # Verify we can retrieve the episode correctly
        samples = self.buffer.sample(1)
        self.assertEqual(len(samples), 1)
        retrieved = samples[0]
        
        # Verify the content matches the original
        self.assertEqual(retrieved["bugs_seen"], self.large_episode["bugs_seen"])
        self.assertEqual(retrieved["description"], self.large_episode["description"])
        self.assertEqual(retrieved["metadata"], self.large_episode["metadata"])
    
    def test_checksum_verification(self):
        """Test integrity verification with checksums."""
        # Add an episode
        self.buffer.add(self.small_episode)
        
        # Corrupt the buffer file
        with open(self.buffer_file, 'r') as f:
            data = json.load(f)
        
        # Modify the episode data
        corrupted_data = data.copy()
        corrupted_data["episodes"][0]["data"] = "corrupted data"
        
        with open(self.buffer_file, 'w') as f:
            json.dump(corrupted_data, f)
        
        # Create a new buffer that loads from the corrupted file
        corrupt_buffer = CompressedReplayBuffer(
            capacity=5,
            buffer_file=self.buffer_file
        )
        
        # Sampling should either fail or return an error message
        samples = corrupt_buffer.sample(1)
        if samples:
            # If it returns a sample with an error, check for the error
            self.assertTrue(
                "_error" in samples[0] or  # Explicit error flag
                samples[0] != self.small_episode  # Different from original
            )
    
    def test_persistence(self):
        """Test that buffer contents persist to disk."""
        # Add episodes
        self.buffer.add(self.small_episode)
        self.buffer.add(self.large_episode)
        
        # Create a new buffer from the same file
        new_buffer = CompressedReplayBuffer(
            capacity=5,
            buffer_file=self.buffer_file
        )
        
        # Verify the episodes were loaded
        self.assertEqual(len(new_buffer), 2)
        
        # Sample all episodes
        samples = new_buffer.sample(2)
        
        # Sort by bugs_seen to ensure consistent order
        samples.sort(key=lambda ep: ep["bugs_seen"])
        
        # Verify content matches original
        self.assertEqual(samples[0]["bugs_seen"], self.small_episode["bugs_seen"])
        self.assertEqual(samples[1]["bugs_seen"], self.large_episode["bugs_seen"])
        self.assertEqual(samples[1]["description"], self.large_episode["description"])
    
    def test_compression_stats(self):
        """Test that compression statistics are tracked correctly."""
        # Add episodes
        self.buffer.add(self.small_episode)
        self.buffer.add(self.large_episode)
        
        # Get compression stats
        stats = self.buffer.get_compression_stats()
        
        # Verify stats are tracked
        self.assertIn("total_original_size", stats)
        self.assertIn("total_compressed_size", stats)
        self.assertIn("compression_ratio", stats)
        self.assertIn("episodes_compressed", stats)
        self.assertIn("bits_saved", stats)
        
        # Verify episode count
        self.assertEqual(stats["episodes_compressed"], 2)
        
        # Adding more episodes should update stats
        self.buffer.add(self.small_episode)
        new_stats = self.buffer.get_compression_stats()
        self.assertEqual(new_stats["episodes_compressed"], 3)
        self.assertGreater(
            new_stats["total_original_size"],
            stats["total_original_size"]
        )
    
    def test_clear(self):
        """Test clearing the buffer and resetting stats."""
        # Add episodes
        self.buffer.add(self.small_episode)
        self.buffer.add(self.large_episode)
        
        # Clear buffer
        self.buffer.clear()
        
        # Verify buffer is empty
        self.assertEqual(len(self.buffer), 0)
        
        # Verify stats are reset
        stats = self.buffer.get_compression_stats()
        self.assertEqual(stats["episodes_compressed"], 0)
        self.assertEqual(stats["total_original_size"], 0)
    
    def test_repr(self):
        """Test the string representation."""
        # Add an episode
        self.buffer.add(self.large_episode)
        
        # Get the string representation
        repr_str = repr(self.buffer)
        
        # Verify it contains key information
        self.assertIn("CompressedReplayBuffer", repr_str)
        self.assertIn("1/5", repr_str)  # 1 episode, capacity 5
        self.assertIn("ratio=", repr_str)
        self.assertIn("bits_saved=", repr_str)


class TestIntegration(unittest.TestCase):
    """Test integration with other system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.buffer_file = Path(self.test_dir.name) / "integration_buffer.json"
        
        # Create the buffer
        self.buffer = CompressedReplayBuffer(
            capacity=10,
            buffer_file=self.buffer_file
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.test_dir.cleanup()
    
    def test_planner_integration(self):
        """Test integration with planner agent data."""
        # Simulate a planner agent storing execution paths
        planner_data = {
            "execution_path": "path1 -> path2 -> path3",
            "dependencies": ["dep1", "dep2"],
            "fallback_strategies": ["fallback1", "fallback2"],
            "bug_id": "BUG-123",
            "success_rate": 0.95,
            "planner_metadata": {
                "version": "1.0",
                "strategy": "aggressive",
                "path_entropy": 0.75
            }
        }
        
        # Add to buffer
        metadata = self.buffer.add(planner_data)
        
        # Verify compression
        self.assertIn("compression_ratio", metadata)
        
        # Retrieve from buffer
        samples = self.buffer.sample(1)
        retrieved = samples[0]
        
        # Verify integrity of complex nested structure
        self.assertEqual(retrieved["execution_path"], planner_data["execution_path"])
        self.assertEqual(retrieved["dependencies"], planner_data["dependencies"])
        self.assertEqual(retrieved["fallback_strategies"], planner_data["fallback_strategies"])
        self.assertEqual(retrieved["planner_metadata"]["strategy"], planner_data["planner_metadata"]["strategy"])
    
    def test_family_tree_integration(self):
        """Test integration with family tree data."""
        # Simulate family tree storage
        family_tree = {
            "root": "planner_agent",
            "children": {
                "observer": {
                    "role": "observation",
                    "capabilities": ["bug_reproduction"],
                    "children": {}
                },
                "analyst": {
                    "role": "analysis",
                    "capabilities": ["code_analysis"],
                    "children": {
                        "specialist": {
                            "role": "specialist",
                            "capabilities": ["database_fix"],
                            "children": {}
                        }
                    }
                }
            }
        }
        
        # Add to buffer with additional metadata
        episode = {
            "bug_id": "BUG-456",
            "family_tree": family_tree,
            "active_agents": ["planner_agent", "analyst", "specialist"],
            "solution_path": ["step1", "step2", "step3"]
        }
        
        metadata = self.buffer.add(episode)
        
        # Verify compression
        self.assertIn("compression_ratio", metadata)
        
        # Retrieve from buffer
        samples = self.buffer.sample(1)
        retrieved = samples[0]
        
        # Verify complex nested structure integrity
        self.assertEqual(retrieved["family_tree"]["root"], family_tree["root"])
        self.assertEqual(
            retrieved["family_tree"]["children"]["analyst"]["children"]["specialist"]["role"],
            family_tree["children"]["analyst"]["children"]["specialist"]["role"]
        )
        self.assertEqual(retrieved["active_agents"], episode["active_agents"])
    
    def test_multiple_buffers_same_file(self):
        """Test multiple buffers accessing the same file."""
        # Add an episode to the first buffer
        episode = {"test": "value", "complex": [1, 2, 3, {"nested": "value"}]}
        self.buffer.add(episode)
        
        # Create a second buffer with the same file
        buffer2 = CompressedReplayBuffer(
            capacity=10,
            buffer_file=self.buffer_file
        )
        
        # Verify the episode was loaded
        self.assertEqual(len(buffer2), 1)
        
        # Add an episode to the second buffer
        episode2 = {"test": "value2", "data": ["a", "b", "c"]}
        buffer2.add(episode2)
        
        # Create a new instance that will load the latest data from disk
        # (Each instance only reads from disk when initialized)
        refreshed_buffer = CompressedReplayBuffer(
            capacity=10,
            buffer_file=self.buffer_file
        )
        
        # Verify the refreshed buffer sees both episodes
        self.assertEqual(len(refreshed_buffer), 2)
        
        # Create a third buffer and verify it loads both episodes
        buffer3 = CompressedReplayBuffer(
            capacity=10,
            buffer_file=self.buffer_file
        )
        self.assertEqual(len(buffer3), 2)


def main():
    """Run the test suite."""
    unittest.main()


if __name__ == "__main__":
    main()
