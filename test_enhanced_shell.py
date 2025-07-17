#!/usr/bin/env python3
"""
Test Enhanced Shell Features

This script tests the enhanced shell environment features:
- Pipeline support for command chaining
- Command output redirection
- Background task execution
"""

import os
import sys
import time
import unittest
import tempfile
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shell components
from shell_environment import ShellEnvironment, ComponentRegistry
from shell_environment_enhanced import CommandPipeline

class TestCommandPipeline(unittest.TestCase):
    """Test the command pipeline functionality."""

    def setUp(self):
        """Set up for tests."""
        self.registry = ComponentRegistry()
        
        # Register test commands
        self.registry.register_command_handler("echo", self._echo_command, "test")
        self.registry.register_command_handler("count", self._count_command, "test")
        self.registry.register_command_handler("uppercase", self._uppercase_command, "test")
        self.registry.register_command_handler("cat", self._cat_command, "test")
        
        # Create command pipeline
        self.pipeline = CommandPipeline(self.registry)
    
    def _echo_command(self, args: str) -> int:
        """Echo command for testing."""
        print(args)
        return 0
    
    def _count_command(self, args: str) -> int:
        """Count lines from input."""
        lines = 0
        for line in sys.stdin:
            lines += 1
        print(f"Lines: {lines}")
        return 0
    
    def _uppercase_command(self, args: str) -> int:
        """Convert input to uppercase."""
        for line in sys.stdin:
            print(line.upper(), end="")
        return 0
    
    def _cat_command(self, args: str) -> int:
        """Simulate the cat command for testing."""
        try:
            file_path = args.strip()
            with open(file_path, 'r') as f:
                for line in f:
                    print(line, end="")
            return 0
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    
    def test_parse_command_line(self):
        """Test parsing command lines."""
        # Test simple command
        parsed = self.pipeline.parse_command_line("echo hello")
        self.assertEqual(parsed["pipeline"], ["echo hello"])
        self.assertFalse(parsed["background"])
        self.assertIsNone(parsed["redirection"])
        
        # Test pipeline
        parsed = self.pipeline.parse_command_line("echo hello | uppercase")
        self.assertEqual(parsed["pipeline"], ["echo hello", "uppercase"])
        self.assertFalse(parsed["background"])
        self.assertIsNone(parsed["redirection"])
        
        # Test redirection
        parsed = self.pipeline.parse_command_line("echo hello > output.txt")
        self.assertEqual(parsed["pipeline"], ["echo hello"])
        self.assertFalse(parsed["background"])
        self.assertEqual(parsed["redirection"], "output.txt")
        self.assertFalse(parsed["append"])
        
        # Test append redirection
        parsed = self.pipeline.parse_command_line("echo hello >> output.txt")
        self.assertEqual(parsed["pipeline"], ["echo hello"])
        self.assertFalse(parsed["background"])
        self.assertEqual(parsed["redirection"], "output.txt")
        self.assertTrue(parsed["append"])
        
        # Test background execution
        parsed = self.pipeline.parse_command_line("echo hello &")
        self.assertEqual(parsed["pipeline"], ["echo hello"])
        self.assertTrue(parsed["background"])
        self.assertIsNone(parsed["redirection"])
        
        # Test combined features
        parsed = self.pipeline.parse_command_line("echo hello | uppercase > output.txt &")
        self.assertEqual(parsed["pipeline"], ["echo hello", "uppercase"])
        self.assertTrue(parsed["background"])
        self.assertEqual(parsed["redirection"], "output.txt")
        self.assertFalse(parsed["append"])
    
    def test_simple_command(self):
        """Test executing a simple command."""
        with StringIO() as buffer, redirect_stdout(buffer):
            exit_code, output = self.pipeline.execute_pipeline(
                self.pipeline.parse_command_line("echo hello world")
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(output.strip(), "hello world")
    
    def test_pipeline_command(self):
        """Test pipeline execution."""
        # Create test file with lines
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_file.write("line1\nline2\nline3\n")
            temp_path = temp_file.name
        
        try:
            # Use cat command to read file and count lines
            exit_code, output = self.pipeline.execute_pipeline(
                self.pipeline.parse_command_line(f"cat {temp_path} | count")
            )
            self.assertEqual(exit_code, 0)
            self.assertIn("Lines: 3", output)
        finally:
            os.unlink(temp_path)
    
    def test_redirection(self):
        """Test output redirection."""
        # Create temp output file
        output_path = tempfile.mktemp()
        
        try:
            # Redirect output to file
            exit_code, output = self.pipeline.execute_pipeline(
                self.pipeline.parse_command_line(f"echo hello > {output_path}")
            )
            self.assertEqual(exit_code, 0)
            
            # Check file contents
            with open(output_path, 'r') as f:
                content = f.read().strip()
                self.assertEqual(content, "hello")
            
            # Test append mode
            exit_code, output = self.pipeline.execute_pipeline(
                self.pipeline.parse_command_line(f"echo world >> {output_path}")
            )
            self.assertEqual(exit_code, 0)
            
            # Check file contents again
            with open(output_path, 'r') as f:
                content = f.read().strip()
                self.assertEqual(content, "hello\nworld")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestBackgroundExecution(unittest.TestCase):
    """Test background execution functionality."""

    def setUp(self):
        """Set up for tests."""
        self.registry = ComponentRegistry()
        
        # Register test commands
        self.registry.register_command_handler("sleep", self._sleep_command, "test")
        self.registry.register_command_handler("echo", self._echo_command, "test")
        
        # Create command pipeline
        self.pipeline = CommandPipeline(self.registry)
    
    def _sleep_command(self, args: str) -> int:
        """Sleep command for testing."""
        seconds = float(args.strip())
        time.sleep(seconds)
        print(f"Slept for {seconds} seconds")
        return 0
    
    def _echo_command(self, args: str) -> int:
        """Echo command for testing."""
        print(args)
        return 0
    
    def test_background_execution(self):
        """Test background task execution."""
        # Start a background task
        task_id = self.pipeline.execute_in_background(
            self.pipeline.parse_command_line("sleep 0.5")
        )
        
        self.assertGreater(task_id, 0)
        
        # Check task status
        task = self.pipeline.get_background_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task["status"], "running")
        
        # Wait for task to complete
        time.sleep(1)
        
        # Check task status again
        task = self.pipeline.get_background_task(task_id)
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["exit_code"], 0)
        self.assertIn("Slept for 0.5 seconds", task["output"])
    
    def test_list_background_tasks(self):
        """Test listing background tasks."""
        # Start multiple background tasks
        task1_id = self.pipeline.execute_in_background(
            self.pipeline.parse_command_line("sleep 0.2")
        )
        task2_id = self.pipeline.execute_in_background(
            self.pipeline.parse_command_line("echo test output")
        )
        
        # Get task list
        tasks = self.pipeline.get_background_tasks()
        task_ids = [t["id"] for t in tasks]
        
        self.assertIn(task1_id, task_ids)
        self.assertIn(task2_id, task_ids)
        
        # Wait for tasks to complete
        time.sleep(0.5)
        
        # Cleanup tasks
        removed = self.pipeline.cleanup_completed_tasks(max_age=0)
        self.assertEqual(removed, 2)
        
        # Check task list again
        tasks = self.pipeline.get_background_tasks()
        self.assertEqual(len(tasks), 0)


if __name__ == "__main__":
    unittest.main()
