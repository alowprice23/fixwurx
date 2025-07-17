#!/usr/bin/env python3
"""
Test Command Executor

This script tests the Secure Command Execution Environment (SCEE) implementation,
including the permission system, credential manager, and blocker detection.
"""

import os
import sys
import json
import logging
import unittest
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_command_executor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TestCommandExecutor")

# Import the components to test
from command_executor import get_instance as get_command_executor
from permission_system import get_instance as get_permission_system
from credential_manager import get_instance as get_credential_manager
from blocker_detection import get_instance as get_blocker_detection


class MockRegistry:
    """
    Mock component registry for testing.
    """
    
    def __init__(self):
        """
        Initialize the mock registry.
        """
        self.components = {}
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
        logger.info(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Any:
        """
        Get a component.
        
        Args:
            name: Component name
            
        Returns:
            Component instance, or None if not found
        """
        return self.components.get(name)


class TestCommandExecutor(unittest.TestCase):
    """
    Test the Secure Command Execution Environment (SCEE) implementation.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """
        logger.info("Setting up test environment")
        
        # Create a mock registry
        cls.registry = MockRegistry()
        
        # Initialize the components
        cls.permission_system = get_permission_system(cls.registry)
        cls.credential_manager = get_credential_manager(cls.registry)
        cls.blocker_detection = get_blocker_detection(cls.registry)
        cls.command_executor = get_command_executor(cls.registry)
        
        # Create security directories if they don't exist
        os.makedirs("security", exist_ok=True)
        os.makedirs("blockers", exist_ok=True)
        os.makedirs("blockers/solutions", exist_ok=True)
        
        # Initialize all components
        cls.permission_system.initialize()
        cls.credential_manager.initialize()
        cls.blocker_detection.initialize()
        cls.command_executor.initialize()
        
        # Set up a test credential
        cls.credential_manager.set_credential("test_api_key", "test_value_12345")
        
        logger.info("Test environment set up complete")
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up the test environment.
        """
        logger.info("Cleaning up test environment")
        
        # Shut down all components
        cls.command_executor.shutdown()
        cls.blocker_detection.shutdown()
        cls.credential_manager.shutdown()
        cls.permission_system.shutdown()
        
        logger.info("Test environment clean up complete")
    
    def test_execute_simple_command(self):
        """
        Test executing a simple command.
        """
        logger.info("Testing execute_simple_command")
        
        # Execute a simple echo command as an admin agent
        result = self.command_executor.execute("echo 'Hello, World!'", "meta_agent")
        
        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("exit_code"), 0)
        self.assertIn("Hello, World!", result.get("stdout", ""))
    
    def test_execute_read_only_command(self):
        """
        Test executing a read-only command.
        """
        logger.info("Testing execute_read_only_command")
        
        # Execute a read-only command as an auditor agent
        result = self.command_executor.execute("ls -la", "auditor_agent", read_only=True)
        
        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("exit_code"), 0)
    
    def test_execute_command_with_permissions(self):
        """
        Test executing a command with permission checks.
        """
        logger.info("Testing execute_command_with_permissions")
        
        # Execute a command as an agent with limited permissions
        result = self.command_executor.execute("echo 'Test permissions'", "planner_agent")
        
        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("exit_code"), 0)
        self.assertIn("Test permissions", result.get("stdout", ""))
        
        # Try to execute a command with insufficient permissions
        try:
            # Temporarily assign a readonly role to the planner_agent
            original_role = self.permission_system.agent_roles.get("planner_agent")
            self.permission_system.agent_roles["planner_agent"] = "readonly"
            
            # This should fail because readonly doesn't have permission to rm
            result = self.command_executor.execute("rm some_file.txt", "planner_agent")
            
            self.assertFalse(result.get("success", True))
            self.assertIn("Permission denied", result.get("error", ""))
        finally:
            # Restore the original role
            self.permission_system.agent_roles["planner_agent"] = original_role
    
    def test_execute_blacklisted_command(self):
        """
        Test executing a blacklisted command.
        """
        logger.info("Testing execute_blacklisted_command")
        
        # Try to execute a blacklisted command, even as an admin
        result = self.command_executor.execute("rm -rf /", "meta_agent")
        
        self.assertFalse(result.get("success", True))
        self.assertIn("blacklisted", result.get("error", ""))
    
    def test_execute_command_requiring_confirmation(self):
        """
        Test executing a command that requires confirmation.
        """
        logger.info("Testing execute_command_requiring_confirmation")
        
        # Try to execute a command that requires confirmation
        result = self.command_executor.execute("rm -r test_dir", "meta_agent")
        
        self.assertFalse(result.get("success", True))
        self.assertTrue(result.get("confirmation_required", False))
        
        # Now execute with confirmation
        if result.get("confirmation_required", False):
            confirm_result = self.command_executor.execute_with_confirmation(
                result.get("command"), "meta_agent", True)
            
            # This still might fail if the directory doesn't exist, but it should try to execute
            self.assertIsNotNone(confirm_result)
    
    def test_execute_command_with_credentials(self):
        """
        Test executing a command with credential placeholders.
        """
        logger.info("Testing execute_command_with_credentials")
        
        # Execute a command with a credential placeholder
        result = self.command_executor.execute(
            "echo 'API Key: $CREDENTIAL(test_api_key)'", "meta_agent")
        
        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("exit_code"), 0)
        self.assertIn("API Key: test_value_12345", result.get("stdout", ""))
    
    def test_execute_script(self):
        """
        Test executing a script.
        """
        logger.info("Testing execute_script")
        
        # Create a simple script
        script_content = """
        #!/bin/bash
        echo "This is a test script"
        echo "Args: $@"
        exit 0
        """
        
        # Execute the script content directly
        result = self.command_executor.execute(script_content, "meta_agent")
        
        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("exit_code"), 0)
        self.assertIn("This is a test script", result.get("stdout", ""))
    
    def test_blocker_detection(self):
        """
        Test the blocker detection system.
        """
        logger.info("Testing blocker_detection")
        
        # First create an invalid command that will fail
        failing_command = "nonexistent_command with arguments"
        
        # Execute the command repeatedly to trigger blocker detection
        for i in range(4):  # We need to exceed max_repeated_failures (default 3)
            result = self.command_executor.execute(failing_command, "planner_agent")
            
            self.assertFalse(result.get("success", True))
            
            # Sleep a bit to avoid rate limiting or other issues
            import time
            time.sleep(0.1)
        
        # Check if a blocker was detected
        blockers = self.blocker_detection.list_active_blockers()
        
        self.assertTrue(blockers.get("success", False))
        self.assertGreaterEqual(len(blockers.get("blockers", {})), 1)


if __name__ == "__main__":
    unittest.main()
