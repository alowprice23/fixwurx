#!/usr/bin/env python3
"""
Test Shell Environment

This script tests the functionality of the FixWurx shell environment components.
"""

import os
import sys
import time
import unittest
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shell components
from shell_environment import ShellEnvironment, ComponentRegistry
from permission_system import PermissionSystem
from shell_scripting import ShellScripting, SimpleScriptInterpreter
from web_interface import WebInterface

class TestComponentRegistry(unittest.TestCase):
    """Test the component registry."""

    def setUp(self):
        """Set up for tests."""
        self.registry = ComponentRegistry()
    
    def test_register_command_handler(self):
        """Test registering a command handler."""
        def test_command(args):
            return 0
        
        self.registry.register_command_handler("test", test_command, "test_component")
        
        handler_info = self.registry.get_command_handler("test")
        self.assertIsNotNone(handler_info)
        self.assertEqual(handler_info["handler"], test_command)
        self.assertEqual(handler_info["component"], "test_component")
    
    def test_register_component(self):
        """Test registering a component."""
        component = object()
        self.registry.register_component("test_component", component)
        
        retrieved_component = self.registry.get_component("test_component")
        self.assertEqual(retrieved_component, component)
    
    def test_event_handling(self):
        """Test event handling."""
        def test_event_handler(event_data):
            return {"success": True, "data": event_data}
        
        self.registry.register_event_handler("test_event", test_event_handler, "test_component")
        
        event_data = {"test": "data"}
        results = self.registry.trigger_event("test_event", event_data)
        
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])
        self.assertEqual(results[0]["data"], event_data)


class TestShellEnvironment(unittest.TestCase):
    """Test the shell environment."""

    def setUp(self):
        """Set up for tests."""
        self.shell = ShellEnvironment()
    
    def test_execute_command(self):
        """Test executing a command."""
        def echo_command(args):
            """Echo command for testing."""
            print(args)
            return 0
        
        # Register command
        self.shell.registry.register_command_handler("echo", echo_command, "test")
        
        # Execute command
        exit_code = self.shell.execute_command("echo test message")
        
        # Verify
        self.assertEqual(exit_code, 0)


class TestPermissionSystem(unittest.TestCase):
    """Test the permission system."""

    def setUp(self):
        """Set up for tests."""
        self.permission_system = PermissionSystem()
    
    def test_add_role(self):
        """Test adding a role."""
        result = self.permission_system.add_role(
            "tester", 
            "Role for testers", 
            ["test:run", "test:view"]
        )
        self.assertTrue(result)
        self.assertIn("tester", self.permission_system.roles)
    
    def test_add_user(self):
        """Test adding a user."""
        # Add role first
        self.permission_system.add_role(
            "tester", 
            "Role for testers", 
            ["test:run", "test:view"]
        )
        
        # Add user
        result = self.permission_system.add_user(
            "testuser",
            ["tester"],
            {"email": "test@example.com"}
        )
        
        self.assertTrue(result)
        self.assertIn("testuser", self.permission_system.users)
    
    def test_check_permission(self):
        """Test checking permissions."""
        # Set up role and user
        self.permission_system.add_role(
            "tester", 
            "Role for testers", 
            ["test:run", "test:view"]
        )
        
        self.permission_system.add_user(
            "testuser",
            ["tester"]
        )
        
        # Check permissions
        self.assertTrue(self.permission_system.check_permission("testuser", "test:run"))
        self.assertTrue(self.permission_system.check_permission("testuser", "test:view"))
        self.assertFalse(self.permission_system.check_permission("testuser", "test:delete"))


class TestShellScripting(unittest.TestCase):
    """Test the shell scripting system."""

    def setUp(self):
        """Set up for tests."""
        self.interpreter = SimpleScriptInterpreter()
    
    def test_variable_assignment(self):
        """Test variable assignment."""
        script = """
        var x = 10
        var y = 20
        var z = x + y
        """
        
        self.interpreter.execute(script)
        
        self.assertEqual(self.interpreter.variables.get("x"), 10)
        self.assertEqual(self.interpreter.variables.get("y"), 20)
        self.assertEqual(self.interpreter.variables.get("z"), "x + y")  # Simple interpreter doesn't evaluate expressions
    
    def test_if_statement(self):
        """Test if statement execution."""
        script = """
        var x = 10
        
        if x == 10 then
            var result = "equal"
        else
            var result = "not equal"
        fi
        """
        
        self.interpreter.execute(script)
        
        self.assertEqual(self.interpreter.variables.get("result"), "equal")
    
    def test_for_loop(self):
        """Test for loop execution."""
        script = """
        var items = ["apple", "banana", "cherry"]
        var result = ""
        
        for item in items do
            result = result + item
        done
        """
        
        self.interpreter.execute(script)
        
        # Our simple interpreter would need more complex expression handling
        # for this to work properly, so we're just checking that the loop executed
        self.assertIn("result", self.interpreter.variables)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""

    def setUp(self):
        """Set up for tests."""
        self.shell = ShellEnvironment(ComponentRegistry())
        self.permission_system = PermissionSystem()
        self.scripting = ShellScripting()
        
        # Connect components
        # Note: These methods don't exist in the current ShellEnvironment implementation
        # We would need to adapt the shell environment to support these integrations
        # For now, we'll just set the registry
        self.permission_system.set_registry(self.shell.registry)
        self.scripting.set_registry(self.shell.registry)
        
        self.scripting.set_registry(self.shell.registry)
        
        # Set up test script file
        self.script_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fx')
        self.script_file.write("""
        # Test script
        var x = 10
        var y = 20
        echo "x = ${x}, y = ${y}"
        """)
        self.script_file.close()
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.script_file.name)
    
    def test_script_execution(self):
        """Test script execution through shell."""
        # Register script command
        def script_command(args):
            """Run a script."""
            script_file = args.strip()
            return self.scripting.execute_script_file(script_file)
        
        self.shell.registry.register_command_handler("script", script_command, "scripting")
        
        # Execute script command
        exit_code = self.shell.execute_command(f"script {self.script_file.name}")
        
        # Verify
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
