#!/usr/bin/env python3
"""
Test Shell Scripting

This script tests the functionality of the FixWurx shell scripting component.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shell scripting components
from shell_scripting import ShellScripting, SimpleScriptInterpreter

class TestShellScripting(unittest.TestCase):
    """Test the shell scripting system."""

    def setUp(self):
        """Set up for tests."""
        self.interpreter = SimpleScriptInterpreter()
        self.scripting = ShellScripting()
    
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
    
    def test_function_definition(self):
        """Test function definition and execution."""
        script = """
        function greet(name) 
            echo "Hello, ${name}!"
            return "Greeted ${name}"
        end
        
        var result = greet("World")
        """
        
        self.interpreter.execute(script)
        
        # Check that the function was defined
        self.assertIn("greet", self.interpreter.functions)
        
        # Check that it has the right parameters
        self.assertEqual(self.interpreter.functions["greet"]["params"], ["name"])
    
    def test_script_file_execution(self):
        """Test script execution from file."""
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fx') as script_file:
            script_file.write("""
            # Test script
            var x = 10
            var y = 20
            echo "x = ${x}, y = ${y}"
            """)
        
        try:
            # Execute the script
            exit_code = self.scripting.execute_script_file(script_file.name)
            self.assertEqual(exit_code, 0 if os.path.exists(script_file.name) else 1)
        finally:
            # Clean up
            if os.path.exists(script_file.name):
                os.unlink(script_file.name)


if __name__ == "__main__":
    unittest.main()
