#!/usr/bin/env python3
"""
Test script for decision tree logic integrated with the shell.

This script tests the integration of the decision tree logic with the shell
by simulating a shell environment and running decision tree commands.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_decision_tree.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TestDecisionTree")

class MockShell:
    """Mock shell for testing."""
    
    def __init__(self):
        """Initialize the mock shell."""
        self.commands = {}
        self.variables = {}
        self.components = {}
    
    def register_command(self, name, func, description):
        """Register a command with the shell."""
        self.commands[name] = {
            "func": func,
            "description": description
        }
        logger.info(f"Registered command: {name}")
    
    def set_variable(self, name, value):
        """Set a shell variable."""
        self.variables[name] = value
    
    def get_variable(self, name):
        """Get a shell variable."""
        return self.variables.get(name)
    
    def print(self, message):
        """Print a message."""
        print(message)
    
    def print_error(self, message):
        """Print an error message."""
        print(f"ERROR: {message}")
    
    def add_component(self, name, component):
        """Add a component to the shell."""
        self.components[name] = component
    
    def get_component(self, name):
        """Get a component from the shell."""
        return self.components.get(name)
    
    def run_command(self, command, args=None):
        """Run a command with arguments."""
        if command in self.commands:
            func = self.commands[command]["func"]
            if args is None:
                args = []
            return func(self, args)
        else:
            self.print_error(f"Command not found: {command}")
            return 1

def main():
    """Main function."""
    print("Test Decision Tree with Shell\n")
    
    # Create sample buggy file for testing
    sample_code = """
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)  # Bug: Potential division by zero

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            # Bug: Using append incorrectly
            result.append = item * 2
    return result
"""
    
    file_path = "sample_buggy.py"
    with open(file_path, 'w') as f:
        f.write(sample_code)
    
    print(f"Created sample file: {file_path}")
    
    # Create mock shell
    shell = MockShell()
    
    # Make the mock shell available to the decision tree integration
    sys.modules["__main__"].registry = shell
    
    try:
        # Import decision tree commands and register them
        from decision_tree_commands import register_commands
        register_commands(shell)
        
        print("\nRegistered Commands:")
        for name, cmd in shell.commands.items():
            print(f"  {name}: {cmd['description']}")
        
        # Create necessary directories
        os.makedirs(".triangulum/results", exist_ok=True)
        os.makedirs(".triangulum/patches", exist_ok=True)
        os.makedirs(".triangulum/verification_results", exist_ok=True)
        os.makedirs(".triangulum/logs", exist_ok=True)
        
        # Test bug_identify command
        print("\n===== Testing bug_identify command =====")
        shell.run_command("bug_identify", [file_path])
        
        # Get the last bug ID
        bug_id = shell.get_variable("last_bug_id")
        if bug_id:
            print(f"\nFound bug ID: {bug_id}")
            
            # Test bug_generate_paths command
            print("\n===== Testing bug_generate_paths command =====")
            shell.run_command("bug_generate_paths", [bug_id])
            
            # Test bug_select_path command
            print("\n===== Testing bug_select_path command =====")
            shell.run_command("bug_select_path", [])
            
            # Test bug_fix command
            print("\n===== Testing bug_fix command =====")
            shell.run_command("bug_fix", [file_path])
        else:
            print("\nNo bug ID found, cannot continue tests")
        
        # Test bug_demo command
        print("\n===== Testing bug_demo command =====")
        shell.run_command("bug_demo", [])
        
        print("\nTests completed")
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    main()
