#!/usr/bin/env python3
"""
FixWurx LLM Shell Capabilities Demo

This script demonstrates the capabilities of the FixWurx LLM Shell by executing
various commands and showcasing the features of each component.
"""

import os
import sys
import time
import subprocess
import shutil
import json
from typing import List, Dict, Any, Optional

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"{Colors.HEADER}{Colors.BOLD}{title}{Colors.ENDC}")
    print("=" * 80 + "\n")

def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 60 + "\n")

def print_step(step: str, description: str) -> None:
    """Print a step in the demo."""
    print(f"{Colors.CYAN}[STEP] {step}{Colors.ENDC}")
    print(f"{description}\n")

def print_command(command: str) -> None:
    """Print a command that will be executed."""
    print(f"{Colors.YELLOW}$ {command}{Colors.ENDC}")

def print_output(output: str) -> None:
    """Print command output."""
    print(f"{Colors.GREEN}{output}{Colors.ENDC}")

def print_error(error: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}ERROR: {error}{Colors.ENDC}")

def run_command(command: str) -> str:
    """
    Run a shell command and return the output.
    
    Args:
        command: The command to run
        
    Returns:
        Command output as string
    """
    try:
        print_command(command)
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        print_output(output)
        return output
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}")
        print_error(e.stderr)
        return ""

def run_fx_command(command: str) -> str:
    """
    Run a command using the fx shell.
    
    Args:
        command: The command to run
        
    Returns:
        Command output as string
    """
    if os.name == 'nt':  # Windows
        return run_command(f"fx.bat -e \"{command}\"")
    else:  # Unix/Linux/macOS
        return run_command(f"./fx -e \"{command}\"")

def run_integration_test() -> bool:
    """
    Run the integration test.
    
    Returns:
        True if test passed, False otherwise
    """
    print_step("Running integration test", "This will test the integration of all components")
    
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run("python test_integration.py", shell=True, check=False)
        else:
            result = subprocess.run("python3 test_integration.py", shell=True, check=False)
        
        return result.returncode == 0
    except Exception as e:
        print_error(f"Failed to run integration test: {e}")
        return False

def demonstrate_conversational_interface() -> None:
    """Demonstrate the Conversational Interface."""
    print_subheader("Conversational Interface Demo")
    
    print_step("Simple query", "Ask a simple question to the conversational interface")
    run_fx_command("What can you help me with?")
    
    print_step("Formatting demo", "Demonstrate different formatting capabilities")
    run_fx_command("Show me examples of code, tables, and lists")

def demonstrate_planning_engine() -> None:
    """Demonstrate the Planning Engine."""
    print_subheader("Planning Engine Demo")
    
    print_step("Goal deconstruction", "Break down a high-level goal into steps")
    run_fx_command("Create a backup of important log files")
    
    print_step("Script generation", "Generate a script to automate a task")
    run_fx_command("Write a script to find large files in the current directory")

def demonstrate_command_execution() -> None:
    """Demonstrate the Command Execution Environment."""
    print_subheader("Command Execution Demo")
    
    print_step("Safe command execution", "Execute a simple command")
    run_fx_command("echo 'Hello from the secure execution environment'")
    
    print_step("Permission system", "Demonstrate permission checks")
    run_fx_command("rm -rf /")  # This should be blocked by the permission system

def demonstrate_script_library() -> None:
    """Demonstrate the Script Library."""
    print_subheader("Script Library Demo")
    
    # Create a test script
    print_step("Adding a script", "Add a script to the library")
    test_script = """
    #!/usr/bin/env bash
    # A simple test script
    echo "Hello from the script library!"
    echo "Current directory: $(pwd)"
    echo "Files: $(ls -la)"
    """
    
    # Create a temporary script file
    with open("temp_script.sh", "w") as f:
        f.write(test_script)
    
    run_fx_command("add_script temp_script.sh --name 'Test Script' --description 'A simple test script' --tags 'test,demo'")
    
    # Clean up
    if os.path.exists("temp_script.sh"):
        os.remove("temp_script.sh")
    
    print_step("Listing scripts", "List all scripts in the library")
    run_fx_command("list_scripts")
    
    print_step("Running a script", "Run a script from the library")
    run_fx_command("run_script 'Test Script'")

def demonstrate_conversation_logger() -> None:
    """Demonstrate the Conversation Logger."""
    print_subheader("Conversation Logger Demo")
    
    print_step("Starting a conversation", "Start a new conversation with the system")
    run_fx_command("start_conversation")
    
    print_step("Adding messages", "Add messages to the conversation")
    run_fx_command("add_message 'Tell me about the conversation logger'")
    
    print_step("Getting conversation history", "Retrieve the conversation history")
    run_fx_command("get_conversation_history")

def demonstrate_collaborative_improvement() -> None:
    """Demonstrate the Collaborative Improvement Framework."""
    print_subheader("Collaborative Improvement Demo")
    
    print_step("Pattern detection", "Demonstrate pattern detection")
    # Execute a sequence of commands multiple times to trigger pattern detection
    commands = [
        "ls -la",
        "grep 'error' log.txt",
        "cat log.txt | grep 'error' > errors.txt"
    ]
    
    print("Executing command sequence multiple times to trigger pattern detection...")
    for _ in range(2):
        for cmd in commands:
            run_fx_command(cmd)
    
    print_step("Script proposal", "Show script proposals based on detected patterns")
    run_fx_command("show_proposals")
    
    print_step("Peer review", "Demonstrate the peer review workflow")
    run_fx_command("review_proposal --id latest --approve --comment 'Looks good!'")

def demonstrate_launchpad() -> None:
    """Demonstrate the Launchpad."""
    print_subheader("Launchpad Demo")
    
    print_step("Component status", "Show the status of all components")
    run_fx_command("show_component_status")
    
    print_step("Configuration", "Show the current configuration")
    run_fx_command("show_config")

def main() -> None:
    """Main function to run the demo."""
    print_header("FixWurx LLM Shell Capabilities Demo")
    
    print("This demo will showcase the capabilities of the FixWurx LLM Shell by demonstrating each component.")
    print("Press Enter to begin...")
    input()
    
    # Run integration test
    test_passed = run_integration_test()
    if test_passed:
        print(f"\n{Colors.GREEN}✅ Integration test passed!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}❌ Integration test failed!{Colors.ENDC}")
        print("Continuing with the demo anyway...\n")
    
    # Demonstrate each component
    demonstrate_launchpad()
    demonstrate_conversational_interface()
    demonstrate_planning_engine()
    demonstrate_command_execution()
    demonstrate_script_library()
    demonstrate_conversation_logger()
    demonstrate_collaborative_improvement()
    
    print_header("Demo Complete!")
    print("You have seen a demonstration of the following components:")
    print("1. Launchpad & System Bootstrap")
    print("2. Conversational Interface")
    print("3. Intent Recognition and Planning Engine")
    print("4. Secure Command Execution Environment")
    print("5. State and Knowledge Repository (Script Library & Conversation Logger)")
    print("6. Collaborative Improvement Framework")
    
    print("\nTo use the FixWurx LLM Shell interactively, run:")
    if os.name == 'nt':  # Windows
        print_command("fx.bat")
    else:  # Unix/Linux/macOS
        print_command("./fx")
    
    print("\nThank you for using FixWurx LLM Shell!")

if __name__ == "__main__":
    main()
