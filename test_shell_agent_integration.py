#!/usr/bin/env python3
"""
test_shell_agent_integration.py
───────────────────────────────
Test script that demonstrates the integration of the Agent System
with the Shell Environment, showing how to use the agent commands
through the shell's command pipeline.
"""

import os
import sys
import logging
from pathlib import Path

import shell_environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestShellAgentIntegration")

def main():
    """Main test function."""
    print("\n=== FixWurx Agent System Shell Integration Test ===\n")
    
    # Create shell environment
    print("Creating shell environment...")
    registry = shell_environment.ComponentRegistry()
    shell_environment.create_base_commands(registry)
    shell_environment.load_command_modules(registry)
    
    # Create command pipeline
    command_pipeline = shell_environment.CommandPipeline(registry)
    registry.register_component("command_pipeline", command_pipeline)
    
    # Define helper function to run commands
    def run_command(cmd):
        print(f"\n>>> {cmd}")
        parsed_cmd = command_pipeline.parse_command_line(cmd)
        exit_code, output = command_pipeline.execute_pipeline(parsed_cmd)
        if output:
            print(output.rstrip())
        return exit_code
    
    # Test basic commands
    print("\n--- Testing Basic Commands ---")
    run_command("agent status")
    
    # Test bug creation and listing
    print("\n--- Testing Bug Management ---")
    run_command("bug create test-bug-1 'Sample bug for testing'")
    run_command("bug create urgent-bug 'Critical issue that needs fixing'")
    run_command("bug list")
    run_command("bugs")  # Test alias
    
    # Test bug details
    print("\n--- Testing Bug Details ---")
    run_command("bug show test-bug-1")
    run_command("bug update test-bug-1 severity=high")
    run_command("bug update test-bug-1 tag=regression")
    run_command("bug show test-bug-1")
    
    # Test plan generation
    print("\n--- Testing Plan Generation ---")
    run_command("plan generate test-bug-1")
    run_command("plan select test-bug-1")
    
    # Test observer agent
    print("\n--- Testing Observer Agent ---")
    run_command("observe analyze test-bug-1")
    
    # Test analyst agent
    print("\n--- Testing Analyst Agent ---")
    run_command("analyze patch test-bug-1")
    
    # Test verifier agent
    print("\n--- Testing Verifier Agent ---")
    run_command("verify test test-bug-1")
    
    # Test agent tree
    print("\n--- Testing Agent Tree ---")
    run_command("agent tree")
    run_command("agents")  # Test alias
    
    # Test agent metrics
    print("\n--- Testing Agent Metrics ---")
    run_command("agent metrics")
    
    # Final status check
    print("\n--- Final Status Check ---")
    run_command("bug show test-bug-1")
    run_command("agent status")
    
    print("\n=== Test completed successfully ===")

if __name__ == "__main__":
    main()
