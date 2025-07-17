#!/usr/bin/env python3
"""
test_meta_agent_integration.py
───────────────────────────────
Test script that demonstrates the Meta Agent capabilities
through the shell environment.
"""

import os
import sys
import logging
from pathlib import Path

import shell_environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMetaAgentIntegration")

def main():
    """Main test function."""
    print("\n=== FixWurx Meta Agent Integration Test ===\n")
    
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
    
    # Test Meta Agent status
    print("\n--- Testing Meta Agent Status ---")
    run_command("meta status")
    
    # Test registered agents
    print("\n--- Testing Registered Agents ---")
    run_command("meta agents")
    
    # Test agent network
    print("\n--- Testing Agent Network ---")
    run_command("agent network")
    
    # Test meta insights
    print("\n--- Testing Meta Insights ---")
    run_command("meta insights")
    
    # Test agent coordination
    print("\n--- Testing Agent Coordination ---")
    run_command("meta coordinate observer,analyst,verifier task-123 bug-fix")
    
    # Test meta conflicts
    print("\n--- Testing Meta Conflicts ---")
    run_command("meta conflicts")
    
    print("\n=== Meta Agent Test completed successfully ===")

if __name__ == "__main__":
    main()
