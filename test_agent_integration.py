#!/usr/bin/env python3
"""
test_agent_integration.py
────────────────────────
Script to test the integration of agent commands with the shell environment.

This script creates a component registry, registers the agent commands,
and demonstrates their basic functionality.
"""

import logging
import shell_environment
import agent_shell_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAgentIntegration")

def main():
    """Main function to test agent integration."""
    print("Creating component registry...")
    registry = shell_environment.ComponentRegistry()
    
    print("Registering agent commands...")
    agent_shell_integration.register_agent_commands(registry)
    
    # Get registered commands
    command_handlers = registry.command_handlers
    agent_commands = [cmd for cmd, info in command_handlers.items() 
                      if info.get("component") == "agent"]
    
    print(f"\nRegistered {len(agent_commands)} agent commands:")
    for cmd in sorted(agent_commands):
        print(f"  - {cmd}")
    
    # Get registered aliases
    aliases = registry.aliases
    agent_aliases = []
    for alias, cmd in aliases.items():
        handler = registry.get_command_handler(cmd)
        if handler and handler.get("component") == "agent":
            agent_aliases.append(alias)
    
    print(f"\nRegistered {len(agent_aliases)} agent aliases:")
    for alias in sorted(agent_aliases):
        print(f"  - {alias} -> {aliases[alias]}")
    
    # Test a few commands
    print("\nTesting bug creation command...")
    handler_info = registry.get_command_handler("bug")
    if handler_info:
        handler = handler_info["handler"]
        exit_code = handler("create test-bug-1 Test Bug Title")
        print(f"Exit code: {exit_code}")
    
    print("\nTesting bug list command...")
    handler_info = registry.get_command_handler("bug")
    if handler_info:
        handler = handler_info["handler"]
        exit_code = handler("list")
        print(f"Exit code: {exit_code}")
    
    print("\nTesting agent status command...")
    handler_info = registry.get_command_handler("agent")
    if handler_info:
        handler = handler_info["handler"]
        exit_code = handler("status")
        print(f"Exit code: {exit_code}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
