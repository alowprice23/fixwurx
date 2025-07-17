#!/usr/bin/env python3
"""
agent_shell_integration.py
─────────────────────────
Integration module for the Agent System with the Shell Environment.

This module registers the agent commands with the shell environment,
initializes the agent system, and provides a clean interface for command execution.
"""

import logging
import sys
from typing import Dict, Any

# Import agent commands
import agent_commands

# Configure logging
logger = logging.getLogger("AgentShellIntegration")

def register_agent_commands(registry) -> None:
    """
    Register agent commands with the shell environment.
    
    Args:
        registry: Component registry
    """
    logger.info("Registering agent commands")
    
    # Initialize agent system with the system configuration
    system_config = {
        "agent_system": {
            "enabled": True,
            "solutions-per-bug": 3,
            "max-path-depth": 5,
            "fallback-threshold": 0.3,
            "learning-rate": 0.1,
            "pattern-threshold": 0.7,
            "history-limit": 1000
        }
    }
    
    # Initialize agent system
    success = agent_commands.initialize(system_config)
    if not success:
        logger.error("Failed to initialize agent system")
        return
    
    # Register commands with the shell
    commands = agent_commands.register()
    
    for command_name, handler in commands.items():
        registry.register_command_handler(command_name, handler, "agent")
    
    # Register agent-specific commands
    registry.register_command_handler("agent:status", agent_commands.agent_status_command, "agent")
    registry.register_command_handler("agent:tree", agent_commands.agent_tree_command, "agent")
    registry.register_command_handler("agent:metrics", agent_commands.agent_metrics_command, "agent")
    
    registry.register_command_handler("plan:generate", agent_commands.plan_generate_command, "agent")
    registry.register_command_handler("plan:select", agent_commands.plan_select_command, "agent")
    
    registry.register_command_handler("observe:analyze", lambda args: agent_commands.observe_command(f"analyze {args}"), "agent")
    registry.register_command_handler("analyze:patch", lambda args: agent_commands.analyze_command(f"patch {args}"), "agent")
    registry.register_command_handler("verify:test", lambda args: agent_commands.verify_command(f"test {args}"), "agent")
    
    registry.register_command_handler("bug:create", agent_commands.bug_create_command, "agent")
    registry.register_command_handler("bug:list", agent_commands.bug_list_command, "agent")
    registry.register_command_handler("bug:show", agent_commands.bug_show_command, "agent")
    registry.register_command_handler("bug:update", agent_commands.bug_update_command, "agent")
    
    # Register helpful aliases
    registry.register_alias("bugs", "bug list")
    registry.register_alias("agents", "agent tree")
    
    logger.info("Agent commands registered successfully")
