#!/usr/bin/env python3
"""
agent_shell_integration.py
─────────────────────────
Integration module for the Agent System with the Shell Environment.

This module registers the agent commands with the shell environment,
initializes the agent system, and provides a clean interface for command execution
and direct agent-to-user communication.
"""

import logging
import sys
from typing import Dict, Any

# Import agent commands and communication modules
import agent_commands
import agent_conversation_logger
import conversation_commands
import agent_communication_system
import agent_progress_tracking

# Configure logging
logger = logging.getLogger("AgentShellIntegration")

def register_agent_commands(registry) -> None:
    """
    Register agent commands with the shell environment.
    
    Args:
        registry: Component registry
    """
    logger.info("Registering agent commands")
    
    # Initialize conversation logger first
    conversation_commands.register_conversation_commands(registry)
    conversation_logger = registry.get_component("conversation_logger")
    
    # Initialize agent communication system
    agent_communication_system.register_communication_commands(registry)
    communication_system = registry.get_component("agent_communication_system")
    
    # Initialize progress tracking system
    agent_progress_tracking.register_progress_commands(registry)
    progress_tracker = registry.get_component("progress_tracker")
    
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
        },
        "conversation_logger": conversation_logger,
        "communication_system": communication_system,
        "progress_tracker": progress_tracker
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
    registry.register_alias("progress", "agent:progress")
    registry.register_alias("say", "agent:speak")
    
    # Register standard agents with the communication system
    communication_system.register_agent("launchpad", "launchpad", ["orchestration", "planning", "task_management"])
    communication_system.register_agent("orchestrator", "orchestrator", ["coordination", "resource_management"])
    communication_system.register_agent("auditor", "auditor", ["monitoring", "verification", "reporting"])
    communication_system.register_agent("triangulum", "triangulum", ["analysis", "problem_solving"])
    communication_system.register_agent("neural_matrix", "neural_matrix", ["learning", "prediction", "optimization"])
    
    # Hook the conversation logger and communication system to all agent commands
    _wrap_agent_commands_with_logger(registry, conversation_logger, communication_system)
    
    logger.info("Agent commands registered successfully")

def _wrap_agent_commands_with_logger(registry, conversation_logger, communication_system):
    """
    Wrap all agent command handlers with conversation logging functionality
    and integrate with the agent communication system.
    
    Args:
        registry: Component registry
        conversation_logger: Conversation logger instance
        communication_system: Agent communication system instance
    """
    # Get all registered command handlers
    command_handlers = registry.command_handlers.copy()
    
    # Wrap each agent command handler
    for command, handler_info in command_handlers.items():
        if handler_info.get("component") == "agent":
            original_handler = handler_info["handler"]
            
            # Create a wrapper that logs conversations and uses the communication system
            def create_logging_wrapper(cmd, orig_handler):
                def logging_wrapper(args):
                    # Determine the agent ID
                    agent_id = None
                    if ":" in cmd:
                        agent_id = cmd.split(":")[0]
                    else:
                        # Default to appropriate agent based on command context
                        if cmd.startswith("bug") or cmd.startswith("verify"):
                            agent_id = "triangulum"
                        elif cmd.startswith("plan"):
                            agent_id = "orchestrator"
                        elif cmd.startswith("analyze"):
                            agent_id = "auditor"
                        elif cmd.startswith("neural") or cmd.startswith("train"):
                            agent_id = "neural_matrix"
                        else:
                            agent_id = "launchpad"
                    
                    # Log the user message
                    session_id = conversation_logger.log_user_message(
                        user_input=args,
                        command=cmd,
                        agent_id=agent_id
                    )
                    
                    # Use communication system to inform about command execution
                    if communication_system.agent_is_active(agent_id):
                        communication_system.speak_info(
                            agent_id=agent_id,
                            info_message=f"Executing command: {cmd} {args}",
                            session_id=session_id
                        )
                    
                    # Execute the original handler
                    try:
                        # Check if this is an LLM-based command (might use openai)
                        llm_used = any(keyword in str(orig_handler.__code__.co_consts) 
                                       for keyword in ["openai", "llm", "prompt", "gpt"])
                        
                        # Execute the command
                        result = orig_handler(args)
                        success = True if isinstance(result, int) and result == 0 else False
                        
                        # Format response for display
                        response_message = result if not isinstance(result, int) else f"Command completed with exit code {result}"
                        
                        # Log the agent response
                        conversation_logger.log_agent_response(
                            session_id=session_id,
                            agent_id=agent_id or "shell",
                            response=response_message,
                            success=success,
                            llm_used=llm_used
                        )
                        
                        # Use communication system to display the result
                        if success and communication_system.agent_is_active(agent_id):
                            if isinstance(result, int):
                                communication_system.speak_success(
                                    agent_id=agent_id,
                                    success_message=f"Command completed successfully",
                                    session_id=session_id
                                )
                            else:
                                communication_system.speak(
                                    agent_id=agent_id,
                                    message=str(response_message),
                                    message_type="default",
                                    session_id=session_id
                                )
                        
                        return result
                    except Exception as e:
                        # Log the error response
                        error_message = f"Error: {str(e)}"
                        conversation_logger.log_agent_response(
                            session_id=session_id,
                            agent_id=agent_id or "shell",
                            response=error_message,
                            success=False,
                            llm_used=False
                        )
                        
                        # Use communication system to display the error
                        if communication_system.agent_is_active(agent_id):
                            communication_system.speak_error(
                                agent_id=agent_id,
                                error_message=error_message,
                                session_id=session_id
                            )
                        
                        raise
                
                return logging_wrapper
            
            # Replace the original handler with the wrapped version
            registry.command_handlers[command]["handler"] = create_logging_wrapper(command, original_handler)
            logger.info(f"Wrapped command handler with conversation logging: {command}")
