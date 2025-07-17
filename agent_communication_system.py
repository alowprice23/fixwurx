#!/usr/bin/env python3
"""
agent_communication_system.py
────────────────────────────
Direct agent communication system for the FixWurx shell environment.

This module provides a unified interface for all agents to communicate
directly with users through the shell environment, enabling rich
interactions, progress reporting, and dynamic responses.
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import shlex
import argparse

# Configure logging
logger = logging.getLogger("AgentCommunicationSystem")

# Import conversation logger
import agent_conversation_logger

# Global instance
_instance = None

class AgentCommunicationSystem:
    """
    System for direct agent-to-user communication through the shell environment.
    
    This class provides a unified interface for all agents to:
    1. Output messages directly to the shell
    2. Format responses in various styles (text, JSON, markdown, etc.)
    3. Report progress on long-running tasks
    4. Provide rich interactive responses
    5. Support multi-agent conversations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Agent Communication System.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        
        # Get conversation logger
        self.conversation_logger = agent_conversation_logger.get_instance()
        
        # Agent state tracking
        self.active_agents = {}
        self.agent_sessions = {}
        
        # Output formatting preferences
        self.output_formats = {
            "default": {
                "prefix": "🤖 ",
                "color": "\033[94m",  # Blue
                "reset": "\033[0m"
            },
            "launchpad": {
                "prefix": "🚀 ",
                "color": "\033[92m",  # Green
                "reset": "\033[0m"
            },
            "orchestrator": {
                "prefix": "🎮 ",
                "color": "\033[95m",  # Purple
                "reset": "\033[0m"
            },
            "auditor": {
                "prefix": "🔍 ",
                "color": "\033[93m",  # Yellow
                "reset": "\033[0m"
            },
            "error": {
                "prefix": "❌ ",
                "color": "\033[91m",  # Red
                "reset": "\033[0m"
            },
            "success": {
                "prefix": "✅ ",
                "color": "\033[92m",  # Green
                "reset": "\033[0m"
            },
            "warning": {
                "prefix": "⚠️ ",
                "color": "\033[93m",  # Yellow
                "reset": "\033[0m"
            },
            "info": {
                "prefix": "ℹ️ ",
                "color": "\033[96m",  # Cyan
                "reset": "\033[0m"
            },
            "progress": {
                "prefix": "⏳ ",
                "color": "\033[96m",  # Cyan
                "reset": "\033[0m"
            }
        }
        
        # Thread lock for thread safety
        self._lock = threading.Lock()
        
        # Progress tracking
        self.progress_trackers = {}
        self.progress_lock = threading.Lock()
        
        # Register global instance
        global _instance
        _instance = self
        
        logger.info("Agent Communication System initialized")
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]) -> None:
        """
        Register an agent with the communication system.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., "launchpad", "orchestrator")
            capabilities: List of agent capabilities
        """
        with self._lock:
            self.active_agents[agent_id] = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "capabilities": capabilities,
                "registered_at": time.time(),
                "last_active": time.time(),
                "status": "active"
            }
            
            # Use agent type for formatting if available, otherwise use default
            if agent_type.lower() not in self.output_formats:
                self.output_formats[agent_id] = self.output_formats["default"].copy()
            else:
                self.output_formats[agent_id] = self.output_formats[agent_type.lower()].copy()
            
            logger.info(f"Agent registered: {agent_id} ({agent_type})")
    
    def speak(self, agent_id: str, message: str, message_type: str = "default", 
              session_id: Optional[str] = None, format_output: bool = True) -> None:
        """
        Output a message from an agent directly to the user through the shell.
        
        Args:
            agent_id: The agent's ID
            message: The message to output
            message_type: Message type for formatting
            session_id: Optional session ID for conversation tracking
            format_output: Whether to apply formatting
        """
        if not self.enabled:
            return
        
        # Update agent's last active timestamp
        with self._lock:
            if agent_id in self.active_agents:
                self.active_agents[agent_id]["last_active"] = time.time()
        
        # Get or create session ID
        if not session_id:
            session_id = self._get_or_create_session(agent_id)
        
        # Format the message
        if format_output:
            formatted_message = self._format_message(agent_id, message, message_type)
        else:
            formatted_message = message
        
        # Print the message to the console
        print(formatted_message)
        
        # Log the message if it's not a progress update
        if message_type != "progress":
            self.conversation_logger.log_agent_response(
                session_id=session_id,
                agent_id=agent_id,
                response=message,
                success=True,
                llm_used=False
            )
    
    def speak_error(self, agent_id: str, error_message: str, session_id: Optional[str] = None) -> None:
        """
        Output an error message from an agent.
        
        Args:
            agent_id: The agent's ID
            error_message: The error message
            session_id: Optional session ID for conversation tracking
        """
        self.speak(agent_id, error_message, "error", session_id)
    
    def speak_success(self, agent_id: str, success_message: str, session_id: Optional[str] = None) -> None:
        """
        Output a success message from an agent.
        
        Args:
            agent_id: The agent's ID
            success_message: The success message
            session_id: Optional session ID for conversation tracking
        """
        self.speak(agent_id, success_message, "success", session_id)
    
    def speak_warning(self, agent_id: str, warning_message: str, session_id: Optional[str] = None) -> None:
        """
        Output a warning message from an agent.
        
        Args:
            agent_id: The agent's ID
            warning_message: The warning message
            session_id: Optional session ID for conversation tracking
        """
        self.speak(agent_id, warning_message, "warning", session_id)
    
    def speak_info(self, agent_id: str, info_message: str, session_id: Optional[str] = None) -> None:
        """
        Output an informational message from an agent.
        
        Args:
            agent_id: The agent's ID
            info_message: The informational message
            session_id: Optional session ID for conversation tracking
        """
        self.speak(agent_id, info_message, "info", session_id)
    
    def start_progress(self, agent_id: str, task_id: str, description: str, 
                      total_steps: int = 100, session_id: Optional[str] = None) -> str:
        """
        Start tracking progress for a long-running task.
        
        Args:
            agent_id: The agent's ID
            task_id: Unique task identifier
            description: Task description
            total_steps: Total number of steps
            session_id: Optional session ID for conversation tracking
            
        Returns:
            Progress tracker ID
        """
        if not self.enabled:
            return ""
        
        # Get or create session ID
        if not session_id:
            session_id = self._get_or_create_session(agent_id)
        
        # Create progress tracker ID
        tracker_id = f"{agent_id}_{task_id}_{int(time.time())}"
        
        with self.progress_lock:
            self.progress_trackers[tracker_id] = {
                "agent_id": agent_id,
                "task_id": task_id,
                "description": description,
                "total_steps": total_steps,
                "current_step": 0,
                "status": "in_progress",
                "start_time": time.time(),
                "last_update": time.time(),
                "session_id": session_id
            }
        
        # Output initial progress message
        self.speak(agent_id, f"Starting: {description} (0%)", "progress", session_id)
        
        return tracker_id
    
    def update_progress(self, tracker_id: str, current_step: int, 
                       status_message: Optional[str] = None) -> None:
        """
        Update progress for a long-running task.
        
        Args:
            tracker_id: Progress tracker ID
            current_step: Current step number
            status_message: Optional status message
        """
        if not self.enabled:
            return
        
        with self.progress_lock:
            if tracker_id not in self.progress_trackers:
                logger.warning(f"Progress tracker not found: {tracker_id}")
                return
            
            tracker = self.progress_trackers[tracker_id]
            agent_id = tracker["agent_id"]
            description = tracker["description"]
            total_steps = tracker["total_steps"]
            session_id = tracker["session_id"]
            
            # Update tracker
            tracker["current_step"] = current_step
            tracker["last_update"] = time.time()
            
            # Calculate percentage
            percentage = min(100, int((current_step / total_steps) * 100))
            
            # Generate progress message
            if status_message:
                progress_message = f"{description}: {status_message} ({percentage}%)"
            else:
                progress_message = f"{description} ({percentage}%)"
            
            # Output progress message
            self.speak(agent_id, progress_message, "progress", session_id)
    
    def complete_progress(self, tracker_id: str, success: bool = True, 
                         completion_message: Optional[str] = None) -> None:
        """
        Complete a progress tracker.
        
        Args:
            tracker_id: Progress tracker ID
            success: Whether the task completed successfully
            completion_message: Optional completion message
        """
        if not self.enabled:
            return
        
        with self.progress_lock:
            if tracker_id not in self.progress_trackers:
                logger.warning(f"Progress tracker not found: {tracker_id}")
                return
            
            tracker = self.progress_trackers[tracker_id]
            agent_id = tracker["agent_id"]
            description = tracker["description"]
            session_id = tracker["session_id"]
            
            # Update tracker
            tracker["current_step"] = tracker["total_steps"]
            tracker["status"] = "completed" if success else "failed"
            tracker["end_time"] = time.time()
            
            # Generate completion message
            if completion_message:
                message = completion_message
            else:
                message = f"{description} completed successfully" if success else f"{description} failed"
            
            # Output completion message with appropriate type
            message_type = "success" if success else "error"
            self.speak(agent_id, message, message_type, session_id)
            
            # Clean up tracker
            del self.progress_trackers[tracker_id]
    
    def agent_is_active(self, agent_id: str) -> bool:
        """
        Check if an agent is active.
        
        Args:
            agent_id: The agent's ID
            
        Returns:
            True if the agent is active, False otherwise
        """
        with self._lock:
            return agent_id in self.active_agents and self.active_agents[agent_id]["status"] == "active"
    
    def deactivate_agent(self, agent_id: str) -> None:
        """
        Deactivate an agent.
        
        Args:
            agent_id: The agent's ID
        """
        with self._lock:
            if agent_id in self.active_agents:
                self.active_agents[agent_id]["status"] = "inactive"
                self.active_agents[agent_id]["last_active"] = time.time()
                
                # Log deactivation
                logger.info(f"Agent deactivated: {agent_id}")
    
    def _format_message(self, agent_id: str, message: str, message_type: str) -> str:
        """
        Format a message with appropriate styling.
        
        Args:
            agent_id: The agent's ID
            message: The message to format
            message_type: Message type for formatting
            
        Returns:
            Formatted message
        """
        # Get formatting options
        if message_type in self.output_formats:
            format_options = self.output_formats[message_type]
        elif agent_id in self.output_formats:
            format_options = self.output_formats[agent_id]
        else:
            format_options = self.output_formats["default"]
        
        # Get agent name for display
        agent_name = agent_id
        with self._lock:
            if agent_id in self.active_agents:
                agent_type = self.active_agents[agent_id].get("agent_type", "")
                if agent_type:
                    agent_name = agent_type.capitalize()
        
        # Format the message
        prefix = format_options.get("prefix", "")
        color = format_options.get("color", "")
        reset = format_options.get("reset", "")
        
        # Check if color should be applied (disable for non-TTY)
        use_color = sys.stdout.isatty()
        
        if use_color:
            return f"{color}{prefix}[{agent_name}] {message}{reset}"
        else:
            return f"{prefix}[{agent_name}] {message}"
    
    def _get_or_create_session(self, agent_id: str) -> str:
        """
        Get an existing session or create a new one for an agent.
        
        Args:
            agent_id: The agent's ID
            
        Returns:
            Session ID
        """
        with self._lock:
            # Check if agent already has an active session
            if agent_id in self.agent_sessions:
                return self.agent_sessions[agent_id]
            
            # Create a new session
            session_id = f"shell_{int(time.time())}_{agent_id}"
            self.agent_sessions[agent_id] = session_id
            
            return session_id


def get_instance(config: Optional[Dict[str, Any]] = None) -> AgentCommunicationSystem:
    """
    Get the singleton instance of the Agent Communication System.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        AgentCommunicationSystem instance
    """
    global _instance
    if _instance is None:
        _instance = AgentCommunicationSystem(config)
    return _instance


# Command handler for agent communication
def agent_speak_command(args: str) -> int:
    """
    Command to make an agent speak directly to the user.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Make an agent speak to the user")
    parser.add_argument("agent_id", help="Agent ID")
    parser.add_argument("message", nargs='+', help="Message to speak")
    parser.add_argument("--type", "-t", choices=["default", "error", "success", "warning", "info"], 
                        default="default", help="Message type")
    parser.add_argument("--raw", "-r", action="store_true", help="Output raw message without formatting")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get agent communication system
    acs = get_instance()
    
    # Combine message parts
    message = " ".join(cmd_args.message)
    
    # Speak the message
    acs.speak(
        agent_id=cmd_args.agent_id,
        message=message,
        message_type=cmd_args.type,
        format_output=not cmd_args.raw
    )
    
    return 0


def register_communication_commands(registry) -> None:
    """
    Register agent communication commands with the shell environment.
    
    Args:
        registry: Component registry
    """
    logger.info("Registering agent communication commands")
    
    # Initialize agent communication system
    config = {
        "enabled": True
    }
    
    acs = get_instance(config)
    registry.register_component("agent_communication_system", acs)
    
    # Register agent communication commands
    registry.register_command_handler("agent:speak", agent_speak_command, "agent")
    
    # Register helpful aliases
    registry.register_alias("speak", "agent:speak")
    
    logger.info("Agent communication commands registered successfully")
