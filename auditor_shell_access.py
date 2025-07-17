#!/usr/bin/env python3
"""
Auditor Shell Access

This module provides the Auditor Agent with read-only shell access to run
diagnostic commands. It wraps the command executor with a restricted interface
that only allows read-only operations.
"""

import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuditorShellAccess")

class AuditorShellAccess:
    """
    Provides a restricted shell access interface for the Auditor Agent.
    
    This class ensures that the Auditor Agent can only execute read-only
    commands that don't modify the system state. It serves as a security
    layer between the Auditor Agent and the command execution environment.
    """
    
    def __init__(self, command_executor: Any = None, registry: Any = None):
        """
        Initialize the Auditor Shell Access.
        
        Args:
            command_executor: The command executor component
            registry: The component registry for accessing other components
        """
        self.command_executor = command_executor
        self.registry = registry
        self.allowed_commands = self._get_allowed_commands()
        logger.info("Auditor Shell Access initialized")
    
    def _get_allowed_commands(self) -> Dict[str, List[str]]:
        """
        Define the list of allowed commands for the Auditor Agent.
        
        Returns:
            Dictionary of allowed command categories and their commands
        """
        # Only read-only commands are allowed
        return {
            "system": [
                "ls", "cat", "pwd", "echo", "grep", "find", "wc",
                "ps", "top", "htop", "df", "du", "free", "uname",
                "stat", "head", "tail", "file", "which", "whereis"
            ],
            "fixwurx": [
                "analyze", "scan", "diagnose", "stats", "metrics",
                "report", "errors", "sensors", "monitor", "trace",
                "audit", "status", "tree", "dashboard"
            ],
            "git": [
                "git status", "git log", "git show", "git diff", 
                "git ls-files", "git branch", "git remote -v"
            ]
        }
    
    def is_command_allowed(self, command: str) -> bool:
        """
        Check if a command is allowed to be executed by the Auditor Agent.
        
        Args:
            command: The command to check
            
        Returns:
            True if the command is allowed, False otherwise
        """
        # Extract the base command (before any arguments)
        base_command = command.strip().split(" ")[0]
        
        # Check if the base command is in any of the allowed categories
        for category, commands in self.allowed_commands.items():
            if base_command in commands:
                return True
            
            # Check for compound commands (e.g., "git status")
            for allowed_command in commands:
                if command.strip().startswith(allowed_command):
                    return True
        
        return False
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a command on behalf of the Auditor Agent.
        
        Args:
            command: The command to execute
            
        Returns:
            Result dictionary with output, success status, and error (if any)
        """
        if not self.command_executor:
            logger.error("Command Executor not available")
            return {
                "success": False,
                "error": "Command Executor not available",
                "output": ""
            }
        
        # Check if command is allowed
        if not self.is_command_allowed(command):
            logger.warning(f"Command not allowed for Auditor Agent: {command}")
            return {
                "success": False,
                "error": f"Command not allowed for Auditor Agent: {command}",
                "output": ""
            }
        
        # Add read-only flag to the execution context
        try:
            context = {
                "read_only": True,
                "agent_id": "auditor",
                "timestamp": time.time()
            }
            
            # Execute the command with read-only flag
            result = self.command_executor.execute(command, "auditor", context=context)
            return result
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    def execute_diagnostic_script(self, script_content: str) -> Dict[str, Any]:
        """
        Execute a diagnostic script on behalf of the Auditor Agent.
        
        Args:
            script_content: The content of the script to execute
            
        Returns:
            Result dictionary with output, success status, and error (if any)
        """
        if not self.command_executor:
            logger.error("Command Executor not available")
            return {
                "success": False,
                "error": "Command Executor not available",
                "output": ""
            }
        
        # Check each command in the script
        lines = script_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if not self.is_command_allowed(line):
                logger.warning(f"Script contains disallowed command: {line}")
                return {
                    "success": False,
                    "error": f"Script contains disallowed command: {line}",
                    "output": ""
                }
        
        # Add read-only flag to the execution context
        try:
            context = {
                "read_only": True,
                "agent_id": "auditor",
                "timestamp": time.time()
            }
            
            # Execute the script with read-only flag
            result = self.command_executor.execute_script(script_content, "auditor", context=context)
            return result
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }

# Singleton instance
_instance = None

def get_instance(registry: Any = None) -> AuditorShellAccess:
    """
    Get or create the singleton instance of the Auditor Shell Access.
    
    Args:
        registry: The component registry for accessing other components
        
    Returns:
        AuditorShellAccess instance
    """
    global _instance
    if _instance is None:
        # Get command executor from registry
        command_executor = None
        if registry:
            command_executor = registry.get_component("command_executor")
        
        _instance = AuditorShellAccess(command_executor, registry)
    
    return _instance

def register(registry: Any) -> None:
    """
    Register the Auditor Shell Access with the component registry.
    
    Args:
        registry: The component registry
    """
    if registry:
        instance = get_instance(registry)
        registry.register_component("auditor_shell_access", instance)
        logger.info("Registered Auditor Shell Access with component registry")
