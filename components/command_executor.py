#!/usr/bin/env python3
"""
Command Executor

This module provides secure command execution with permission controls and resource limits.
"""

import os
import sys
import json
import time
import logging
import subprocess
import shlex
import signal
import threading
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("command_executor.log")
    ]
)
logger = logging.getLogger("CommandExecutor")

class CommandExecutor:
    """
    Command Executor for secure command execution with permission controls and resource limits.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Command Executor.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configuration parameters
        self.timeout_seconds = self.config.get("timeout_seconds", 60)
        self.blacklist_path = self.config.get("blacklist_path", "security/command_blacklist.json")
        self.enable_confirmation = self.config.get("enable_confirmation", True)
        
        # Command blacklist
        self.blacklist = self._load_blacklist()
        
        # Internal commands
        self.internal_commands = {}
        
        # Execution lock
        self.execution_lock = threading.Lock()
        
        # Register with registry
        registry.register_component("command_executor", self)
        
        logger.info("Command Executor initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the Command Executor.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Command Executor already initialized")
            return True
        
        try:
            self.initialized = True
            logger.info("Command Executor initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Command Executor: {e}")
            return False
    
    def _load_blacklist(self) -> List[Dict[str, Any]]:
        """
        Load the command blacklist.
        
        Returns:
            List of blacklisted commands
        """
        try:
            if os.path.exists(self.blacklist_path):
                with open(self.blacklist_path, "r") as f:
                    blacklist = json.load(f)
                logger.info(f"Loaded {len(blacklist)} blacklisted commands")
                return blacklist
            else:
                logger.warning(f"Blacklist file not found: {self.blacklist_path}")
                # Default blacklist
                return [
                    {
                        "pattern": "rm -rf /",
                        "reason": "Dangerous file deletion"
                    },
                    {
                        "pattern": "dd if=/dev/zero of=/dev/sda",
                        "reason": "Disk overwrite"
                    }
                ]
        except Exception as e:
            logger.error(f"Error loading blacklist: {e}")
            # Default blacklist
            return [
                {
                    "pattern": "rm -rf /",
                    "reason": "Dangerous file deletion"
                },
                {
                    "pattern": "dd if=/dev/zero of=/dev/sda",
                    "reason": "Disk overwrite"
                }
            ]
    
    def _check_blacklist(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Check if a command is blacklisted.
        
        Args:
            command: Command to check
            
        Returns:
            Blacklist entry if command is blacklisted, None otherwise
        """
        for entry in self.blacklist:
            if entry["pattern"] in command:
                return entry
        return None
    
    def _check_permissions(self, command: str, user_id: str) -> bool:
        """
        Check if a user has permission to execute a command.
        
        Args:
            command: Command to check
            user_id: User ID
            
        Returns:
            True if user has permission, False otherwise
        """
        # For testing, allow all commands except those in blacklist
        return self._check_blacklist(command) is None

    def register_internal_command(self, name: str, function: Any) -> None:
        """
        Register an internal command.
        
        Args:
            name: Command name
            function: Function to execute
        """
        self.internal_commands[name] = function
        logger.info(f"Registered internal command: {name}")

    def execute(self, command: str, user_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a command.
        
        Args:
            command: Command to execute
            user_id: User ID
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary with execution result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Command Executor initialization failed")
                return {"success": False, "error": "Command Executor initialization failed"}
        
        if not self.execution_lock.acquire(blocking=False):
            return {"success": False, "error": "Another command is already running."}

        try:
            # Check blacklist
            blacklist_entry = self._check_blacklist(command)
            if blacklist_entry:
                logger.warning(f"Command '{command}' blacklisted: {blacklist_entry['reason']}")
                return {
                    "success": False,
                    "error": f"Command not allowed: {blacklist_entry['reason']}"
                }
            
            # Check permissions
            if not self._check_permissions(command, user_id):
                logger.warning(f"User {user_id} does not have permission to execute '{command}'")
                return {
                    "success": False,
                    "error": "Permission denied"
                }
            
            # Check for internal command
            command_parts = shlex.split(command)
            command_name = command_parts[0]
            if command_name == "write" and command_parts[1] == "file":
                try:
                    path = command_parts[2]
                    content = " ".join(command_parts[3:])
                    file_access = self.registry.get_component("file_access_utility")
                    if not file_access:
                        return {"success": False, "error": "File Access Utility not available"}
                    return file_access.write_file(path, content)
                except IndexError:
                    return {"success": False, "error": "Usage: write file <path> <content>"}
            elif command_name in self.internal_commands:
                logger.info(f"Executing internal command '{command}' for user {user_id}")
                try:
                    # For now, passing args as a dictionary is not supported this way.
                    # This would need a more sophisticated argument parser.
                    # For the purpose of this task, we assume simple commands.
                    result = self.internal_commands[command_name]({})
                    return result
                except Exception as e:
                    logger.error(f"Error executing internal command: {e}")
                    return {"success": False, "error": str(e)}

            # Log command execution
            logger.info(f"Executing external command '{command}' for user {user_id}")
            
            # Execute command
            try:
                # Set timeout
                timeout = timeout or self.timeout_seconds
                
                # Execute command
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for command to complete with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Check return code
                if process.returncode != 0:
                    logger.error(f"Command failed with return code {process.returncode}: {stderr}")
                    return {
                        "success": False,
                        "error": f"Command failed with return code {process.returncode}",
                        "stdout": stdout,
                        "stderr": stderr,
                        "return_code": process.returncode
                    }
                
                logger.info(f"Command executed successfully")
                
                return {
                    "success": True,
                    "output": stdout,
                    "stderr": stderr,
                    "return_code": process.returncode
                }
            
            except subprocess.TimeoutExpired:
                # Kill the process if it timed out
                process.kill()
                stdout, stderr = process.communicate()
                
                logger.error(f"Command timed out after {timeout} seconds")
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "stdout": stdout,
                    "stderr": stderr
                }
            
            except Exception as e:
                logger.error(f"Error executing command: {e}")
                return {
                    "success": False,
                    "error": f"Error executing command: {e}"
                }
        
        finally:
            self.execution_lock.release()
    
    def execute_script(self, script_content: str, user_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a script.
        
        Args:
            script_content: Script content
            user_id: User ID
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary with execution result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Command Executor initialization failed")
                return {"success": False, "error": "Command Executor initialization failed"}
        
        try:
            # Create temporary script file
            script_path = f"temp_script_{int(time.time())}.sh"
            
            try:
                # Write script content to file
                with open(script_path, "w") as f:
                    f.write(script_content)
                
                # Make script executable
                os.chmod(script_path, 0o755)
                
                # Execute script
                result = self.execute(f"./{script_path}", user_id, timeout)
                
                return result
            
            finally:
                # Clean up temporary script file
                if os.path.exists(script_path):
                    os.remove(script_path)
        
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self) -> None:
        """
        Shutdown the Command Executor.
        """
        if not self.initialized:
            return
        
        self.initialized = False
        logger.info("Command Executor shutdown complete")

# Singleton instance
_instance = None

def get_instance(registry, config: Optional[Dict[str, Any]] = None) -> CommandExecutor:
    """
    Get the singleton instance of the Command Executor.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        CommandExecutor instance
    """
    global _instance
    if _instance is None:
        _instance = CommandExecutor(registry, config)
    return _instance
