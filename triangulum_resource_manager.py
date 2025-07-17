#!/usr/bin/env python3
"""
Triangulum Resource Manager Extension

This module extends the base ResourceManager class with additional
functionality needed for Triangulum integration.
"""

import os
import sys
import json
import logging
import time
import datetime
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from resource_manager import ResourceManager as BaseResourceManager

logger = logging.getLogger("TriangulumResourceManager")

class ResourceManager(BaseResourceManager):
    """
    Extended ResourceManager for Triangulum Integration
    
    This class adds Triangulum-specific functionality to the base ResourceManager
    """
    
    def __init__(self, config: Dict[str, Any] = None, total_agents: int = None):
        """
        Initialize Triangulum Resource Manager
        
        Args:
            config: Configuration options
            total_agents: Total number of agents (passed from system_config.yaml)
        """
        # If total_agents is provided, add it to the config
        if total_agents is not None:
            config = config or {}
            config["total_agents"] = total_agents
            
        super().__init__(config or {})
        self.process_id = None
        self.start_time = None
        self.last_status_update = None
        self.total_agents = total_agents
        
    def is_running(self) -> bool:
        """
        Check if Triangulum system is running
        
        Returns:
            True if running, False otherwise
        """
        # First check for the status file (most reliable)
        try:
            import json
            status_file = ".triangulum/status.json"
            if os.path.exists(status_file):
                # Check if the file was modified recently (within the last 30 seconds)
                if time.time() - os.path.getmtime(status_file) < 30:
                    # Read the status file
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                        if status.get('status') == 'running':
                            # Update our instance with the PID from the status file
                            pid = status.get('pid')
                            if pid:
                                self.process_id = pid
                                # Also update start time if we don't have it
                                if not self.start_time and 'timestamp' in status:
                                    try:
                                        # Calculate an approximate start time
                                        uptime_seconds = status.get('uptime_seconds', 0)
                                        timestamp = datetime.datetime.fromisoformat(status['timestamp'])
                                        start_time = timestamp - datetime.timedelta(seconds=uptime_seconds)
                                        self.start_time = start_time
                                    except (ValueError, TypeError):
                                        pass
                                return True
        except (json.JSONDecodeError, IOError, KeyError, ValueError, TypeError):
            pass
        
        # Next, check if we have a process ID in our instance
        if self.process_id is not None:
            try:
                import psutil
                process = psutil.Process(self.process_id)
                if process.is_running():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ImportError):
                self.process_id = None
        
        # If no valid process ID in instance, check the process file
        try:
            import json
            process_file = ".triangulum/process.json"
            if os.path.exists(process_file):
                with open(process_file, 'r') as f:
                    data = json.load(f)
                    pid = data.get('pid')
                    if pid:
                        try:
                            import psutil
                            process = psutil.Process(pid)
                            if process.is_running():
                                # Update our instance with the found process ID
                                self.process_id = pid
                                if not self.start_time and 'start_time' in data:
                                    try:
                                        self.start_time = datetime.datetime.fromisoformat(data['start_time'])
                                    except (ValueError, TypeError):
                                        pass
                                return True
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ImportError):
                            # Process doesn't exist, clean up the file
                            try:
                                os.remove(process_file)
                            except (OSError, PermissionError):
                                pass
        except (json.JSONDecodeError, IOError, KeyError):
            pass
        
        # Process no longer exists
        self.process_id = None
        return False
    
    def set_process_id(self, process_id: int) -> None:
        """
        Set the Triangulum process ID
        
        Args:
            process_id: Process ID
        """
        self.process_id = process_id
        logger.info(f"Set Triangulum process ID: {process_id}")
    
    def clear_process_id(self) -> None:
        """
        Clear the Triangulum process ID
        """
        self.process_id = None
        logger.info("Cleared Triangulum process ID")
    
    def set_start_time(self, start_time) -> None:
        """
        Set the Triangulum start time
        
        Args:
            start_time: Start time
        """
        self.start_time = start_time
        logger.info(f"Set Triangulum start time: {start_time}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get Triangulum system status
        
        Returns:
            Status information
        """
        status = super().get_status()
        
        # Add Triangulum-specific status
        status.update({
            "process_id": self.process_id,
            "start_time": self.start_time,
            "uptime": time.time() - self.start_time.timestamp() if self.start_time else 0,
            "last_status_update": time.time()
        })
        
        self.last_status_update = time.time()
        
        return status
    
    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute a plan
        
        Args:
            plan_id: Plan ID
            
        Returns:
            Result information
        """
        logger.info(f"Executing plan: {plan_id}")
        
        # Return success response
        return {
            "success": True,
            "execution_id": f"exec-{plan_id}-{int(time.time())}"
        }
    
    def cancel_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Cancel a plan
        
        Args:
            plan_id: Plan ID
            
        Returns:
            Result information
        """
        logger.info(f"Cancelling plan: {plan_id}")
        
        # Return success response
        return {
            "success": True
        }
    
    def list_agents(self, agent_type: str = None) -> List[Dict[str, Any]]:
        """
        List agents
        
        Args:
            agent_type: Filter by agent type
            
        Returns:
            List of agents
        """
        # Mock implementation
        return []
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent status
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent status
        """
        # Mock implementation
        return None
    
    def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Stop an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Result information
        """
        # Mock implementation
        return {
            "success": False,
            "error": "Agent not found"
        }
    
    def record_allocation(self, resource_type: str, resource_id: str, allocation: Dict[str, Any]) -> None:
        """
        Record resource allocation
        
        Args:
            resource_type: Resource type
            resource_id: Resource ID
            allocation: Allocation details
        """
        logger.info(f"Recording {resource_type} allocation for {resource_id}: {allocation}")
        # Implementation would record the allocation in a database or other store

# Create and initialize the resource manager
def create_triangulum_resource_manager(config: Dict[str, Any] = None) -> ResourceManager:
    """
    Create and initialize a Triangulum resource manager
    
    Args:
        config: Configuration options
        
    Returns:
        Initialized Triangulum resource manager
    """
    # Extract total_agents from config if present
    total_agents = None
    if config and "agents" in config and "total" in config["agents"]:
        total_agents = config["agents"]["total"]
    
    manager = ResourceManager(config, total_agents=total_agents)
    return manager
