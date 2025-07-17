#!/usr/bin/env python3
"""
Triangulum Integration Helper Functions

This module provides helper functions for the triangulum_integration module.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Callable

# Import component classes from separate modules
from triangulum_components.system_monitor import SystemMonitor
from triangulum_components.dashboard import DashboardVisualizer
from triangulum_components.queue_manager import QueueManager
from triangulum_components.rollback_manager import RollbackManager
from triangulum_components.plan_executor import PlanExecutor
from triangulum_client import TriangulumClient

logger = logging.getLogger("TriangulumIntegration")

def create_system_monitor(config: Dict[str, Any] = None) -> SystemMonitor:
    """Create a system monitor instance."""
    return SystemMonitor(config)

def create_dashboard_visualizer(config: Dict[str, Any] = None) -> DashboardVisualizer:
    """Create a dashboard visualizer instance."""
    return DashboardVisualizer(config)

def create_queue_manager(config: Dict[str, Any] = None) -> QueueManager:
    """Create a queue manager instance."""
    return QueueManager(config)

def create_rollback_manager(config: Dict[str, Any] = None) -> RollbackManager:
    """Create a rollback manager instance."""
    return RollbackManager(config)

def create_plan_executor(config: Dict[str, Any] = None) -> PlanExecutor:
    """Create a plan executor instance."""
    return PlanExecutor(config)

def create_triangulum_client(config: Dict[str, Any] = None) -> TriangulumClient:
    """Create a Triangulum client instance."""
    return TriangulumClient(config)

def connect_to_triangulum(config: Dict[str, Any] = None) -> bool:
    """
    Connect to Triangulum.
    
    Args:
        config: Configuration options
        
    Returns:
        Whether connection was successful
    """
    try:
        client = create_triangulum_client(config)
        success = client.connect()
        
        if success:
            logger.info("Connected to Triangulum")
        else:
            logger.error("Failed to connect to Triangulum")
        
        return success
    except Exception as e:
        logger.error(f"Error connecting to Triangulum: {e}")
        return False

def execute_plan(plan_id: str, config: Dict[str, Any] = None, async_execution: bool = False) -> bool:
    """
    Execute a plan.
    
    Args:
        plan_id: ID of the plan
        config: Configuration options
        async_execution: Whether to execute asynchronously
        
    Returns:
        Whether the plan execution was started
    """
    try:
        executor = create_plan_executor(config)
        return executor.execute_plan(plan_id, async_execution)
    except Exception as e:
        logger.error(f"Error executing plan: {e}")
        return False

def start_system_monitoring(callback: Callable = None, config: Dict[str, Any] = None) -> bool:
    """
    Start system monitoring.
    
    Args:
        callback: Callback function to call with metrics
        config: Configuration options
        
    Returns:
        Whether monitoring was started
    """
    try:
        monitor = create_system_monitor(config)
        return monitor.start_monitoring(callback)
    except Exception as e:
        logger.error(f"Error starting system monitoring: {e}")
        return False

def start_dashboard(data_provider: Callable = None, config: Dict[str, Any] = None) -> Optional[str]:
    """
    Start dashboard visualization.
    
    Args:
        data_provider: Function to call to get data to visualize
        config: Configuration options
        
    Returns:
        Dashboard URL, or None if failed
    """
    try:
        visualizer = create_dashboard_visualizer(config)
        success = visualizer.start(data_provider)
        
        if success:
            dashboard_url = visualizer.get_dashboard_url()
            logger.info(f"Dashboard started at {dashboard_url}")
            return dashboard_url
        else:
            logger.error("Failed to start dashboard")
            return None
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return None
