#!/usr/bin/env python3
"""
Triangulum Integration Helper Functions

This module provides helper functions for the triangulum_integration module.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Callable

# Import from triangulum_integration
from triangulum_integration import (
    SystemMonitor, DashboardVisualizer, QueueManager, 
    RollbackManager, PlanExecutor, logger,
    create_triangulum_client, create_plan_executor, create_system_monitor,
    create_dashboard_visualizer
)

# Import from triangulum_client
from triangulum_client import TriangulumClient

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

# Import these functions back into the main module
if __name__ != "__main__":
    # When imported, patch these functions into the original module
    import triangulum_integration
    triangulum_integration.connect_to_triangulum = connect_to_triangulum
    triangulum_integration.execute_plan = execute_plan
    triangulum_integration.start_system_monitoring = start_system_monitoring
    triangulum_integration.start_dashboard = start_dashboard
