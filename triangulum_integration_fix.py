#!/usr/bin/env python3
"""
Triangulum Integration Fix

This module provides a fallback for the triangulum_integration module.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("triangulum_integration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TriangulumIntegration")

# Mock mode for testing
MOCK_MODE = os.environ.get("TRIANGULUM_TEST_MODE", "0") == "1"

# Import all component classes from separate modules
from triangulum_components.system_monitor import SystemMonitor
from triangulum_components.dashboard import DashboardVisualizer
from triangulum_components.queue_manager import QueueManager
from triangulum_components.rollback_manager import RollbackManager
from triangulum_components.plan_executor import PlanExecutor
from triangulum_client import TriangulumClient

# Factory functions for creating components
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

# Export all relevant symbols
__all__ = [
    # Classes
    'SystemMonitor',
    'DashboardVisualizer', 
    'QueueManager',
    'RollbackManager',
    'PlanExecutor',
    'TriangulumClient',
    
    # Factory functions
    'create_system_monitor',
    'create_dashboard_visualizer',
    'create_queue_manager',
    'create_rollback_manager',
    'create_plan_executor',
    'create_triangulum_client',
    
    # Constants
    'MOCK_MODE',
    'logger'
]
