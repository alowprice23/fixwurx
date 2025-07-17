#!/usr/bin/env python3
"""
Triangulum Integration Module

This module provides integration with the Triangulum system by importing
and re-exporting components from specialized modules.
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


# Export all relevant symbols
__all__ = [
    # Classes
    'SystemMonitor',
    'DashboardVisualizer', 
    'QueueManager',
    'RollbackManager',
    'PlanExecutor',
    'TriangulumClient',
    
    # Constants
    'MOCK_MODE',
    'logger'
]

# Handle imports when triangulum_components are not available
if __name__ != "__main__":
    try:
        # Try importing from component modules
        from triangulum_components.system_monitor import SystemMonitor
        from triangulum_components.dashboard import DashboardVisualizer
        from triangulum_components.queue_manager import QueueManager
        from triangulum_components.rollback_manager import RollbackManager
        from triangulum_components.plan_executor import PlanExecutor
    except ImportError:
        # Fallback to the _fix version if components aren't available
        try:
            from triangulum_integration_fix import (
                SystemMonitor,
                DashboardVisualizer,
                QueueManager,
                RollbackManager,
                PlanExecutor,
                create_system_monitor,
                create_dashboard_visualizer,
                create_queue_manager,
                create_rollback_manager,
                create_plan_executor
            )
            logger.info("Using triangulum_integration_fix implementation")
        except ImportError:
            logger.error("Could not import triangulum components")
