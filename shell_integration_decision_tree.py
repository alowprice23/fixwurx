#!/usr/bin/env python3
"""
Shell Integration for Decision Tree

This module integrates the decision tree logic with the shell environment.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ShellIntegrationDecisionTree")

def integrate_with_shell():
    """
    Integrate decision tree logic with the shell environment.
    """
    logger.info("Integrating decision tree logic with shell environment")
    
    try:
        # Import decision tree commands
        from decision_tree_commands import register_commands
        
        # Get registry
        registry = sys.modules.get("__main__").registry
        
        # Register commands
        register_commands(registry)
        
        logger.info("Decision tree commands registered with shell")
        
        # Create necessary directories
        os.makedirs(".triangulum/results", exist_ok=True)
        os.makedirs(".triangulum/patches", exist_ok=True)
        os.makedirs(".triangulum/verification_results", exist_ok=True)
        os.makedirs(".triangulum/logs", exist_ok=True)
        
        logger.info("Decision tree directories created")
        
        return True
    except Exception as e:
        logger.error(f"Error integrating decision tree with shell: {e}")
        return False

if __name__ == "__main__":
    # When run as a script, attempt to integrate with the shell
    if integrate_with_shell():
        print("Decision tree logic integrated with shell successfully")
    else:
        print("Error integrating decision tree logic with shell")
        sys.exit(1)
