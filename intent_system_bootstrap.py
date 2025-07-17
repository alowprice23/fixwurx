#!/usr/bin/env python3
"""
Intent Classification System Bootstrap

This script initializes and bootstraps the intent classification system,
integrating it with the FixWurx ecosystem.
"""

import os
import sys
import json
import logging
import argparse
import importlib
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intent_bootstrap.log"), logging.StreamHandler()]
)
logger = logging.getLogger("IntentBootstrap")

def create_config():
    """Create the necessary configuration if it doesn't exist."""
    os.makedirs("config", exist_ok=True)
    
    config_path = "config/intent_system_config.json"
    if not os.path.exists(config_path):
        default_config = {
            "cache_capacity": 100,
            "history_size": 50,
            "window_size": 10,
            "shell_integration": {"enabled": True, "command_prefix": "!intent"},
            "agent_integration": {"enabled": True, "fallback_enabled": True},
            "neural_integration": {"enabled": True, "confidence_threshold": 0.7},
            "triangulum_integration": {"enabled": True, "distribution_threshold": 0.8}
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            logger.info(f"Created default configuration at {config_path}")
    return config_path

class SimpleRegistry:
    """Simple component registry for testing."""
    def __init__(self):
        self.components = {}
    
    def register_component(self, name, component):
        self.components[name] = component
        logger.info(f"Registered component: {name}")
    
    def get_component(self, name):
        return self.components.get(name)

def run_demo():
    """Run a simple demo of the intent classification system."""
    from intent_classification_system import IntentClassificationSystem, initialize_system
    from components.intent_caching_system import IntentOptimizationSystem
    
    # Create registry and initialize components
    registry = SimpleRegistry()
    
    # Create a simple file access utility for testing
    class SimpleFileAccess:
        def read_file(self, path):
            if os.path.exists(path):
                with open(path, 'r') as f:
                    content = f.read()
                return {"success": True, "content": content, "size": len(content)}
            return {"success": False, "error": f"File not found: {path}"}
    
    # Register components
    registry.register_component("file_access_utility", SimpleFileAccess())
    
    # Initialize intent system
    intent_system = initialize_system(registry)
    
    # Initialize intent optimization system
    optimization_system = IntentOptimizationSystem(
        cache_capacity=100,
        history_size=50,
        window_size=10
    )
    registry.register_component("intent_optimization_system", optimization_system)
    
    # Create test file
    with open("demo_file.txt", 'w') as f:
        f.write("This is a test file for the demo.\nIt has multiple lines of content.\nYou can modify this file through the conversational interface.")
    
    # Test query
    query = "read demo_file.txt"
    context = {"state": {"current_folder": os.getcwd()}}
    
    # Process query twice to test caching
    print("First Query Response:", intent_system.classify_intent(query, context).to_dict())
    
    # Get cached result
    cached_intent = optimization_system.get_cached_intent(query)
    
    # Process query again
    print("Second Query Response:", SimpleFileAccess().read_file("demo_file.txt"))
    print(f"Classification called {intent_system.classification_count if hasattr(intent_system, 'classification_count') else 0} times (should be 1 if caching works)")
    
    print("\nAll tests completed.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Initialize and bootstrap the intent classification system")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--demo", action="store_true", help="Run a simple demo")
    parser.add_argument("--integrate", action="store_true", help="Integrate with existing FixWurx components")
    args = parser.parse_args()
    
    # Create configuration
    config_path = args.config or create_config()
    
    if args.demo:
        run_demo()
        return
    
    if args.integrate:
        # Import integration module
        try:
            from fixwurx_intent_integration import integrate_with_fixwurx
            from shell_environment import get_environment
            from launchpad import get_launchpad_instance
            
            # Get shell environment and launchpad
            shell_env = get_environment()
            launchpad = get_launchpad_instance()
            
            # Integrate intent system with FixWurx
            integration = integrate_with_fixwurx(shell_env, launchpad)
            
            logger.info("Intent classification system integrated with FixWurx")
            logger.info(f"Integration stats: {integration.get_stats()}")
        except ImportError as e:
            logger.error(f"Integration failed: {e}")
            logger.error("Make sure fixwurx_intent_integration.py and related components are available")
    else:
        # Just initialize the system
        try:
            from intent_classification_system import initialize_system
            
            # Create a simple registry
            registry = SimpleRegistry()
            
            # Initialize the system
            intent_system = initialize_system(registry)
            
            logger.info("Intent classification system initialized")
        except ImportError as e:
            logger.error(f"Initialization failed: {e}")
            logger.error("Make sure intent_classification_system.py is available")

if __name__ == "__main__":
    main()
