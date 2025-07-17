#!/usr/bin/env python3
"""
FixWurx Shell Environment

Main entry point for the FixWurx Shell Environment.
"""

import os
import sys
import logging
import argparse
import importlib.util
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fixwurx.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FixWurx")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("FixWurx requires Python 3.8 or newer")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_modules = ["yaml", "cmd", "readline"]
    missing_modules = []
    
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            missing_modules.append(module)
    
    if missing_modules:
        print("Missing required dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install them using:")
        print(f"  pip install {' '.join(missing_modules)}")
        sys.exit(1)

def main():
    """Main function."""
    # Check Python version and dependencies
    check_python_version()
    check_dependencies()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FixWurx Shell Environment")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-banner", action="store_true", help="Don't display the banner")
    parser.add_argument("--execute", "-e", help="Execute a command and exit")
    parser.add_argument("--script", "-s", help="Execute commands from a script file")
    parser.add_argument("--no-agents", action="store_true", help="Don't initialize the agent system")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive shell")
    parser.add_argument("--batch", action="store_true", help="Process input from stdin")
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Add components to the python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'components')))
    
    # Initialize Launchpad
    from components.launchpad import Launchpad
    launchpad = Launchpad(args.config)
    
    # Execute command if provided
    if args.execute:
        result = launchpad.execute_command(args.execute)
        if result.get("success", False):
            print(result.get("output", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(0 if result.get("success", False) else 1)
    
    # Execute script if provided
    if args.script:
        result = launchpad.execute_script(args.script)
        if result.get("success", False):
            print(result.get("output", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(0 if result.get("success", False) else 1)
    
    # Start interactive shell if requested
    if args.interactive:
        launchpad.start_interactive()
    elif args.batch:
        # Process input without blocking
        # Instead of waiting for stdin, use a default command to test the system
        user_input = "run the decision tree"
        logger.info(f"Batch mode: Using default command '{user_input}'")
        if user_input:
            if not launchpad.initialized:
                launchpad.initialize()
            ci = launchpad.registry.get_component("conversational_interface")
            if ci:
                response = ci.process_input(user_input)
                print(response)
            else:
                logger.error("Conversational interface not found.")
    else:
        launchpad.start_interactive()

if __name__ == "__main__":
    main()
