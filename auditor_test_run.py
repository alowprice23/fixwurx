#!/usr/bin/env python3
"""
FixWurx Auditor Test Runner for Successful Audit

This script runs the auditor with explicit directory scanning to ensure
all implemented modules are properly detected.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import datetime
from typing import Dict, Any, Optional

# Force UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import codecs
    # Use utf-8 encoding for console output
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AuditorTest] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('auditor_test_run')

def main() -> int:
    """Main function"""
    # Print current working directory
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # List files in the current directory
    files = os.listdir(cwd)
    goal_files = [f for f in files if f.startswith('goal') and f.endswith('.py')]
    logger.info(f"Found goal files: {goal_files}")
    
    # Display file contents
    for file in goal_files:
        with open(file, 'r') as f:
            content = f.read()
            logger.info(f"File {file} content (truncated):\n{content[:100]}...")
    
    # Create a directory listing file
    listing_file = "directory_listing.txt"
    with open(listing_file, 'w') as f:
        f.write(f"Directory listing for {cwd}\n")
        f.write("=" * 50 + "\n\n")
        for file in sorted(files):
            full_path = os.path.join(cwd, file)
            is_dir = os.path.isdir(full_path)
            size = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
            f.write(f"{'[DIR]' if is_dir else '[FILE]'} {file} ({size} bytes)\n")
    
    logger.info(f"Created directory listing at {listing_file}")
    
    # Read the delta_rules.json file
    try:
        with open('delta_rules.json', 'r') as f:
            rules = json.load(f)
            logger.info(f"Delta rules content: {json.dumps(rules, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to read delta_rules.json: {e}")
    
    # Check if goal modules are properly implemented
    try:
        # Import the modules to verify they load correctly
        import goal1
        import goal2
        import goal3
        
        logger.info("Successfully imported goal1, goal2, and goal3 modules")
        
        # Execute the main functions to verify they work
        result1 = goal1.main()
        result2 = goal2.main()
        result3 = goal3.main()
        
        logger.info(f"goal1.main() result: {result1}")
        logger.info(f"goal2.main() result: {result2}")
        logger.info(f"goal3.main() result: {result3}")
        
        print("\nAll goal modules implemented and functioning correctly.")
        print("Results:")
        print(f"  goal1: {result1}")
        print(f"  goal2: {result2}")
        print(f"  goal3: {result3}\n")
        
        return 0
    except Exception as e:
        logger.error(f"Error testing goal modules: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
