#!/usr/bin/env python3
"""
Start Triangulum Helper Script

This script starts the Triangulum system and sets the process ID in the resource manager
so that the status command can correctly detect that the system is running.
"""

import os
import sys
import time
import json
import subprocess
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StartTriangulum")

def start_triangulum():
    """
    Start the Triangulum system using the daemon process.
    """
    try:
        # Import required modules
        sys.path.append('.')
        from triangulum_resource_manager import create_triangulum_resource_manager
        import yaml
        
        # Load configuration
        config_path = "system_config.yaml"
        if not os.path.exists(config_path):
            logger.error(f"Config file {config_path} not found")
            return 1
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create directories
        os.makedirs(".triangulum", exist_ok=True)
        
        # Save configuration
        config_json_path = ".triangulum/config.json"
        with open(config_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 1. First, start the daemon process
        daemon_cmd = [sys.executable, "triangulum_daemon.py"]
        
        logger.info(f"Starting Triangulum daemon with command: {' '.join(daemon_cmd)}")
        
        # Start the daemon process in a separate window so it stays running
        daemon_process = None
        if os.name == 'nt':  # Windows
            daemon_process = subprocess.Popen(
                daemon_cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # Unix/Linux
            daemon_process = subprocess.Popen(
                daemon_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
        
        # Wait a moment to ensure daemon starts
        time.sleep(2)
        
        # 2. Now start the main Triangulum process
        main_cmd = [sys.executable, "main.py", "--config", config_path]
        
        logger.info(f"Starting Triangulum main process with command: {' '.join(main_cmd)}")
        
        # Start the main process
        main_process = subprocess.Popen(
            main_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Register with the resource manager
        # Create resource manager
        resource_manager = create_triangulum_resource_manager(config)
        
        # Get the daemon's PID from the process file
        daemon_pid = None
        process_file = ".triangulum/process.json"
        if os.path.exists(process_file):
            try:
                with open(process_file, 'r') as f:
                    data = json.load(f)
                    daemon_pid = data.get('pid')
            except (json.JSONDecodeError, IOError):
                pass
        
        # If we couldn't get the daemon PID from the file, use the one we have
        if daemon_pid is None and daemon_process is not None:
            daemon_pid = daemon_process.pid
            
        # Set the process ID and start time in the resource manager
        if daemon_pid:
            resource_manager.set_process_id(daemon_pid)
            resource_manager.set_start_time(datetime.datetime.now())
            
            logger.info(f"Triangulum system started with daemon PID {daemon_pid}")
            print(f"Triangulum system started with daemon PID {daemon_pid}")
            print("Use 'triangulum:status' to check status")
        else:
            logger.error("Failed to get daemon PID")
            print("Triangulum system started but failed to get daemon PID")
            print("Use 'triangulum:status' to check status")
        
        return 0
    except Exception as e:
        logger.error(f"Error starting Triangulum: {e}")
        print(f"Error starting Triangulum: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(start_triangulum())
