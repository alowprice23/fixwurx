#!/usr/bin/env python3
"""
Triangulum Daemon

This is a simple daemon process that keeps running to indicate that Triangulum is active.
The triangulum:status command checks for this process to determine if Triangulum is running.
"""

import os
import sys
import time
import json
import signal
import logging
import datetime
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(".triangulum/daemon.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TriangulumDaemon")

# Global flag to indicate if the daemon should keep running
running = True

def signal_handler(sig, frame):
    """Handle signals to gracefully shut down the daemon."""
    global running
    logger.info(f"Received signal {sig}, shutting down...")
    running = False

def update_status():
    """Update the status file periodically."""
    status_file = ".triangulum/status.json"
    count = 0
    
    while running:
        try:
            # Create status data
            status = {
                "timestamp": datetime.datetime.now().isoformat(),
                "uptime_seconds": count * 5,
                "status": "running",
                "pid": os.getpid(),
                "heartbeat_count": count
            }
            
            # Write status to file
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            logger.debug(f"Updated status file: {status}")
            count += 1
        except Exception as e:
            logger.error(f"Error updating status: {e}")
        
        # Sleep for 5 seconds
        for _ in range(5):
            if not running:
                break
            time.sleep(1)

def create_process_file():
    """Create the process file."""
    process_file = ".triangulum/process.json"
    
    try:
        # Create data
        data = {
            "pid": os.getpid(),
            "start_time": datetime.datetime.now().isoformat(),
            "daemon_version": "1.0"
        }
        
        # Write to file
        with open(process_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created process file: {data}")
    except Exception as e:
        logger.error(f"Error creating process file: {e}")
        sys.exit(1)

def main():
    """Main daemon function."""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create directories
        os.makedirs(".triangulum", exist_ok=True)
        
        # Log startup
        logger.info(f"Triangulum daemon starting with PID {os.getpid()}")
        
        # Create process file
        create_process_file()
        
        # Start status update thread
        update_thread = threading.Thread(target=update_status)
        update_thread.daemon = True
        update_thread.start()
        
        print(f"Triangulum daemon running with PID {os.getpid()}")
        print("Press Ctrl+C to stop")
        
        # Main loop
        while running:
            time.sleep(1)
        
        logger.info("Triangulum daemon shutting down")
    except Exception as e:
        logger.error(f"Error in daemon: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
