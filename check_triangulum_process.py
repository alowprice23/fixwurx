#!/usr/bin/env python3
"""
Debug utility to check if the Triangulum process file exists and if the process is running
"""

import os
import sys
import json
import psutil

def check_process_file():
    """Check if the process file exists and if the process is running."""
    process_file = ".triangulum/process.json"
    
    # Check if directory exists
    if not os.path.exists(".triangulum"):
        print(f"Directory '.triangulum' does not exist!")
        return False
    
    # Check if file exists
    if not os.path.exists(process_file):
        print(f"Process file '{process_file}' does not exist!")
        return False
    
    # Read the file
    try:
        with open(process_file, 'r') as f:
            data = json.load(f)
            print(f"Process file contents: {data}")
            
            # Check if pid exists
            pid = data.get('pid')
            if not pid:
                print(f"No PID found in process file!")
                return False
            
            # Check if process is running
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    print(f"Process {pid} is running with name: {process.name()}")
                    return True
                else:
                    print(f"Process {pid} is not running!")
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                print(f"Error checking process {pid}!")
                return False
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading process file: {e}")
        return False

if __name__ == "__main__":
    if check_process_file():
        print("Triangulum is running!")
    else:
        print("Triangulum is not running!")
