#!/usr/bin/env python3
"""
Creates a symlink from triangulum_integration.py to triangulum_integration_fix.py
"""

import os
import sys
import shutil

def create_symlink():
    """Create a symlink from triangulum_integration.py to triangulum_integration_fix.py"""
    print("Creating symlink from triangulum_integration.py to triangulum_integration_fix.py")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to original and fixed files
    original_file = os.path.join(current_dir, "triangulum_integration.py")
    fixed_file = os.path.join(current_dir, "triangulum_integration_fix.py")
    
    # Create backup of original file if it exists
    if os.path.exists(original_file):
        backup_file = os.path.join(current_dir, "triangulum_integration.py.bak")
        print(f"Backing up original file to {backup_file}")
        shutil.copy2(original_file, backup_file)
    
    # Copy fixed file to original file location
    print(f"Copying {fixed_file} to {original_file}")
    shutil.copy2(fixed_file, original_file)
    print("Symlink created successfully")

if __name__ == "__main__":
    create_symlink()
