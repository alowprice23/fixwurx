#!/usr/bin/env python3
"""
Fix Auditor Commands Script

This script appends the missing functions from auditor_missing_functions.py 
to auditor_commands.py to fix the 'name errors_command is not defined' error.
"""

import os
import sys

def main():
    # Check if both files exist
    if not os.path.exists('auditor_commands.py'):
        print("Error: auditor_commands.py not found")
        return 1
    
    if not os.path.exists('auditor_missing_functions.py'):
        print("Error: auditor_missing_functions.py not found")
        return 1
    
    # Read the missing functions
    with open('auditor_missing_functions.py', 'r') as f:
        missing_functions = f.read()
    
    # Strip the comment at the top
    missing_functions = missing_functions.replace('# Missing functions for auditor_commands.py\n\n', '')
    
    # Read the auditor_commands.py file
    with open('auditor_commands.py', 'r') as f:
        auditor_commands = f.read()
    
    # Check if auditor_commands.py already has errors_command
    if 'def errors_command(' in auditor_commands:
        print("Note: errors_command already exists in auditor_commands.py")
        return 0
    
    # Append the missing functions to auditor_commands.py
    with open('auditor_commands.py', 'a') as f:
        f.write('\n')
        f.write(missing_functions)
    
    print("Successfully added missing functions to auditor_commands.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
