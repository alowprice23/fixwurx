#!/usr/bin/env python3
"""
Script to fix the syntax error in config_manager.py
"""

import re

def fix_security_section():
    # Read the file
    with open('config_manager.py', 'r') as f:
        content = f.read()
    
    # Use regex to find and fix the security section with missing comma
    pattern = r'("sensitive_data_encryption": True)\s+("throttling_enabled": True)'
    replacement = r'\1,\n                \2'
    
    fixed_content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back
    with open('config_manager.py', 'w') as f:
        f.write(fixed_content)
    
    print("Fixed security section in config_manager.py")

if __name__ == "__main__":
    fix_security_section()
