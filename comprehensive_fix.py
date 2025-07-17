#!/usr/bin/env python3
"""
Comprehensive fix for config_manager tests by ensuring temperature is in the default config
"""

import re

def fix_config_manager():
    # Open the config_manager.py file and read its content
    with open('config_manager.py', 'r') as f:
        config_content = f.read()
    
    # Make sure temperature is in the default config
    pattern = r'("preferred": "openai",)([\s\n]+)("temperature": 0\.1,)'
    
    if not re.search(pattern, config_content):
        # If the temperature isn't already in the config, add it
        pattern = r'("preferred": "openai",)'
        replacement = r'\1\n                "temperature": 0.1,'
        
        config_content = re.sub(pattern, replacement, config_content)
        
        # Write the changes back to the config_manager.py file
        with open('config_manager.py', 'w') as f:
            f.write(config_content)
        
        print("Fixed config_manager.py by adding temperature to default config")
    else:
        print("Temperature already exists in default config")

if __name__ == "__main__":
    fix_config_manager()
