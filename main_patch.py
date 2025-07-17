#!/usr/bin/env python3
"""
Patch to fix the ResourceManager import in main.py
"""

import os
import sys
import re

def patch_main_py():
    # Path to main.py
    main_path = "main.py"
    
    # Read the current content
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Replace the import
    updated_content = re.sub(
        r"from resource_manager import ResourceManager",
        "from triangulum_resource_manager import ResourceManager",
        content
    )
    
    # Since we've patched the TriangulationEngine class to support resource_manager,
    # we don't need to remove the parameter anymore. Let's make sure we restore it.
    updated_content = re.sub(
        r"engine = TriangulationEngine\(config=cfg\)",
        "engine = TriangulationEngine(resource_manager=res_mgr, config=cfg)",
        updated_content
    )
    
    # Write back the updated content
    with open(main_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully patched {main_path} to use the extended ResourceManager and fix TriangulationEngine initialization")

if __name__ == "__main__":
    patch_main_py()
