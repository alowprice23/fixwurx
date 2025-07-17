#!/usr/bin/env python3
"""
Patch to fix the TriangulationEngine to store the resource_manager
"""

import re

def patch_triangulation_engine():
    # Path to triangulation_engine.py
    file_path = "triangulation_engine.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the TriangulationEngine __init__ method
    init_pattern = r"def __init__\(self, config: Dict\[str, Any\] = None\):"
    init_replacement = "def __init__(self, config: Dict[str, Any] = None, resource_manager=None):"
    
    # Replace the __init__ method signature
    updated_content = re.sub(init_pattern, init_replacement, content)
    
    # Find where to add the resource_manager attribute assignment
    # Look for the line after "self.config = config or {}"
    config_pattern = r"self\.config = config or \{\}"
    config_replacement = "self.config = config or {}\n        self.resource_manager = resource_manager"
    
    # Add the resource_manager attribute
    updated_content = re.sub(config_pattern, config_replacement, updated_content)
    
    # Write back the updated content
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully patched {file_path} to add resource_manager attribute")

if __name__ == "__main__":
    patch_triangulation_engine()
