"""
tooling/clean_comments.py
─────────────────────────
Utility to remove static analysis "Fixed:" comments from Python files.

This script removes comments matching patterns like:
- # Fixed: weak_crypto - Use of insecure random number generator
- # Fixed: null_reference - Potential null/None reference detected by AST analysis
- # Fixed: null_reference - Exception caught but not handled properly

It can be run on specific files or the entire codebase.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Set


def clean_fixed_comments(file_path: str | Path) -> bool:
    """
    Remove all "# Fixed:" comments from the specified Python file.
    
    Args:
        file_path: Path to the Python file to process
        
    Returns:
        True if changes were made, False otherwise
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.suffix != '.py':
        print(f"Skipping {path} (not a Python file or doesn't exist)")
        return False
        
    # Read file content
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Pattern to match Fixed comments
    fixed_pattern = re.compile(r'^\s*# Fixed:.*$')
    
    # Filter out Fixed comments
    new_lines = []
    changes_made = False
    for line in lines:
        if fixed_pattern.match(line):
            changes_made = True
            continue
        new_lines.append(line)
    
    # Write back if changes were made
    if changes_made:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Cleaned {path}")
        
    return changes_made


def find_python_files(directory: str | Path) -> List[Path]:
    """
    Find all Python files in the given directory and its subdirectories.
    
    Args:
        directory: Root directory to search in
        
    Returns:
        List of paths to Python files
    """
    dir_path = Path(directory)
    py_files = []
    
    for root, dirs, files in os.walk(dir_path):
        # Skip hidden directories and common build/dependency directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                   d not in ['__pycache__', 'venv', '.venv', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                py_files.append(Path(root) / file)
                
    return py_files


def main():
    """Main entry point for the script."""
    if len(sys.argv) > 1:
        # Process specific files
        files = [Path(arg) for arg in sys.argv[1:]]
    else:
        # Process all Python files in the current directory
        files = find_python_files('.')
    
    total_files = len(files)
    cleaned_files = 0
    
    for file_path in files:
        if clean_fixed_comments(file_path):
            cleaned_files += 1
    
    print(f"Cleaned {cleaned_files} out of {total_files} Python files")


if __name__ == "__main__":
    main()
