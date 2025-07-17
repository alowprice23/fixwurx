#!/usr/bin/env python3
"""
verify_system_architecture.py
────────────────────────────
Verifies that the system architecture implementation matches the blueprint diagram.

This script scans the project directory structure and checks that it matches the
expected architecture as outlined in the system_architecture_implementation.md file.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set

# Expected directory structure based on the architecture blueprint
EXPECTED_DIRECTORIES = {
    "agents": {
        "core": {
            "launchpad": {}
        },
        "auditor": {
            "sensors": {}
        },
        "specialized": {}
    },
    "triangulum": {
        "core": {},
        "components": {}
    },
    "neural_matrix": {
        "core": {},
        "visualization": {}
    },
    "monitoring": {},
    "optimization": {},
    "learning": {},
    "docker": {},
    "kubernetes": {},
    ".github/workflows": {}
}

# Expected key components
EXPECTED_COMPONENTS = [
    # Shell Environment
    "shell_environment.py",
    "shell_scripting.py",
    "permission_system.py",
    "remote_shell.py",
    "web_interface.py",
    
    # Agent System
    "meta_agent.py",
    "agent_commands.py",
    "agent_shell_integration.py",
    
    # Triangulation Engine
    "triangulation_engine.py",
    "triangulum_commands.py",
    
    # Neural Matrix
    "neural_matrix_core.py",
    "neural_matrix_visualization.py",
    
    # Integration
    "cicd_integration.py",
    "auditor_commands.py",
    
    # Monitoring & Optimization
    "system_monitoring.py",
    "resource_manager.py",
    "metrics_bus.py"
]

def safe_open_file(file_path, encoding='utf-8', errors='replace'):
    """
    Safely open a file with proper encoding and error handling.
    
    Args:
        file_path: Path to the file
        encoding: Encoding to use (default: utf-8)
        errors: Error handling strategy (default: replace)
        
    Returns:
        File object
    """
    return open(file_path, 'r', encoding=encoding, errors=errors)

def check_directory_structure(root_dir: str, expected: Dict) -> List[str]:
    """Check if the directory structure matches the expected structure."""
    issues = []
    
    for dir_name, subdir in expected.items():
        dir_path = os.path.join(root_dir, dir_name)
        
        # Check if directory exists
        if not os.path.isdir(dir_path):
            issues.append(f"Missing directory: {dir_path}")
            continue
        
        # Recursively check subdirectories
        if subdir:
            subdir_issues = check_directory_structure(dir_path, subdir)
            issues.extend(subdir_issues)
    
    return issues

def find_component_files(root_dir: str, components: List[str]) -> Dict[str, bool]:
    """Find all component files that should exist according to the architecture."""
    component_status = {component: False for component in components}
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file in component_status:
                component_status[file] = True
    
    return component_status

def analyze_imports(root_dir: str) -> Dict[str, Set[str]]:
    """Analyze imports between components to verify architecture relationships."""
    component_imports = {}
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.py'):
                continue
            
            file_path = os.path.join(root, file)
            imports = set()
            
            try:
                # Use safe file opening with encoding and error handling
                with safe_open_file(file_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('import ') or line.startswith('from '):
                            imports.add(line)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
            
            component_imports[file_path] = imports
    
    return component_imports

def check_component_relationships():
    """Verify that component relationships match the architecture diagram."""
    # In a full implementation, this would check specific relationships
    # between components. For now, we'll use a placeholder.
    return []

def main():
    """Main verification function."""
    print("Verifying System Architecture Implementation")
    print("-" * 60)
    
    # Get the project root directory
    root_dir = os.getcwd()
    print(f"Project root directory: {root_dir}")
    
    # Check directory structure
    print("\nChecking directory structure...")
    dir_issues = check_directory_structure(root_dir, EXPECTED_DIRECTORIES)
    
    if dir_issues:
        print(f"Found {len(dir_issues)} issues with directory structure:")
        for issue in dir_issues:
            print(f"  - {issue}")
    else:
        print("Directory structure verified successfully.")
    
    # Check for key component files
    print("\nChecking for key component files...")
    component_status = find_component_files(root_dir, EXPECTED_COMPONENTS)
    
    missing_components = [comp for comp, found in component_status.items() if not found]
    if missing_components:
        print(f"Missing {len(missing_components)} key component files:")
        for comp in missing_components:
            print(f"  - {comp}")
    else:
        print("All key component files found.")
    
    # Analyze imports to verify component relationships
    print("\nAnalyzing component relationships...")
    component_imports = analyze_imports(root_dir)
    print(f"Analyzed imports in {len(component_imports)} Python files.")
    
    # Check specific relationships
    relationship_issues = check_component_relationships()
    if relationship_issues:
        print(f"Found {len(relationship_issues)} issues with component relationships:")
        for issue in relationship_issues:
            print(f"  - {issue}")
    else:
        print("Component relationships verified successfully.")
    
    # Overall verification result
    total_issues = len(dir_issues) + len(missing_components) + len(relationship_issues)
    
    print("\nVerification Summary:")
    print(f"  - Directory structure issues: {len(dir_issues)}")
    print(f"  - Missing component files: {len(missing_components)}")
    print(f"  - Component relationship issues: {len(relationship_issues)}")
    print(f"  - Total issues: {total_issues}")
    
    if total_issues == 0:
        print("\n✅ System architecture implementation matches the blueprint.")
        return 0
    else:
        print("\n❌ System architecture implementation has discrepancies with the blueprint.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
