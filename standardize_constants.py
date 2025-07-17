"""
tooling/standardize_constants.py
────────────────────────────────
Utility to standardize constant naming conventions across the codebase.

This script:
1. Identifies constants (variables that should never change)
2. Ensures they follow the UPPER_SNAKE_CASE naming convention
3. Makes the constants consistent across the codebase
4. Adds the Final type annotation where appropriate

Constants are identified by:
- Variables with leading underscore and UPPER_CASE
- Variables with ALL_UPPER_CASE names
- Values with the Final type annotation
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class ConstantVisitor(ast.NodeVisitor):
    """AST visitor to find constants and their usage in Python files."""
    
    def __init__(self):
        self.constants = {}  # name -> value
        self.constant_nodes = []  # track constant assignment nodes
        self.usages = {}  # name -> list of usage nodes
        self.imports = set()  # track imported names
        self.scope_stack = ["global"]  # track current scope
        
    def visit_Import(self, node):
        """Track imported names."""
        for name in node.names:
            self.imports.add(name.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track imported names."""
        for name in node.names:
            self.imports.add(name.name)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Track function scope."""
        self.scope_stack.append(f"function:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()
        
    def visit_ClassDef(self, node):
        """Track class scope."""
        self.scope_stack.append(f"class:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()
        
    def visit_Assign(self, node):
        """Find constant assignments."""
        # Only care about assignments in the global or class scope
        current_scope = self.scope_stack[-1]
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                
                # Check if this looks like a constant
                is_constant = (
                    (name.isupper() and len(name) > 1) or  # ALL_CAPS
                    (name.startswith('_') and name[1:].isupper() and len(name) > 2)  # _ALL_CAPS
                )
                
                if is_constant and current_scope.startswith(("global", "class")):
                    self.constant_nodes.append(node)
                    
                    # Try to get the constant value
                    if isinstance(node.value, ast.Constant):
                        self.constants[name] = node.value.value
                    else:
                        self.constants[name] = None  # Complex value
        
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node):
        """Find annotated assignments like name: Final[int] = value."""
        if isinstance(node.target, ast.Name):
            name = node.target.id
            
            # Check if this is a Final annotation
            is_final = False
            if isinstance(node.annotation, ast.Subscript):
                if isinstance(node.annotation.value, ast.Name):
                    if node.annotation.value.id == 'Final':
                        is_final = True
            
            if is_final or name.isupper():
                self.constant_nodes.append(node)
                
                # Try to get the constant value
                if node.value and isinstance(node.value, ast.Constant):
                    self.constants[name] = node.value.value
                else:
                    self.constants[name] = None  # Complex value
        
        self.generic_visit(node)
        
    def visit_Name(self, node):
        """Track usage of constants."""
        if isinstance(node.ctx, ast.Load) and node.id in self.constants:
            if node.id not in self.usages:
                self.usages[node.id] = []
            self.usages[node.id].append(node)
        
        self.generic_visit(node)


def is_standard_constant_name(name: str) -> bool:
    """Check if a name follows standard constant naming convention."""
    # Should be ALL_UPPERCASE_WITH_UNDERSCORES
    return name.isupper() and '_' in name and not name.startswith('__')


def convert_to_standard_name(name: str) -> str:
    """Convert a non-standard constant name to the standard format."""
    # Remove leading underscores (keep double __ for special names)
    if name.startswith('_') and not name.startswith('__'):
        name = name[1:]
    
    # Ensure it's uppercase
    return name.upper()


def analyze_file(file_path: Path) -> Tuple[Dict[str, str], Dict[str, List[ast.AST]]]:
    """
    Analyze a Python file for constants and their usage.
    
    Returns:
        Tuple of (constants_to_rename, usage_nodes)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        visitor = ConstantVisitor()
        visitor.visit(tree)
        
        # Identify constants that need to be renamed
        constants_to_rename = {}
        for name in visitor.constants:
            if not is_standard_constant_name(name):
                standard_name = convert_to_standard_name(name)
                constants_to_rename[name] = standard_name
        
        return constants_to_rename, visitor.usages
        
    except SyntaxError:
        print(f"Syntax error in {file_path}, skipping...")
        return {}, {}


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory recursively."""
    py_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common build/dependency directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                  d not in ['__pycache__', 'venv', '.venv', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                py_files.append(Path(root) / file)
                
    return py_files


def rename_constants_in_file(file_path: Path, 
                            constants_to_rename: Dict[str, str], 
                            add_final: bool = True) -> bool:
    """
    Rename constants in a file and optionally add Final type annotations.
    
    Args:
        file_path: Path to the Python file
        constants_to_rename: Dictionary mapping old names to new names
        add_final: Whether to add Final type annotations
        
    Returns:
        True if changes were made, False otherwise
    """
    if not constants_to_rename:
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First pass: simple find and replace for usages
    new_content = content
    for old_name, new_name in constants_to_rename.items():
        # Careful replacement to avoid matching substrings
        pattern = fr'\b{re.escape(old_name)}\b'
        new_content = re.sub(pattern, new_name, new_content)
    
    # Second pass: add Final annotations to declarations if needed
    if add_final:
        for old_name, new_name in constants_to_rename.items():
            # Look for assignment patterns
            assignment_pattern = fr'(?m)^(\s*)({re.escape(new_name)}\s*=\s*.+)$'
            replacement = r'\1\2  # type: Final'
            
            # Only add if there's no annotation already
            if 'Final' not in new_content:
                new_content = re.sub(assignment_pattern, replacement, new_content)
    
    # Only write if changes were made
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    
    return False


def main():
    """Main function to standardize constants across the codebase."""
    print("Scanning codebase for constants...")
    py_files = find_python_files(Path('.'))
    
    # First pass: identify all constants
    all_constants = {}
    for file_path in py_files:
        constants_to_rename, _ = analyze_file(file_path)
        all_constants.update(constants_to_rename)
    
    print(f"Found {len(all_constants)} constants to standardize:")
    for old_name, new_name in all_constants.items():
        print(f"  {old_name} -> {new_name}")
    
    # Second pass: rename constants in all files
    files_modified = 0
    for file_path in py_files:
        if rename_constants_in_file(file_path, all_constants):
            print(f"Updated constants in {file_path}")
            files_modified += 1
    
    print(f"Standardized {len(all_constants)} constants across {files_modified} files")


if __name__ == "__main__":
    main()
