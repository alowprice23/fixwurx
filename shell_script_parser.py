#!/usr/bin/env python3
"""
shell_script_parser.py
────────────────────────
Utility for parsing and executing shell scripts with Markdown-style code blocks.

This script preprocesses shell scripts to handle Markdown-style code blocks (```),
allowing users to run scripts with code examples without syntax errors.
"""

import os
import sys
import re
import argparse
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ShellScriptParser")

def preprocess_script(content):
    """
    Preprocess script content to handle Markdown-style code blocks.
    
    Args:
        content: The script content
        
    Returns:
        Preprocessed script content
    """
    # Remove Markdown code block markers
    # Match ``` or ```language and remove it
    content = re.sub(r'^```[a-zA-Z]*\s*$', '# --- CODE BLOCK START ---', content, flags=re.MULTILINE)
    content = re.sub(r'^```\s*$', '# --- CODE BLOCK END ---', content, flags=re.MULTILINE)
    
    # Remove any other Markdown formatting
    content = re.sub(r'^#+\s+(.*)$', '# \\1', content, flags=re.MULTILINE)  # Headers
    
    return content

def parse_script_file(file_path):
    """
    Parse a script file and handle Markdown-style code blocks.
    
    Args:
        file_path: Path to the script file
        
    Returns:
        Preprocessed script content
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        return preprocess_script(content)
    except Exception as e:
        logger.error(f"Error parsing script file: {e}")
        return None

def execute_script(content, interpreter=None):
    """
    Execute a preprocessed script.
    
    Args:
        content: The preprocessed script content
        interpreter: Optional interpreter to use
        
    Returns:
        Exit code
    """
    try:
        # Create a temporary file for the preprocessed script
        temp_file = Path(".temp_script.fx")
        with open(temp_file, 'w') as f:
            f.write(content)
        
        # Execute the script
        if interpreter:
            cmd = [interpreter, str(temp_file)]
        else:
            cmd = ["fx", str(temp_file)]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        # Clean up
        temp_file.unlink()
        
        return result.returncode
    except Exception as e:
        logger.error(f"Error executing script: {e}")
        return 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Parse and execute shell scripts with Markdown-style code blocks")
    parser.add_argument("file", help="Script file to execute")
    parser.add_argument("--interpreter", "-i", help="Interpreter to use (default: fx)")
    parser.add_argument("--print", "-p", action="store_true", help="Print the preprocessed script instead of executing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse the script file
    content = parse_script_file(args.file)
    if content is None:
        return 1
    
    # Print or execute
    if args.print:
        print(content)
        return 0
    else:
        return execute_script(content, args.interpreter)

if __name__ == "__main__":
    sys.exit(main())
