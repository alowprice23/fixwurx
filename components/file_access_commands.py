#!/usr/bin/env python3
"""
File Access Commands

This module provides CLI commands for file access operations using the File Access Utility.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("file_access.log")
    ]
)
logger = logging.getLogger("FileAccessCommands")

def get_commands(registry):
    """
    Get the file access commands.
    
    Args:
        registry: Component registry
        
    Returns:
        Dictionary of commands
    """
    file_access = registry.get_component("file_access_utility")
    if not file_access:
        logger.error("File Access Utility not available")
        return {}
    
    return {
        "read:file": lambda args: read_file(file_access, args),
        "write:file": lambda args: write_file(file_access, args),
        "list:directory": lambda args: list_directory(file_access, args),
        "search:files": lambda args: search_files(file_access, args),
        "copy:file": lambda args: copy_file(file_access, args)
    }

def read_file(file_access, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read a file.
    
    Args:
        file_access: File Access Utility
        args: Command arguments
        
    Returns:
        Command result
    """
    try:
        path = args.get("path")
        if not path:
            return {"success": False, "error": "Missing required argument: path"}
        
        result = file_access.read_file(path)
        
        if not result.get("success", False):
            return result
        
        # Format result for display
        if result.get("is_binary", False):
            return {
                "success": True,
                "output": f"Binary file: {path} ({result.get('size', 0)} bytes)",
                "binary": True,
                "path": path,
                "size": result.get("size", 0)
            }
        else:
            return {
                "success": True,
                "output": f"File: {path} ({result.get('size', 0)} bytes)\n\n{result.get('content', '')}",
                "content": result.get("content", ""),
                "path": path,
                "size": result.get("size", 0)
            }
    
    except Exception as e:
        logger.error(f"Error in read_file command: {e}")
        return {"success": False, "error": f"Error reading file: {e}"}

def write_file(file_access, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write to a file.
    
    Args:
        file_access: File Access Utility
        args: Command arguments
        
    Returns:
        Command result
    """
    try:
        path = args.get("path")
        content = args.get("content")
        
        if not path:
            return {"success": False, "error": "Missing required argument: path"}
        if content is None:
            return {"success": False, "error": "Missing required argument: content"}
        
        result = file_access.write_file(path, content)
        
        if not result.get("success", False):
            return result
        
        return {
            "success": True,
            "output": f"File written: {path} ({result.get('size', 0)} bytes)",
            "path": path,
            "size": result.get("size", 0)
        }
    
    except Exception as e:
        logger.error(f"Error in write_file command: {e}")
        return {"success": False, "error": f"Error writing file: {e}"}

def list_directory(file_access, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    List a directory.
    
    Args:
        file_access: File Access Utility
        args: Command arguments
        
    Returns:
        Command result
    """
    try:
        path = args.get("path")
        recursive = args.get("recursive", False)
        
        if not path:
            return {"success": False, "error": "Missing required argument: path"}
        
        result = file_access.list_directory(path, recursive)
        
        if not result.get("success", False):
            return result
        
        files = result.get("files", [])
        output = f"Directory: {path} ({len(files)} files)\n\n"
        for file in files:
            output += f"- {file}\n"
        
        return {
            "success": True,
            "output": output,
            "files": files,
            "path": path,
            "count": len(files)
        }
    
    except Exception as e:
        logger.error(f"Error in list_directory command: {e}")
        return {"success": False, "error": f"Error listing directory: {e}"}

def search_files(file_access, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search files for a pattern.
    
    Args:
        file_access: File Access Utility
        args: Command arguments
        
    Returns:
        Command result
    """
    try:
        path = args.get("path")
        pattern = args.get("pattern")
        recursive = args.get("recursive", True)
        
        if not path:
            return {"success": False, "error": "Missing required argument: path"}
        if not pattern:
            return {"success": False, "error": "Missing required argument: pattern"}
        
        result = file_access.search_files(path, pattern, recursive)
        
        if not result.get("success", False):
            return result
        
        matches = result.get("matches", [])
        output = f"Search results for '{pattern}' in {path} ({len(matches)} files matched)\n\n"
        
        for match in matches:
            file = match.get("file", "")
            match_contexts = match.get("matches", [])
            
            output += f"File: {file}\n"
            
            for context in match_contexts:
                line_number = context.get("line_number", 0)
                context_text = context.get("context", "")
                
                output += f"  Line {line_number}:\n"
                
                # Add indentation to context text
                context_lines = context_text.split("\n")
                for line in context_lines:
                    output += f"    {line}\n"
                
                output += "\n"
        
        return {
            "success": True,
            "output": output,
            "matches": matches,
            "path": path,
            "pattern": pattern,
            "count": len(matches)
        }
    
    except Exception as e:
        logger.error(f"Error in search_files command: {e}")
        return {"success": False, "error": f"Error searching files: {e}"}

def copy_file(file_access, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy a file.
    
    Args:
        file_access: File Access Utility
        args: Command arguments
        
    Returns:
        Command result
    """
    try:
        source = args.get("source")
        destination = args.get("destination")
        
        if not source:
            return {"success": False, "error": "Missing required argument: source"}
        if not destination:
            return {"success": False, "error": "Missing required argument: destination"}
        
        result = file_access.copy_file(source, destination)
        
        if not result.get("success", False):
            return result
        
        return {
            "success": True,
            "output": f"File copied: {source} -> {destination} ({result.get('size', 0)} bytes)",
            "source": source,
            "destination": destination,
            "size": result.get("size", 0)
        }
    
    except Exception as e:
        logger.error(f"Error in copy_file command: {e}")
        return {"success": False, "error": f"Error copying file: {e}"}
