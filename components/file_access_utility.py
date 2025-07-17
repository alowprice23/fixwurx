#!/usr/bin/env python3
"""
File Access Utility

This module provides secure file access capabilities beyond the current working directory,
allowing FixWurx to analyze and repair files in external directories like Carewurx.
"""

import os
import sys
import json
import logging
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("file_access.log")
    ]
)
logger = logging.getLogger("FileAccessUtility")

class FileAccessUtility:
    """
    Utility for securely accessing files and directories outside the working directory.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the File Access Utility.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configure access controls
        self.allowed_paths = self.config.get("allowed_paths", [])
        # If no paths are explicitly allowed, we'll allow everything but with warnings
        if not self.allowed_paths:
            self.allow_all = True
            logger.warning("No allowed paths configured. All paths will be accessible (not recommended for production).")
        else:
            self.allow_all = False
            logger.info(f"Configured {len(self.allowed_paths)} allowed paths")
        
        # Register with registry
        registry.register_component("file_access_utility", self)
        
        logger.info("File Access Utility initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the File Access Utility.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("File Access Utility already initialized")
            return True
        
        try:
            self.initialized = True
            logger.info("File Access Utility initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing File Access Utility: {e}")
            return False
    
    def _check_path_allowed(self, path: str) -> bool:
        """
        Check if access to a path is allowed.
        
        Args:
            path: Path to check
            
        Returns:
            True if access is allowed, False otherwise
        """
        if self.allow_all:
            logger.warning(f"Allowing access to path outside allowed list: {path}")
            return True
        
        # Normalize path for comparison
        norm_path = os.path.normpath(path)
        
        # Check if path is in allowed paths
        for allowed_path in self.allowed_paths:
            norm_allowed = os.path.normpath(allowed_path)
            if norm_path == norm_allowed or norm_path.startswith(norm_allowed + os.sep):
                return True
        
        logger.warning(f"Access denied to path: {path}")
        return False
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read a file from any location.
        
        Args:
            path: Path to the file
            
        Returns:
            Dictionary with read result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("File Access Utility initialization failed")
                return {"success": False, "error": "File Access Utility initialization failed"}
        
        try:
            # Check if path is allowed
            if not self._check_path_allowed(path):
                return {
                    "success": False,
                    "error": f"Access denied to path: {path}"
                }
            
            # Check if file exists
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
            
            # Check if path is a file
            if not os.path.isfile(path):
                logger.error(f"Path is not a file: {path}")
                return {
                    "success": False,
                    "error": f"Path is not a file: {path}"
                }
            
            # Read file
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            logger.info(f"Read file: {path}")
            
            return {
                "success": True,
                "content": content,
                "path": path,
                "size": os.path.getsize(path)
            }
        
        except UnicodeDecodeError:
            # Try binary mode for non-text files
            try:
                with open(path, "rb") as f:
                    content = f.read()
                
                logger.info(f"Read binary file: {path}")
                
                return {
                    "success": True,
                    "content": "Binary file content not displayed",
                    "is_binary": True,
                    "path": path,
                    "size": os.path.getsize(path)
                }
            except Exception as e:
                logger.error(f"Error reading binary file: {e}")
                return {
                    "success": False,
                    "error": f"Error reading binary file: {e}"
                }
        
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {
                "success": False,
                "error": f"Error reading file: {e}"
            }
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write to a file at any location.
        
        Args:
            path: Path to the file
            content: Content to write
            
        Returns:
            Dictionary with write result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("File Access Utility initialization failed")
                return {"success": False, "error": "File Access Utility initialization failed"}
        
        try:
            # Check if path is allowed
            if not self._check_path_allowed(path):
                return {
                    "success": False,
                    "error": f"Access denied to path: {path}"
                }
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"Wrote file: {path}")
            
            return {
                "success": True,
                "path": path,
                "size": os.path.getsize(path)
            }
        
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            return {
                "success": False,
                "error": f"Error writing file: {e}"
            }
    
    def list_directory(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            path: Path to the directory
            recursive: Whether to list recursively
            
        Returns:
            Dictionary with directory listing
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("File Access Utility initialization failed")
                return {"success": False, "error": "File Access Utility initialization failed"}
        
        try:
            # Check if path is allowed
            if not self._check_path_allowed(path):
                return {
                    "success": False,
                    "error": f"Access denied to path: {path}"
                }
            
            # Check if directory exists
            if not os.path.exists(path):
                logger.error(f"Directory not found: {path}")
                return {
                    "success": False,
                    "error": f"Directory not found: {path}"
                }
            
            # Check if path is a directory
            if not os.path.isdir(path):
                logger.error(f"Path is not a directory: {path}")
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }
            
            # List directory
            if recursive:
                file_list = []
                for root, dirs, files in os.walk(path):
                    # Get relative path for cleaner output
                    rel_root = os.path.relpath(root, path)
                    if rel_root == ".":
                        rel_root = ""
                    
                    for file in files:
                        file_path = os.path.join(rel_root, file) if rel_root else file
                        file_list.append(file_path)
            else:
                file_list = os.listdir(path)
            
            logger.info(f"Listed directory: {path}")
            
            return {
                "success": True,
                "path": path,
                "files": file_list,
                "count": len(file_list)
            }
        
        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return {
                "success": False,
                "error": f"Error listing directory: {e}"
            }
    
    def search_files(self, path: str, pattern: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Search files in a directory for a pattern.
        
        Args:
            path: Path to the directory
            pattern: Pattern to search for
            recursive: Whether to search recursively
            
        Returns:
            Dictionary with search result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("File Access Utility initialization failed")
                return {"success": False, "error": "File Access Utility initialization failed"}
        
        try:
            # Check if path is allowed
            if not self._check_path_allowed(path):
                return {
                    "success": False,
                    "error": f"Access denied to path: {path}"
                }
            
            # Check if directory exists
            if not os.path.exists(path):
                logger.error(f"Directory not found: {path}")
                return {
                    "success": False,
                    "error": f"Directory not found: {path}"
                }
            
            # Check if path is a directory
            if not os.path.isdir(path):
                logger.error(f"Path is not a directory: {path}")
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }
            
            # Search files
            matches = []
            
            # List files to search
            files_to_search = []
            if recursive:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        files_to_search.append(os.path.join(root, file))
            else:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isfile(item_path):
                        files_to_search.append(item_path)
            
            # Search each file
            for file_path in files_to_search:
                try:
                    # Try to read as text
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if pattern in content:
                        # Get context around match
                        context = []
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if pattern in line:
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                context.append({
                                    "line_number": i + 1,
                                    "context": "\n".join(lines[start:end])
                                })
                        
                        matches.append({
                            "file": file_path,
                            "matches": context
                        })
                except:
                    # Skip binary files or files with encoding issues
                    pass
            
            logger.info(f"Searched files in: {path}")
            
            return {
                "success": True,
                "path": path,
                "pattern": pattern,
                "matches": matches,
                "count": len(matches)
            }
        
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return {
                "success": False,
                "error": f"Error searching files: {e}"
            }
    
    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a file from one location to another.
        
        Args:
            source: Source path
            destination: Destination path
            
        Returns:
            Dictionary with copy result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("File Access Utility initialization failed")
                return {"success": False, "error": "File Access Utility initialization failed"}
        
        try:
            # Check if paths are allowed
            if not self._check_path_allowed(source) or not self._check_path_allowed(destination):
                return {
                    "success": False,
                    "error": f"Access denied to path: {source} or {destination}"
                }
            
            # Check if source exists
            if not os.path.exists(source):
                logger.error(f"Source file not found: {source}")
                return {
                    "success": False,
                    "error": f"Source file not found: {source}"
                }
            
            # Check if source is a file
            if not os.path.isfile(source):
                logger.error(f"Source path is not a file: {source}")
                return {
                    "success": False,
                    "error": f"Source path is not a file: {source}"
                }
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(destination)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Copy file
            shutil.copy2(source, destination)
            
            logger.info(f"Copied file from {source} to {destination}")
            
            return {
                "success": True,
                "source": source,
                "destination": destination,
                "size": os.path.getsize(destination)
            }
        
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return {
                "success": False,
                "error": f"Error copying file: {e}"
            }
    
    def shutdown(self) -> None:
        """
        Shutdown the File Access Utility.
        """
        if not self.initialized:
            return
        
        self.initialized = False
        logger.info("File Access Utility shutdown complete")

# Singleton instance
_instance = None

def get_instance(registry, config: Optional[Dict[str, Any]] = None) -> FileAccessUtility:
    """
    Get the singleton instance of the File Access Utility.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        FileAccessUtility instance
    """
    global _instance
    if _instance is None:
        _instance = FileAccessUtility(registry, config)
    return _instance
