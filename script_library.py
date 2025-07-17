#!/usr/bin/env python3
"""
Script Library

This module provides script storage, versioning, and management.
"""

import os
import sys
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger("ScriptLibrary")

class ScriptLibrary:
    """
    Script Library for storing, managing, and retrieving scripts.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Script Library.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configuration parameters
        self.library_path = self.config.get("library_path", "script_library")
        self.git_enabled = self.config.get("git_enabled", True)
        
        # Storage for scripts
        self.scripts = {}
        
        # Register with registry
        registry.register_component("script_library", self)
        
        logger.info("Script Library initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the Script Library.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Script Library already initialized")
            return True
        
        try:
            # Create library directory if it doesn't exist
            if not os.path.exists(self.library_path):
                os.makedirs(self.library_path)
            
            # Load existing scripts
            self._load_scripts()
            
            self.initialized = True
            logger.info("Script Library initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Script Library: {e}")
            return False
    
    def _load_scripts(self) -> None:
        """Load existing scripts from the library path."""
        try:
            # Load scripts from index file if it exists
            index_path = os.path.join(self.library_path, "index.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    self.scripts = json.load(f)
            else:
                self.scripts = {}
                
            logger.info(f"Loaded {len(self.scripts)} scripts from library")
        except Exception as e:
            logger.error(f"Error loading scripts: {e}")
            self.scripts = {}
    
    def _save_scripts(self) -> None:
        """Save scripts to the index file."""
        try:
            # Create library directory if it doesn't exist
            if not os.path.exists(self.library_path):
                os.makedirs(self.library_path)
            
            # Save scripts to index file
            index_path = os.path.join(self.library_path, "index.json")
            with open(index_path, "w") as f:
                json.dump(self.scripts, f, indent=2)
            
            logger.info(f"Saved {len(self.scripts)} scripts to library")
        except Exception as e:
            logger.error(f"Error saving scripts: {e}")
    
    def add_script(self, script_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a script to the library.
        
        Args:
            script_content: Script content
            metadata: Script metadata
            
        Returns:
            Dictionary with result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Library initialization failed")
                return {"success": False, "error": "Script Library initialization failed"}
        
        try:
            # Generate script ID
            script_hash = hashlib.md5(script_content.encode()).hexdigest()
            script_id = f"script_{int(time.time())}_{script_hash[:8]}"
            
            # Create script directory
            script_dir = os.path.join(self.library_path, script_id)
            if not os.path.exists(script_dir):
                os.makedirs(script_dir)
            
            # Save script content
            script_path = os.path.join(script_dir, "script.fx")
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Save metadata
            metadata_path = os.path.join(script_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Update script index
            self.scripts[script_id] = {
                "id": script_id,
                "name": metadata.get("name", "Unnamed Script"),
                "description": metadata.get("description", ""),
                "author": metadata.get("author", "unknown"),
                "version": metadata.get("version", "1.0"),
                "tags": metadata.get("tags", []),
                "created": time.time(),
                "modified": time.time(),
                "metadata": metadata
            }
            
            # Save script index
            self._save_scripts()
            
            logger.info(f"Added script {script_id} to library")
            
            return {
                "success": True,
                "script_id": script_id
            }
        except Exception as e:
            logger.error(f"Error adding script: {e}")
            return {"success": False, "error": str(e)}
    
    def get_script(self, script_id: str) -> Dict[str, Any]:
        """
        Get a script from the library.
        
        Args:
            script_id: Script ID
            
        Returns:
            Dictionary with script content and metadata
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Library initialization failed")
                return {"success": False, "error": "Script Library initialization failed"}
        
        try:
            # Check if script exists
            if script_id not in self.scripts:
                return {"success": False, "error": f"Script {script_id} not found"}
            
            # Get script content
            script_path = os.path.join(self.library_path, script_id, "script.fx")
            if not os.path.exists(script_path):
                return {"success": False, "error": f"Script file not found: {script_path}"}
            
            with open(script_path, "r") as f:
                script_content = f.read()
            
            # Get metadata
            metadata_path = os.path.join(self.library_path, script_id, "metadata.json")
            if not os.path.exists(metadata_path):
                return {"success": False, "error": f"Metadata file not found: {metadata_path}"}
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            logger.info(f"Retrieved script {script_id} from library")
            
            return {
                "success": True,
                "script_id": script_id,
                "content": script_content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error getting script: {e}")
            return {"success": False, "error": str(e)}
    
    def update_script(self, script_id: str, script_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a script in the library.
        
        Args:
            script_id: Script ID
            script_content: Script content
            metadata: Script metadata
            
        Returns:
            Dictionary with result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Library initialization failed")
                return {"success": False, "error": "Script Library initialization failed"}
        
        try:
            # Check if script exists
            if script_id not in self.scripts:
                return {"success": False, "error": f"Script {script_id} not found"}
            
            # Update script content
            script_path = os.path.join(self.library_path, script_id, "script.fx")
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Update metadata
            metadata_path = os.path.join(self.library_path, script_id, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Update script index
            self.scripts[script_id].update({
                "name": metadata.get("name", self.scripts[script_id].get("name", "Unnamed Script")),
                "description": metadata.get("description", self.scripts[script_id].get("description", "")),
                "version": metadata.get("version", self.scripts[script_id].get("version", "1.0")),
                "tags": metadata.get("tags", self.scripts[script_id].get("tags", [])),
                "modified": time.time(),
                "metadata": metadata
            })
            
            # Save script index
            self._save_scripts()
            
            logger.info(f"Updated script {script_id} in library")
            
            return {
                "success": True,
                "script_id": script_id
            }
        except Exception as e:
            logger.error(f"Error updating script: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_script(self, script_id: str) -> Dict[str, Any]:
        """
        Delete a script from the library.
        
        Args:
            script_id: Script ID
            
        Returns:
            Dictionary with result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Library initialization failed")
                return {"success": False, "error": "Script Library initialization failed"}
        
        try:
            # Check if script exists
            if script_id not in self.scripts:
                return {"success": False, "error": f"Script {script_id} not found"}
            
            # Delete script directory
            script_dir = os.path.join(self.library_path, script_id)
            if os.path.exists(script_dir):
                import shutil
                shutil.rmtree(script_dir)
            
            # Remove from script index
            del self.scripts[script_id]
            
            # Save script index
            self._save_scripts()
            
            logger.info(f"Deleted script {script_id} from library")
            
            return {
                "success": True
            }
        except Exception as e:
            logger.error(f"Error deleting script: {e}")
            return {"success": False, "error": str(e)}
    
    def list_scripts(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        List scripts in the library.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            Dictionary with script list
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Library initialization failed")
                return {"success": False, "error": "Script Library initialization failed"}
        
        try:
            # Filter scripts by tags if provided
            if tags:
                filtered_scripts = {}
                for script_id, script in self.scripts.items():
                    script_tags = script.get("tags", [])
                    if any(tag in script_tags for tag in tags):
                        filtered_scripts[script_id] = script
            else:
                filtered_scripts = self.scripts
            
            logger.info(f"Listed {len(filtered_scripts)} scripts from library")
            
            return {
                "success": True,
                "scripts": filtered_scripts,
                "count": len(filtered_scripts)
            }
        except Exception as e:
            logger.error(f"Error listing scripts: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self) -> None:
        """
        Shutdown the Script Library.
        """
        if not self.initialized:
            return
        
        # Save script index
        self._save_scripts()
        
        self.initialized = False
        logger.info("Script Library shutdown complete")

# Singleton instance
_instance = None

def get_instance(registry, config: Optional[Dict[str, Any]] = None) -> ScriptLibrary:
    """
    Get the singleton instance of the Script Library.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        ScriptLibrary instance
    """
    global _instance
    if _instance is None:
        _instance = ScriptLibrary(registry, config)
    return _instance
