"""
agent_memory.py
───────────────
Memory management for agents in the FixWurx system.

This module provides the AgentMemory class, which allows agents to store
and retrieve information for neural-based learning and pattern recognition.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass

@dataclass
class SolutionPathVersion:
    """
    Version tracking for solution paths.
    
    This class tracks versions of solution paths, including the original
    and any modified versions, along with metrics about their effectiveness.
    """
    path_id: str
    version: int
    parent_version: Optional[int] = None
    actions: List[Dict[str, Any]] = None
    metrics: Dict[str, Any] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        """Initialize derived fields after initialization."""
        if self.actions is None:
            self.actions = []
        if self.metrics is None:
            self.metrics = {}
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON serializable dictionary."""
        return {
            "path_id": self.path_id,
            "version": self.version,
            "parent_version": self.parent_version,
            "actions": self.actions,
            "metrics": self.metrics,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SolutionPathVersion":
        """Create from JSON dictionary."""
        return cls(
            path_id=data["path_id"],
            version=data["version"],
            parent_version=data.get("parent_version"),
            actions=data.get("actions", []),
            metrics=data.get("metrics", {}),
            created_at=data.get("created_at", time.time())
        )

class AgentMemory:
    """
    Memory system for agents in the FixWurx system.
    
    The AgentMemory class provides persistent storage capabilities for agents
    to remember past actions, decisions, and results for learning purposes.
    """
    
    def __init__(self, storage_path: str = ".triangulum/memory", retention_days: int = 30, 
                 mem_path: str = None, kv_path: str = None, compressed_path: str = None,
                 family_tree_path: str = None):
        """
        Initialize the agent memory.
        
        Args:
            storage_path: Path to store memory data
            retention_days: Number of days to retain memory
            mem_path: Alternative path parameter (for backward compatibility)
            kv_path: Another alternative path parameter (for backward compatibility)
            compressed_path: Path for compressed storage (for backward compatibility)
            family_tree_path: Path for family tree storage (for backward compatibility)
        """
        # Use alternative paths if provided (for backward compatibility)
        if mem_path is not None:
            storage_path = mem_path
        elif kv_path is not None:
            storage_path = kv_path
        elif compressed_path is not None:
            storage_path = compressed_path
        elif family_tree_path is not None:
            storage_path = family_tree_path
            
        self.storage_path = Path(storage_path)
        self.retention_days = retention_days
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value in memory.
        
        Args:
            key: Key to store the value under
            value: Value to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Create a filename from the key
            filename = self._key_to_filename(key)
            filepath = self.storage_path / filename
            
            # Ensure value is serializable
            if hasattr(value, "to_json"):
                value = value.to_json()
            
            # Add timestamp if not present
            if isinstance(value, dict) and "timestamp" not in value:
                value["timestamp"] = time.time()
            
            # Write to file
            with open(filepath, "w") as f:
                json.dump(value, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error storing memory {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from memory.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Retrieved value or None if not found
        """
        try:
            # Create a filename from the key
            filename = self._key_to_filename(key)
            filepath = self.storage_path / filename
            
            # Check if file exists
            if not filepath.exists():
                return None
            
            # Read from file
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error retrieving memory {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from memory.
        
        Args:
            key: Key to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Create a filename from the key
            filename = self._key_to_filename(key)
            filepath = self.storage_path / filename
            
            # Check if file exists
            if not filepath.exists():
                return False
            
            # Delete file
            os.remove(filepath)
            
            return True
        except Exception as e:
            print(f"Error deleting memory {key}: {e}")
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List all keys in memory.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of keys
        """
        try:
            # Get all files in storage directory
            files = list(self.storage_path.glob("*.json"))
            
            # Convert filenames to keys
            keys = [self._filename_to_key(f.name) for f in files]
            
            # Filter by prefix if provided
            if prefix:
                keys = [k for k in keys if k.startswith(prefix)]
            
            return keys
        except Exception as e:
            print(f"Error listing memory keys: {e}")
            return []
    
    def cleanup(self) -> int:
        """
        Clean up old memory entries.
        
        Returns:
            Number of entries deleted
        """
        try:
            # Calculate cutoff timestamp
            cutoff = time.time() - (self.retention_days * 24 * 60 * 60)
            
            # Get all files in storage directory
            files = list(self.storage_path.glob("*.json"))
            
            # Track number of deleted files
            deleted = 0
            
            # Check each file
            for filepath in files:
                try:
                    # Read file to get timestamp
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    
                    # Get timestamp
                    timestamp = data.get("timestamp", 0)
                    
                    # Delete if older than cutoff
                    if timestamp < cutoff:
                        os.remove(filepath)
                        deleted += 1
                except:
                    # Skip files that can't be processed
                    continue
            
            return deleted
        except Exception as e:
            print(f"Error cleaning up memory: {e}")
            return 0
    
    def _key_to_filename(self, key: str) -> str:
        """Convert a key to a valid filename."""
        # Replace invalid filename characters
        filename = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        
        # Add extension
        return f"{filename}.json"
    
    def _filename_to_key(self, filename: str) -> str:
        """Convert a filename back to a key."""
        # Remove extension
        if filename.endswith(".json"):
            return filename[:-5]
        return filename
