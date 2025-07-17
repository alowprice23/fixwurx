#!/usr/bin/env python3
"""
Rollback Manager Component for Triangulum Integration

This module provides the RollbackManager class for managing system state snapshots.
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional

# Configure logging if not already configured
logger = logging.getLogger("TriangulumIntegration")

# Mock mode for testing
MOCK_MODE = os.environ.get("TRIANGULUM_TEST_MODE", "0") == "1"

class RollbackManager:
    """
    Manages system state snapshots and rollbacks.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize rollback manager.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.snapshot_dir = self.config.get("snapshot_dir", "snapshots")
        self.snapshots = {}
        
        # Create snapshot directory if it doesn't exist
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        logger.info(f"Rollback manager initialized with snapshot dir {self.snapshot_dir}")
    
    def create_snapshot(self, name: str, data: Dict[str, Any]) -> str:
        """
        Create a snapshot of the current state.
        
        Args:
            name: Name of the snapshot
            data: Data to snapshot
            
        Returns:
            ID of the created snapshot
        """
        snapshot_id = str(uuid.uuid4())
        
        snapshot = {
            "id": snapshot_id,
            "name": name,
            "timestamp": time.time(),
            "state": data
        }
        
        self.snapshots[snapshot_id] = snapshot
        
        # Write snapshot to disk (simplified for testing)
        if not MOCK_MODE:
            try:
                snapshot_path = os.path.join(self.snapshot_dir, f"{snapshot_id}.json")
                with open(snapshot_path, "w") as f:
                    json.dump(snapshot, f, indent=2)
            except Exception as e:
                logger.error(f"Error writing snapshot to disk: {e}")
        
        logger.info(f"Created snapshot: {name} ({snapshot_id})")
        return snapshot_id
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a snapshot by ID.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            Snapshot data, or None if not found
        """
        if snapshot_id not in self.snapshots:
            logger.warning(f"Snapshot {snapshot_id} not found")
            return None
        
        return self.snapshots[snapshot_id]
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all snapshots.
        
        Returns:
            List of snapshots
        """
        return list(self.snapshots.values())
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            Whether the snapshot was deleted
        """
        if snapshot_id not in self.snapshots:
            logger.warning(f"Snapshot {snapshot_id} not found")
            return False
        
        # Delete snapshot from disk (simplified for testing)
        if not MOCK_MODE:
            try:
                snapshot_path = os.path.join(self.snapshot_dir, f"{snapshot_id}.json")
                if os.path.exists(snapshot_path):
                    os.remove(snapshot_path)
            except Exception as e:
                logger.error(f"Error deleting snapshot from disk: {e}")
        
        del self.snapshots[snapshot_id]
        logger.info(f"Deleted snapshot: {snapshot_id}")
        return True
    
    def rollback(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Rollback to a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            Snapshot data, or None if not found
        """
        if snapshot_id not in self.snapshots:
            logger.warning(f"Snapshot {snapshot_id} not found")
            return None
        
        snapshot = self.snapshots[snapshot_id]
        logger.info(f"Rolled back to snapshot: {snapshot['name']} ({snapshot_id})")
        return snapshot
