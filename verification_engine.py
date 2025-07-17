#!/usr/bin/env python3
"""
verification_engine.py
──────────────────────
Verification engine for the FixWurx system.

This module provides the VerificationEngine class, which manages
verification of solution correctness and neural matrix integrity.
"""

import os
import json
import time
import hashlib
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification_engine")

class VerificationEngine:
    """
    Engine for verifying solution correctness and neural matrix integrity.
    
    The VerificationEngine manages verification of solution correctness,
    neural matrix integrity, and other system components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the verification engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("verification_engine")
        
        # Configuration options
        self.neural_enabled = self.config.get("neural_matrix", {}).get("enabled", False)
        self._neural_matrix_path = Path(self.config.get("neural_matrix", {}).get("path", ".triangulum/neural_matrix"))
        self._db_path = self.config.get("neural_matrix", {}).get("db_path", str(self._neural_matrix_path / "patterns" / "patterns.db"))
        
        # Neural matrix integrity tracking
        self._last_neural_matrix_hash = None
        self._neural_weights_cache = {}
        self._neural_weight_update_time = 0
        
        # Initialize neural matrix verification
        if self.neural_enabled:
            self._check_neural_weights()
            self._check_neural_patterns()
    
    def verify_solution(self, bug_id: str, solution: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a solution for a bug.
        
        Args:
            bug_id: ID of the bug
            solution: Solution to verify
            
        Returns:
            Tuple of (success, details)
        """
        # Placeholder for solution verification
        # In a real implementation, this would run tests, verify fix, etc.
        return True, {"message": "Solution verified"}
    
    def _check_neural_matrix_integrity(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check the integrity of the neural matrix.
        
        Args:
            options: Optional configuration options
            
        Returns:
            Dictionary with check results
        """
        results = {
            "success": True,
            "checks": [],
            "timestamp": time.time()
        }
        
        # Check if directories exist
        required_dirs = ["patterns", "weights", "history", "connections"]
        for dir_name in required_dirs:
            dir_path = self._neural_matrix_path / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            
            results["checks"].append({
                "name": f"directory_{dir_name}",
                "passed": exists,
                "path": str(dir_path)
            })
            
            if not exists:
                results["success"] = False
        
        # Check if weights file exists
        weights_file = self._neural_matrix_path / "weights" / "default_weights.json"
        weights_exist = weights_file.exists() and weights_file.is_file()
        
        results["checks"].append({
            "name": "weights_file",
            "passed": weights_exist,
            "path": str(weights_file)
        })
        
        if not weights_exist:
            results["success"] = False
        
        # Check if database exists
        db_exists = Path(self._db_path).exists() and Path(self._db_path).is_file()
        
        results["checks"].append({
            "name": "database",
            "passed": db_exists,
            "path": str(self._db_path)
        })
        
        if not db_exists:
            results["success"] = False
        
        # Calculate integrity hash
        if results["success"]:
            hash_value = self._calculate_neural_matrix_hash()
            results["hash"] = hash_value
            
            # Check if hash changed
            if self._last_neural_matrix_hash is not None and self._last_neural_matrix_hash != hash_value:
                self.logger.info(f"Neural matrix hash changed: {self._last_neural_matrix_hash} -> {hash_value}")
                results["hash_changed"] = True
            else:
                results["hash_changed"] = False
            
            # Update last hash
            self._last_neural_matrix_hash = hash_value
        
        return results
    
    def _calculate_neural_matrix_hash(self) -> str:
        """
        Calculate a hash of the neural matrix state.
        
        Returns:
            Hash string
        """
        hash_components = []
        
        # Add weights file to hash
        weights_file = self._neural_matrix_path / "weights" / "default_weights.json"
        if weights_file.exists():
            with open(weights_file, 'rb') as f:
                weights_data = f.read()
                hash_components.append(hashlib.sha256(weights_data).hexdigest())
        
        # Add database structure to hash
        if Path(self._db_path).exists():
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            # Get pattern count
            cursor.execute("SELECT COUNT(*) FROM neural_patterns")
            pattern_count = cursor.fetchone()[0]
            hash_components.append(f"patterns:{pattern_count}")
            
            # Get weight count
            cursor.execute("SELECT COUNT(*) FROM neural_weights")
            weight_count = cursor.fetchone()[0]
            hash_components.append(f"weights:{weight_count}")
            
            conn.close()
        
        # Calculate combined hash
        combined = ":".join(hash_components)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _check_neural_weights(self) -> None:
        """
        Check and load neural weights.
        
        This method checks if the neural weights file exists and loads it.
        """
        weights_file = self._neural_matrix_path / "weights" / "default_weights.json"
        
        if weights_file.exists():
            try:
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                
                # Extract agent weights
                for agent, weight in weights_data.get("agent_weights", {}).items():
                    self._neural_weights_cache[agent] = weight
                
                # Extract feature weights
                for feature, weight in weights_data.get("feature_weights", {}).items():
                    self._neural_weights_cache[f"feature_{feature}"] = weight
                
                self._neural_weight_update_time = time.time()
                self.logger.info(f"Loaded {len(self._neural_weights_cache)} neural weights")
            except Exception as e:
                self.logger.error(f"Error loading neural weights: {e}")
        else:
            self.logger.warning(f"Neural weights file not found: {weights_file}")
    
    def _check_neural_patterns(self) -> None:
        """
        Check neural patterns database.
        
        This method checks if the neural patterns database exists and is valid.
        """
        db_path = Path(self._db_path)
        
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neural_patterns'")
                has_patterns_table = cursor.fetchone() is not None
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neural_weights'")
                has_weights_table = cursor.fetchone() is not None
                
                if has_patterns_table and has_weights_table:
                    # Count patterns
                    cursor.execute("SELECT COUNT(*) FROM neural_patterns")
                    pattern_count = cursor.fetchone()[0]
                    
                    # Count weights
                    cursor.execute("SELECT COUNT(*) FROM neural_weights")
                    weight_count = cursor.fetchone()[0]
                    
                    self.logger.info(f"Neural database contains {pattern_count} patterns and {weight_count} weights")
                else:
                    missing_tables = []
                    if not has_patterns_table:
                        missing_tables.append("neural_patterns")
                    if not has_weights_table:
                        missing_tables.append("neural_weights")
                    
                    self.logger.warning(f"Neural database missing tables: {', '.join(missing_tables)}")
                
                conn.close()
            except Exception as e:
                self.logger.error(f"Error checking neural patterns database: {e}")
        else:
            self.logger.warning(f"Neural patterns database not found: {db_path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the verification engine.
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "neural_enabled": self.neural_enabled,
            "neural_weight_count": len(self._neural_weights_cache),
            "neural_weight_update_time": self._neural_weight_update_time
        }
        
        # Add neural matrix hash if available
        if self._last_neural_matrix_hash is not None:
            metrics["neural_matrix_hash"] = self._last_neural_matrix_hash
        
        return metrics
