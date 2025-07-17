#!/usr/bin/env python3
"""
neural_matrix_init.py
─────────────────────
Initialization utility for the Neural Matrix.

This module provides functions for initializing the Neural Matrix directory
structure, default weights, and pattern database.
"""

import os
import json
import time
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural_matrix_init")

def create_directory_structure(base_dir: str) -> Dict[str, Any]:
    """
    Create the directory structure for the Neural Matrix.
    
    Args:
        base_dir: Base directory for the neural matrix
        
    Returns:
        Dictionary with creation result
    """
    try:
        base_path = Path(base_dir)
        
        # Create base directory
        os.makedirs(base_path, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["patterns", "weights", "history", "connections", "test_data"]
        for subdir in subdirs:
            os.makedirs(base_path / subdir, exist_ok=True)
        
        logger.info(f"Created neural matrix directory structure at {base_dir}")
        
        return {
            "success": True,
            "base_dir": str(base_path),
            "subdirectories": subdirs
        }
    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def initialize_neural_weights(base_dir: str) -> Dict[str, Any]:
    """
    Initialize neural weights with default values.
    
    Args:
        base_dir: Base directory for the neural matrix
        
    Returns:
        Dictionary with initialization result
    """
    try:
        base_path = Path(base_dir)
        weights_dir = base_path / "weights"
        weights_file = weights_dir / "default_weights.json"
        
        # Create weights directory if it doesn't exist
        os.makedirs(weights_dir, exist_ok=True)
        
        # Create default weights
        default_weights = {
            "agent_weights": {
                "observer": 1.0,
                "analyst": 1.2,
                "verifier": 0.9,
                "planner": 1.5
            },
            "feature_weights": {
                "pattern_match": 1.5,
                "entropy": 0.8,
                "fallback": 0.5,
                "similarity": 1.2
            },
            "learning_parameters": {
                "learning_rate": 0.1,
                "pattern_recognition_threshold": 0.7,
                "solution_history_limit": 1000,
                "weight_update_frequency": 10
            },
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Write weights to file
        with open(weights_file, 'w') as f:
            json.dump(default_weights, f, indent=2)
        
        logger.info(f"Initialized neural weights at {weights_file}")
        
        return {
            "success": True,
            "weights_file": str(weights_file),
            "weights": default_weights
        }
    except Exception as e:
        logger.error(f"Error initializing neural weights: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def initialize_pattern_database(base_dir: str) -> Dict[str, Any]:
    """
    Initialize pattern database with starter patterns.
    
    Args:
        base_dir: Base directory for the neural matrix
        
    Returns:
        Dictionary with initialization result
    """
    try:
        base_path = Path(base_dir)
        patterns_dir = base_path / "patterns"
        patterns_file = patterns_dir / "starter_patterns.json"
        db_path = patterns_dir / "patterns.db"
        
        # Create patterns directory if it doesn't exist
        os.makedirs(patterns_dir, exist_ok=True)
        
        # Create database if it doesn't exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS neural_patterns (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id   TEXT    NOT NULL,
            bug_type     TEXT    NOT NULL,
            tags         TEXT    NOT NULL,
            features     TEXT    NOT NULL,
            success_rate REAL    NOT NULL DEFAULT 0.0,
            sample_count INTEGER NOT NULL DEFAULT 0,
            created_at   REAL    NOT NULL,
            updated_at   REAL    NOT NULL
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS neural_weights (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_key   TEXT    NOT NULL,
            weight_value REAL    NOT NULL,
            category     TEXT    NOT NULL,
            description  TEXT,
            created_at   REAL    NOT NULL,
            updated_at   REAL    NOT NULL
        );
        """)
        
        # Create starter patterns
        starter_patterns = [
            {
                "pattern_id": "memory-leak-1",
                "bug_type": "memory-leak",
                "tags": ["memory", "leak", "resource"],
                "features": [
                    {"name": "tokens", "value": ["memory", "leak", "allocation", "free"], "weight": 1.0},
                    {"name": "solution_approach", "value": "Check resource allocation and deallocation", "weight": 0.8}
                ],
                "success_rate": 0.8,
                "sample_count": 10
            },
            {
                "pattern_id": "security-vuln-1",
                "bug_type": "security-vulnerability",
                "tags": ["security", "vulnerability", "injection"],
                "features": [
                    {"name": "tokens", "value": ["security", "vulnerability", "injection", "sql"], "weight": 1.0},
                    {"name": "solution_approach", "value": "Add input validation and parameterized queries", "weight": 0.9}
                ],
                "success_rate": 0.7,
                "sample_count": 5
            },
            {
                "pattern_id": "race-condition-1",
                "bug_type": "race-condition",
                "tags": ["race", "condition", "concurrent", "threading"],
                "features": [
                    {"name": "tokens", "value": ["race", "condition", "concurrent", "threading", "lock"], "weight": 1.0},
                    {"name": "solution_approach", "value": "Add proper synchronization mechanisms", "weight": 0.85}
                ],
                "success_rate": 0.65,
                "sample_count": 8
            }
        ]
        
        # Insert starter patterns into database
        now = time.time()
        for pattern in starter_patterns:
            cursor.execute(
                """
                INSERT INTO neural_patterns
                (pattern_id, bug_type, tags, features, success_rate, sample_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern["pattern_id"],
                    pattern["bug_type"],
                    json.dumps(pattern["tags"]),
                    json.dumps(pattern["features"]),
                    pattern["success_rate"],
                    pattern["sample_count"],
                    now,
                    now
                )
            )
        
        # Insert default weights into database
        weights_data = [
            ("observer", 1.0, "agent", "Observer agent weight", now, now),
            ("analyst", 1.2, "agent", "Analyst agent weight", now, now),
            ("verifier", 0.9, "agent", "Verifier agent weight", now, now),
            ("planner", 1.5, "agent", "Planner agent weight", now, now),
            ("pattern_match", 1.5, "feature", "Pattern matching weight", now, now),
            ("entropy", 0.8, "feature", "Entropy calculation weight", now, now),
            ("fallback", 0.5, "feature", "Fallback strategy weight", now, now),
            ("similarity", 1.2, "feature", "Similarity calculation weight", now, now)
        ]
        
        cursor.executemany(
            """
            INSERT INTO neural_weights
            (weight_key, weight_value, category, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            weights_data
        )
        
        conn.commit()
        conn.close()
        
        # Write starter patterns to file
        with open(patterns_file, 'w') as f:
            json.dump(starter_patterns, f, indent=2)
        
        logger.info(f"Initialized pattern database at {db_path}")
        
        return {
            "success": True,
            "patterns_file": str(patterns_file),
            "db_path": str(db_path),
            "patterns_count": len(starter_patterns)
        }
    except Exception as e:
        logger.error(f"Error initializing pattern database: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def initialize_neural_matrix(base_dir: str) -> Dict[str, Any]:
    """
    Initialize the neural matrix.
    
    Args:
        base_dir: Base directory for the neural matrix
        
    Returns:
        Dictionary with initialization result
    """
    # Create directory structure
    dir_result = create_directory_structure(base_dir)
    if not dir_result["success"]:
        return dir_result
    
    # Initialize weights
    weights_result = initialize_neural_weights(base_dir)
    if not weights_result["success"]:
        return weights_result
    
    # Initialize pattern database
    db_result = initialize_pattern_database(base_dir)
    if not db_result["success"]:
        return db_result
    
    logger.info(f"Neural matrix initialization complete at {base_dir}")
    
    return {
        "success": True,
        "base_dir": base_dir,
        "weights_file": weights_result["weights_file"],
        "patterns_file": db_result["patterns_file"],
        "db_path": db_result["db_path"]
    }

if __name__ == "__main__":
    # Initialize neural matrix in default location when run directly
    initialize_neural_matrix(".triangulum/neural_matrix")
