#!/usr/bin/env python3
"""
neural_matrix_init.py
────────────────────
Neural Matrix Initialization Script

This script initializes the neural matrix system by creating the necessary directory
structure and default configuration files. It sets up:

1. Neural matrix directory structure
2. Default neural weights
3. Initial pattern database
4. Family tree neural connections
5. Test data for verification

Usage:
    python neural_matrix_init.py [--force]

Arguments:
    --force  Overwrite existing files (use with caution)
"""

import argparse
import json
import os
import sys
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import hub  # Neural connection for communication center
import verification_engine  # Neural connection for invariant validation


# Neural matrix directory constants
NEURAL_MATRIX_DIR = Path(".triangulum/neural_matrix")
NEURAL_PATTERNS_DIR = NEURAL_MATRIX_DIR / "patterns"
NEURAL_WEIGHTS_DIR = NEURAL_MATRIX_DIR / "weights"
NEURAL_HISTORY_DIR = NEURAL_MATRIX_DIR / "history"
NEURAL_CONNECTIONS_DIR = NEURAL_MATRIX_DIR / "connections"
NEURAL_TEST_DATA_DIR = NEURAL_MATRIX_DIR / "test_data"

def create_directory_structure(base_dir: Path, force: bool = False) -> None:
    """
    Create the neural matrix directory structure.
    
    Args:
        base_dir: Base directory for neural matrix
        force: Whether to overwrite existing files
    """
    print(f"Creating neural matrix directory structure in {base_dir}")
    
    # Create main directories
    directories = [
        base_dir,
        base_dir / "patterns",
        base_dir / "weights",
        base_dir / "history",
        base_dir / "connections",
        base_dir / "test_data"
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


def initialize_neural_weights(base_dir: Path, force: bool = False) -> None:
    """
    Initialize default neural weights.
    
    Args:
        base_dir: Base directory for neural matrix
        force: Whether to overwrite existing files
    """
    weights_file = base_dir / "weights" / "default_weights.json"
    
    if weights_file.exists() and not force:
        print(f"Weights file already exists: {weights_file}")
        return
    
    # Default neural weights
    weights = {
        "agent_weights": {
            "observer": 1.0,
            "analyst": 1.0,
            "verifier": 1.0,
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
    
    # Save weights to file
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"Initialized neural weights: {weights_file}")


def initialize_pattern_database(base_dir: Path, force: bool = False) -> None:
    """
    Initialize pattern database with starter patterns.
    
    Args:
        base_dir: Base directory for neural matrix
        force: Whether to overwrite existing files
    """
    patterns_file = base_dir / "patterns" / "starter_patterns.json"
    
    if patterns_file.exists() and not force:
        print(f"Patterns file already exists: {patterns_file}")
        return
    
    # Starter patterns for common bug types
    patterns = [
        {
            "pattern_id": "memory-leak-pattern",
            "bug_type": "memory-leak",
            "tags": ["memory", "performance", "leak"],
            "features": [
                {"name": "tokens", "value": ["memory", "leak", "allocation", "free"], "weight": 1.0},
                {"name": "complexity", "value": 0.7, "weight": 0.8}
            ],
            "success_rate": 0.8,
            "sample_count": 10,
            "created_at": time.time()
        },
        {
            "pattern_id": "null-pointer-pattern",
            "bug_type": "null-pointer",
            "tags": ["pointer", "crash", "null"],
            "features": [
                {"name": "tokens", "value": ["null", "pointer", "dereference", "crash"], "weight": 1.0},
                {"name": "complexity", "value": 0.5, "weight": 0.6}
            ],
            "success_rate": 0.9,
            "sample_count": 15,
            "created_at": time.time()
        },
        {
            "pattern_id": "race-condition-pattern",
            "bug_type": "race-condition",
            "tags": ["concurrency", "threading", "synchronization"],
            "features": [
                {"name": "tokens", "value": ["race", "condition", "thread", "lock", "mutex"], "weight": 1.0},
                {"name": "complexity", "value": 0.8, "weight": 0.9}
            ],
            "success_rate": 0.7,
            "sample_count": 8,
            "created_at": time.time()
        }
    ]
    
    # Save patterns to file
    with open(patterns_file, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"Initialized pattern database: {patterns_file}")


def initialize_family_tree_connections(base_dir: Path, force: bool = False) -> None:
    """
    Initialize family tree with neural connections.
    
    Args:
        base_dir: Base directory for neural matrix
        force: Whether to overwrite existing files
    """
    family_tree_path = Path(".triangulum/family_tree.json")
    
    # Check if family tree exists
    if not family_tree_path.exists():
        # Create basic family tree structure
        tree_data = {
            "relationships": {
                "planner": {
                    "children": ["observer-1", "analyst-1", "verifier-1"],
                    "metadata": {
                        "type": "root",
                        "created_at": time.time()
                    }
                },
                "observer-1": {
                    "parent": "planner",
                    "children": [],
                    "metadata": {
                        "type": "observer",
                        "created_at": time.time()
                    }
                },
                "analyst-1": {
                    "parent": "planner",
                    "children": [],
                    "metadata": {
                        "type": "analyst",
                        "created_at": time.time()
                    }
                },
                "verifier-1": {
                    "parent": "planner",
                    "children": [],
                    "metadata": {
                        "type": "verifier",
                        "created_at": time.time()
                    }
                }
            },
            "updated_at": time.time()
        }
        
        # Create directory if it doesn't exist
        family_tree_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the initial family tree
        with open(family_tree_path, 'w') as f:
            json.dump(tree_data, f, indent=2)
            
        print(f"Created initial family tree: {family_tree_path}")
    else:
        # Load existing family tree
        with open(family_tree_path, 'r') as f:
            tree_data = json.load(f)
        
        print(f"Using existing family tree: {family_tree_path}")
    
    # Check if neural connections already exist
    if "neural_connections" in tree_data and not force:
        print("Neural connections already exist in family tree")
        return
    
    # Add neural connections to family tree
    tree_data["neural_connections"] = {
        "planner": ["observer-1", "analyst-1", "verifier-1"],
        "observer-1": ["analyst-1"],
        "analyst-1": ["verifier-1"],
        "verifier-1": ["planner"]
    }
    
    # Update timestamp
    tree_data["updated_at"] = time.time()
    
    # Save updated family tree
    with open(family_tree_path, 'w') as f:
        json.dump(tree_data, f, indent=2)
    
    print("Added neural connections to family tree")
    
    # Create connection visualization data
    connections_file = base_dir / "connections" / "neural_graph.json"
    
    # Generate more detailed connection graph
    connection_graph = {
        "nodes": [
            {"id": "planner", "type": "root", "connections": 3},
            {"id": "observer-1", "type": "observer", "connections": 1},
            {"id": "analyst-1", "type": "analyst", "connections": 1},
            {"id": "verifier-1", "type": "verifier", "connections": 1}
        ],
        "edges": [
            {"source": "planner", "target": "observer-1", "weight": 1.0, "type": "bidirectional"},
            {"source": "planner", "target": "analyst-1", "weight": 1.0, "type": "bidirectional"},
            {"source": "planner", "target": "verifier-1", "weight": 1.0, "type": "bidirectional"},
            {"source": "observer-1", "target": "analyst-1", "weight": 0.8, "type": "forward"},
            {"source": "analyst-1", "target": "verifier-1", "weight": 0.8, "type": "forward"},
            {"source": "verifier-1", "target": "planner", "weight": 0.8, "type": "feedback"}
        ],
        "metadata": {
            "created_at": time.time(),
            "description": "Neural matrix connection graph for visualization"
        }
    }
    
    # Save connection graph
    with open(connections_file, 'w') as f:
        json.dump(connection_graph, f, indent=2)
    
    print(f"Created neural connection graph: {connections_file}")


def initialize_database_tables(db_path: Path, force: bool = False) -> None:
    """
    Initialize database tables for neural matrix.
    
    Args:
        db_path: Path to SQLite database
        force: Whether to drop and recreate existing tables
    """
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Create neural pattern table
        if force:
            conn.execute("DROP TABLE IF EXISTS neural_patterns")
            conn.execute("DROP TABLE IF EXISTS neural_weights")
            conn.execute("DROP TABLE IF EXISTS solution_similarities")
            
        # Create neural patterns table
        conn.execute("""
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
        )
        """)
        
        # Create neural weights table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS neural_weights (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_key   TEXT    NOT NULL,
            weight_value REAL    NOT NULL,
            category     TEXT    NOT NULL,
            description  TEXT,
            created_at   REAL    NOT NULL,
            updated_at   REAL    NOT NULL
        )
        """)
        
        # Create solution similarities table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS solution_similarities (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            solution_a   INTEGER NOT NULL,
            solution_b   INTEGER NOT NULL,
            similarity   REAL    NOT NULL,
            created_at   REAL    NOT NULL
        )
        """)
        
        # Insert sample patterns
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM neural_patterns")
        pattern_count = cursor.fetchone()[0]
        
        if pattern_count == 0 or force:
            patterns = [
                {
                    "pattern_id": "memory-leak-pattern",
                    "bug_type": "memory-leak",
                    "tags": json.dumps(["memory", "performance", "leak"]),
                    "features": json.dumps([
                        {"name": "tokens", "value": ["memory", "leak", "allocation", "free"], "weight": 1.0},
                        {"name": "complexity", "value": 0.7, "weight": 0.8}
                    ]),
                    "success_rate": 0.8,
                    "sample_count": 10
                },
                {
                    "pattern_id": "null-pointer-pattern",
                    "bug_type": "null-pointer",
                    "tags": json.dumps(["pointer", "crash", "null"]),
                    "features": json.dumps([
                        {"name": "tokens", "value": ["null", "pointer", "dereference", "crash"], "weight": 1.0},
                        {"name": "complexity", "value": 0.5, "weight": 0.6}
                    ]),
                    "success_rate": 0.9,
                    "sample_count": 15
                },
                {
                    "pattern_id": "race-condition-pattern",
                    "bug_type": "race-condition",
                    "tags": json.dumps(["concurrency", "threading", "synchronization"]),
                    "features": json.dumps([
                        {"name": "tokens", "value": ["race", "condition", "thread", "lock", "mutex"], "weight": 1.0},
                        {"name": "complexity", "value": 0.8, "weight": 0.9}
                    ]),
                    "success_rate": 0.7,
                    "sample_count": 8
                }
            ]
            
            now = time.time()
            for pattern in patterns:
                conn.execute(
                    """
                    INSERT INTO neural_patterns
                    (pattern_id, bug_type, tags, features, success_rate, sample_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pattern["pattern_id"],
                        pattern["bug_type"],
                        pattern["tags"],
                        pattern["features"],
                        pattern["success_rate"],
                        pattern["sample_count"],
                        now,
                        now
                    )
                )
            
            print(f"Inserted {len(patterns)} sample patterns into database")
        
        # Insert sample weights
        cursor.execute("SELECT COUNT(*) FROM neural_weights")
        weight_count = cursor.fetchone()[0]
        
        if weight_count == 0 or force:
            weights = [
                {"weight_key": "observer", "weight_value": 1.0, "category": "agent", "description": "Observer agent weight"},
                {"weight_key": "analyst", "weight_value": 1.2, "category": "agent", "description": "Analyst agent weight"},
                {"weight_key": "verifier", "weight_value": 0.9, "category": "agent", "description": "Verifier agent weight"},
                {"weight_key": "planner", "weight_value": 1.5, "category": "agent", "description": "Planner agent weight"},
                {"weight_key": "pattern_match", "weight_value": 1.5, "category": "feature", "description": "Pattern matching weight"},
                {"weight_key": "entropy", "weight_value": 0.8, "category": "feature", "description": "Entropy-based weight"},
                {"weight_key": "learning_rate", "weight_value": 0.1, "category": "parameter", "description": "Neural learning rate"}
            ]
            
            now = time.time()
            for weight in weights:
                conn.execute(
                    """
                    INSERT INTO neural_weights
                    (weight_key, weight_value, category, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        weight["weight_key"],
                        weight["weight_value"],
                        weight["category"],
                        weight["description"],
                        now,
                        now
                    )
                )
            
            print(f"Inserted {len(weights)} sample weights into database")
        
        # Commit changes
        conn.commit()
        print(f"Initialized database tables in {db_path}")
        
    finally:
        conn.close()


def create_test_data(base_dir: Path, force: bool = False) -> None:
    """
    Create test data for neural matrix verification.
    
    Args:
        base_dir: Base directory for neural matrix
        force: Whether to overwrite existing files
    """
    test_data_dir = base_dir / "test_data"
    test_data_file = test_data_dir / "test_patterns.json"
    
    if test_data_file.exists() and not force:
        print(f"Test data file already exists: {test_data_file}")
        return
    
    # Create test patterns with deliberate issues for verification testing
    test_data = {
        "valid_patterns": [
            {
                "pattern_id": "test-valid-1",
                "bug_type": "test-bug",
                "tags": ["test", "valid"],
                "features": [
                    {"name": "tokens", "value": ["test", "valid", "pattern"], "weight": 1.0}
                ],
                "success_rate": 0.5,
                "sample_count": 5
            }
        ],
        "invalid_patterns": [
            {
                "pattern_id": "test-invalid-1",
                "bug_type": "test-bug",
                "tags": ["test", "invalid"],
                "features": [
                    {"name": "tokens", "value": ["test", "invalid", "pattern"], "weight": 10.0}  # Invalid weight
                ],
                "success_rate": 1.5,  # Invalid success rate
                "sample_count": 5
            }
        ],
        "valid_connections": {
            "test-agent-1": ["test-agent-2", "test-agent-3"],
            "test-agent-2": ["test-agent-3"],
            "test-agent-3": ["test-agent-1"]
        },
        "invalid_connections": {
            "test-agent-1": ["test-agent-2"],
            "test-agent-2": ["test-agent-3"],
            "test-agent-3": ["test-agent-4"],  # Non-existent agent
            "test-agent-4": ["test-agent-1"]
        },
        "created_at": time.time()
    }
    
    # Save test data to file
    with open(test_data_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test data: {test_data_file}")


def create_readme(base_dir: Path) -> None:
    """
    Create README.md file for neural matrix directory.
    
    Args:
        base_dir: Base directory for neural matrix
    """
    readme_file = base_dir / "README.md"
    
    readme_content = """# Neural Matrix

## Overview

The Neural Matrix is the core learning and pattern recognition system for FixWurx. It enables:

1. Pattern-based bug recognition
2. Learned solution recommendations
3. Neural connectivity between agent components
4. Adaptive learning from past fixes
5. Weight-based optimization

## Directory Structure

- `/patterns/` - Neural patterns for bug recognition
- `/weights/` - Neural weights for learning and adaptation
- `/history/` - Historical solution data for learning
- `/connections/` - Neural connectivity definitions
- `/test_data/` - Test data for verification

## Database Integration

The Neural Matrix stores pattern recognition data and neural weights in the SQLite database
located at `.triangulum/reviews.sqlite`. This enables persistent learning across sessions.

## Usage

The Neural Matrix is integrated with the following components:

1. `planner_agent.py` - Uses neural learning for solution path generation
2. `hub.py` - Provides API endpoints for neural pattern access
3. `verification_engine.py` - Validates neural matrix integrity
4. `scope_filter.py` - Uses neural patterns for file relevance detection

## Visualization

Neural connections can be visualized in the dashboard through the neural graph data
stored in `connections/neural_graph.json`.
"""
    
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README.md: {readme_file}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Initialize neural matrix system")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    args = parser.parse_args()
    
    base_dir = Path(".triangulum/neural_matrix")
    db_path = Path(".triangulum/reviews.sqlite")
    
    try:
        # Initialize neural matrix components
        create_directory_structure(base_dir, args.force)
        initialize_neural_weights(base_dir, args.force)
        initialize_pattern_database(base_dir, args.force)
        initialize_family_tree_connections(base_dir, args.force)
        initialize_database_tables(db_path, args.force)
        create_test_data(base_dir, args.force)
        create_readme(base_dir)
        
        print("\nNeural matrix initialization complete!")
        print(f"Base directory: {base_dir.resolve()}")
        print(f"Database: {db_path.resolve()}")
        print("\nTo test the neural matrix, run the verification tests:")
        print("python test_verification.py")
        
    except Exception as e:
        print(f"Error initializing neural matrix: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
