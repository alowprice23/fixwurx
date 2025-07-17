#!/usr/bin/env python3
"""
neural_matrix.py
──────────────
Core implementation of the Neural Matrix.

This module provides the NeuralMatrix class, which is responsible for
pattern recognition, similarity calculation, and learning.
"""

import os
import json
import time
import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural_matrix")

class NeuralMatrix:
    """
    Neural matrix for pattern recognition and learning.
    
    The NeuralMatrix handles pattern recognition, similarity calculation,
    and learning from past solutions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the neural matrix.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("neural_matrix")
        
        # Configuration options
        self.base_dir = Path(self.config.get("base_dir", ".triangulum/neural_matrix"))
        self.db_path = self.config.get("db_path", str(self.base_dir / "patterns" / "patterns.db"))
        
        # Connection to database
        self.conn = None
        self.cur = None
        
        # Pattern and weight caches
        self.pattern_cache = {}
        self.weight_cache = {}
        self.last_update_time = time.time()
        
        # Initialize database connection
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize database connection."""
        try:
            # Create database directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            os.makedirs(db_dir, exist_ok=True)
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.cur = self.conn.cursor()
            
            # Create tables if they don't exist
            self.cur.execute("""
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
            
            self.cur.execute("""
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
            
            self.conn.commit()
            self.logger.info(f"Initialized neural matrix database: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error initializing neural matrix database: {e}")
            # Continue without database
            self.conn = None
            self.cur = None
    
    def create_pattern(self, pattern_id: str, bug_type: str, tags: List[str], 
                      features: List[Dict[str, Any]], success_rate: float = 0.0, 
                      sample_count: int = 0) -> Dict[str, Any]:
        """
        Create a new neural pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            bug_type: Type of bug this pattern represents
            tags: List of tags for the pattern
            features: List of features for the pattern
            success_rate: Initial success rate (0.0 to 1.0)
            sample_count: Initial sample count
            
        Returns:
            Dictionary with creation result
        """
        if self.conn is None:
            return {"success": False, "error": "Database not initialized"}
        
        try:
            # Check if pattern already exists
            self.cur.execute("SELECT pattern_id FROM neural_patterns WHERE pattern_id = ?", (pattern_id,))
            if self.cur.fetchone():
                return {"success": False, "error": f"Pattern with ID {pattern_id} already exists"}
            
            # Prepare data
            now = time.time()
            pattern_data = (
                pattern_id,
                bug_type,
                json.dumps(tags),
                json.dumps(features),
                success_rate,
                sample_count,
                now,
                now
            )
            
            # Insert pattern
            self.cur.execute(
                """
                INSERT INTO neural_patterns
                (pattern_id, bug_type, tags, features, success_rate, sample_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                pattern_data
            )
            
            self.conn.commit()
            
            # Add to cache
            self.pattern_cache[pattern_id] = {
                "pattern_id": pattern_id,
                "bug_type": bug_type,
                "tags": tags,
                "features": features,
                "success_rate": success_rate,
                "sample_count": sample_count,
                "created_at": now,
                "updated_at": now
            }
            
            return {
                "success": True,
                "pattern_id": pattern_id,
                "created_at": now
            }
        except Exception as e:
            self.logger.error(f"Error creating neural pattern: {e}")
            return {"success": False, "error": str(e)}
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: The pattern ID to look for
            
        Returns:
            Pattern dictionary if found, None otherwise
        """
        # Check cache first
        if pattern_id in self.pattern_cache:
            return self.pattern_cache[pattern_id]
        
        if self.conn is None:
            return None
        
        try:
            # Get pattern from database
            self.cur.execute("SELECT * FROM neural_patterns WHERE pattern_id = ?", (pattern_id,))
            row = self.cur.fetchone()
            
            if row is None:
                return None
            
            # Convert row to dictionary
            pattern = dict(row)
            
            # Parse JSON fields
            if "tags" in pattern and pattern["tags"]:
                pattern["tags"] = json.loads(pattern["tags"])
            else:
                pattern["tags"] = []
            
            if "features" in pattern and pattern["features"]:
                pattern["features"] = json.loads(pattern["features"])
            else:
                pattern["features"] = []
            
            # Add to cache
            self.pattern_cache[pattern_id] = pattern
            
            return pattern
        except Exception as e:
            self.logger.error(f"Error getting neural pattern: {e}")
            return None
    
    def list_patterns(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all patterns.
        
        Args:
            limit: Maximum number of patterns to return
            offset: Offset for pagination
            
        Returns:
            List of pattern dictionaries
        """
        if self.conn is None:
            return []
        
        try:
            # Get patterns from database
            self.cur.execute("SELECT * FROM neural_patterns LIMIT ? OFFSET ?", (limit, offset))
            rows = self.cur.fetchall()
            
            patterns = []
            for row in rows:
                # Convert row to dictionary
                pattern = dict(row)
                
                # Parse JSON fields
                if "tags" in pattern and pattern["tags"]:
                    pattern["tags"] = json.loads(pattern["tags"])
                else:
                    pattern["tags"] = []
                
                if "features" in pattern and pattern["features"]:
                    pattern["features"] = json.loads(pattern["features"])
                else:
                    pattern["features"] = []
                
                # Add to cache and result
                self.pattern_cache[pattern["pattern_id"]] = pattern
                patterns.append(pattern)
            
            return patterns
        except Exception as e:
            self.logger.error(f"Error listing neural patterns: {e}")
            return []
    
    def create_weight(self, weight_key: str, weight_value: float, category: str, 
                     description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new neural weight.
        
        Args:
            weight_key: Key for the weight
            weight_value: Value for the weight
            category: Category for the weight
            description: Optional description
            
        Returns:
            Dictionary with creation result
        """
        if self.conn is None:
            return {"success": False, "error": "Database not initialized"}
        
        try:
            # Check if weight already exists
            self.cur.execute("SELECT weight_key FROM neural_weights WHERE weight_key = ? AND category = ?", 
                           (weight_key, category))
            if self.cur.fetchone():
                return {"success": False, "error": f"Weight with key {weight_key} in category {category} already exists"}
            
            # Prepare data
            now = time.time()
            weight_data = (
                weight_key,
                weight_value,
                category,
                description or "",
                now,
                now
            )
            
            # Insert weight
            self.cur.execute(
                """
                INSERT INTO neural_weights
                (weight_key, weight_value, category, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                weight_data
            )
            
            self.conn.commit()
            
            # Add to cache
            cache_key = f"{category}:{weight_key}"
            self.weight_cache[cache_key] = weight_value
            
            return {
                "success": True,
                "weight_key": weight_key,
                "category": category,
                "created_at": now
            }
        except Exception as e:
            self.logger.error(f"Error creating neural weight: {e}")
            return {"success": False, "error": str(e)}
    
    def update_weight(self, weight_key: str, weight_value: float, 
                     category: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a neural weight.
        
        Args:
            weight_key: Key for the weight
            weight_value: New value for the weight
            category: Optional category for the weight
            
        Returns:
            Dictionary with update result
        """
        if self.conn is None:
            return {"success": False, "error": "Database not initialized"}
        
        try:
            # Prepare query based on whether category is provided
            now = time.time()
            if category:
                query = "UPDATE neural_weights SET weight_value = ?, updated_at = ? WHERE weight_key = ? AND category = ?"
                params = (weight_value, now, weight_key, category)
            else:
                query = "UPDATE neural_weights SET weight_value = ?, updated_at = ? WHERE weight_key = ?"
                params = (weight_value, now, weight_key)
            
            # Execute update
            self.cur.execute(query, params)
            self.conn.commit()
            
            if self.cur.rowcount == 0:
                return {"success": False, "error": f"Weight with key {weight_key} not found"}
            
            # Update cache
            if category:
                cache_key = f"{category}:{weight_key}"
                self.weight_cache[cache_key] = weight_value
            else:
                # Update all weights with this key
                for key in list(self.weight_cache.keys()):
                    if key.endswith(f":{weight_key}"):
                        self.weight_cache[key] = weight_value
            
            return {
                "success": True,
                "weight_key": weight_key,
                "updated": True,
                "updated_at": now
            }
        except Exception as e:
            self.logger.error(f"Error updating neural weight: {e}")
            return {"success": False, "error": str(e)}
    
    def get_weight(self, weight_key: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a weight by key and optional category.
        
        Args:
            weight_key: The weight key to look for
            category: Optional category to filter by
            
        Returns:
            Weight dictionary if found, None otherwise
        """
        # Check cache first
        if category:
            cache_key = f"{category}:{weight_key}"
            if cache_key in self.weight_cache:
                return {
                    "weight_key": weight_key,
                    "weight_value": self.weight_cache[cache_key],
                    "category": category
                }
        
        if self.conn is None:
            return None
        
        try:
            # Prepare query based on whether category is provided
            if category:
                self.cur.execute("SELECT * FROM neural_weights WHERE weight_key = ? AND category = ?", (weight_key, category))
            else:
                self.cur.execute("SELECT * FROM neural_weights WHERE weight_key = ?", (weight_key,))
            
            row = self.cur.fetchone()
            
            if row is None:
                return None
            
            # Convert row to dictionary
            weight = dict(row)
            
            # Add to cache
            cache_key = f"{weight['category']}:{weight['weight_key']}"
            self.weight_cache[cache_key] = weight["weight_value"]
            
            return weight
        except Exception as e:
            self.logger.error(f"Error getting neural weight: {e}")
            return None
    
    def list_weights(self, category: Optional[str] = None, limit: int = 100, 
                    offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all weights, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            limit: Maximum number of weights to return
            offset: Offset for pagination
            
        Returns:
            List of weight dictionaries
        """
        if self.conn is None:
            return []
        
        try:
            # Prepare query based on whether category is provided
            if category:
                self.cur.execute("SELECT * FROM neural_weights WHERE category = ? LIMIT ? OFFSET ?", 
                               (category, limit, offset))
            else:
                self.cur.execute("SELECT * FROM neural_weights LIMIT ? OFFSET ?", (limit, offset))
            
            rows = self.cur.fetchall()
            
            weights = []
            for row in rows:
                # Convert row to dictionary
                weight = dict(row)
                
                # Add to cache and result
                cache_key = f"{weight['category']}:{weight['weight_key']}"
                self.weight_cache[cache_key] = weight["weight_value"]
                weights.append(weight)
            
            return weights
        except Exception as e:
            self.logger.error(f"Error listing neural weights: {e}")
            return []
    
    def calculate_similarity(self, bug_description: str, tags: Optional[List[str]] = None,
                            feature_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Calculate similarity between a bug description and all patterns.
        
        Args:
            bug_description: Description of the bug
            tags: Optional list of tags
            feature_weights: Optional dictionary of feature weights
            
        Returns:
            List of pattern similarities
        """
        # Get all patterns
        patterns = self.list_patterns(limit=1000)
        
        if not patterns:
            return []
        
        # Prepare request dictionary
        request = {
            "bug_description": bug_description,
            "tags": tags or [],
            "feature_weights": feature_weights or {}
        }
        
        # Calculate similarity for each pattern
        similarities = []
        for pattern in patterns:
            similarity = self._calculate_pattern_similarity(pattern, request)
            
            similarities.append({
                "pattern_id": pattern["pattern_id"],
                "bug_type": pattern["bug_type"],
                "similarity": similarity,
                "success_rate": pattern["success_rate"]
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities
    
    def _calculate_pattern_similarity(self, pattern: Dict[str, Any], request: Dict[str, Any]) -> float:
        """
        Calculate similarity between a pattern and a request.
        
        Args:
            pattern: Pattern dictionary
            request: Request dictionary
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Extract features from pattern
        pattern_features = pattern.get("features", [])
        pattern_tags = pattern.get("tags", [])
        
        # Extract features from request
        request_description = request.get("bug_description", "")
        request_tags = request.get("tags", [])
        request_feature_weights = request.get("feature_weights", {})
        
        # Calculate tag similarity
        tag_similarity = 0.0
        if pattern_tags and request_tags:
            common_tags = set(pattern_tags) & set(request_tags)
            tag_similarity = len(common_tags) / max(len(pattern_tags), len(request_tags))
        
        # Calculate token similarity
        token_similarity = 0.0
        for feature in pattern_features:
            if feature["name"] == "tokens":
                # Get tokens from pattern
                pattern_tokens = feature["value"]
                
                # Check tokens in request description
                token_matches = 0
                for token in pattern_tokens:
                    if token.lower() in request_description.lower():
                        token_matches += 1
                
                if pattern_tokens:
                    token_similarity = token_matches / len(pattern_tokens)
                
                # Apply feature weight if provided
                if "tokens" in request_feature_weights:
                    token_similarity *= request_feature_weights["tokens"]
                
                break
        
        # Calculate context similarity
        context_similarity = 0.0
        for feature in pattern_features:
            if feature["name"] == "context":
                # Get context from pattern
                pattern_context = feature["value"]
                
                # Check if context is in request description
                if pattern_context.lower() in request_description.lower():
                    context_similarity = feature["weight"]
                
                # Apply feature weight if provided
                if "context" in request_feature_weights:
                    context_similarity *= request_feature_weights["context"]
                
                break
        
        # Calculate overall similarity
        similarity = (tag_similarity * 0.3) + (token_similarity * 0.5) + (context_similarity * 0.2)
        
        return min(similarity, 1.0)
    
    def update_pattern_success_rate(self, pattern_id: str, success: bool) -> Dict[str, Any]:
        """
        Update the success rate of a pattern based on a new result.
        
        Args:
            pattern_id: ID of the pattern to update
            success: Whether the pattern was successful
            
        Returns:
            Dictionary with update result
        """
        if self.conn is None:
            return {"success": False, "error": "Database not initialized"}
        
        try:
            # Get current pattern
            self.cur.execute("SELECT success_rate, sample_count FROM neural_patterns WHERE pattern_id = ?", (pattern_id,))
            row = self.cur.fetchone()
            
            if row is None:
                return {"success": False, "error": f"Pattern with ID {pattern_id} not found"}
            
            # Calculate new success rate
            current_rate = row["success_rate"]
            current_count = row["sample_count"]
            
            new_count = current_count + 1
            new_rate = ((current_rate * current_count) + (1 if success else 0)) / new_count
            
            # Update pattern
            now = time.time()
            self.cur.execute(
                """
                UPDATE neural_patterns 
                SET success_rate = ?, sample_count = ?, updated_at = ? 
                WHERE pattern_id = ?
                """,
                (new_rate, new_count, now, pattern_id)
            )
            
            self.conn.commit()
            
            # Update cache if pattern is cached
            if pattern_id in self.pattern_cache:
                self.pattern_cache[pattern_id]["success_rate"] = new_rate
                self.pattern_cache[pattern_id]["sample_count"] = new_count
                self.pattern_cache[pattern_id]["updated_at"] = now
            
            return {
                "success": True,
                "pattern_id": pattern_id,
                "old_rate": current_rate,
                "new_rate": new_rate,
                "sample_count": new_count,
                "updated_at": now
            }
        except Exception as e:
            self.logger.error(f"Error updating pattern success rate: {e}")
            return {"success": False, "error": str(e)}
    
    def recommend_solutions(self, bug_description: str, tags: Optional[List[str]] = None,
                          feature_weights: Optional[Dict[str, float]] = None, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend solutions based on pattern similarity.
        
        Args:
            bug_description: Description of the bug
            tags: Optional list of tags
            feature_weights: Optional dictionary of feature weights
            limit: Maximum number of recommendations to return
            
        Returns:
            List of solution recommendations
        """
        # Calculate similarities
        similarities = self.calculate_similarity(bug_description, tags, feature_weights)
        
        # Take top matches
        top_matches = similarities[:limit]
        
        # Enhance with solution recommendations
        recommendations = []
        for match in top_matches:
            pattern_id = match["pattern_id"]
            
            # Get pattern details
            pattern = self.get_pattern(pattern_id)
            if not pattern:
                continue
            
            # Extract solution approach from features
            solution_approach = "Unknown approach"
            for feature in pattern.get("features", []):
                if feature["name"] == "solution_approach":
                    solution_approach = feature["value"]
                    break
            
            recommendations.append({
                "pattern_id": pattern_id,
                "bug_type": pattern["bug_type"],
                "similarity": match["similarity"],
                "success_rate": pattern["success_rate"],
                "solution_approach": solution_approach,
                "confidence": match["similarity"] * pattern["success_rate"]
            })
        
        return recommendations
    
    def extract_features(self, text: str, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract features from text and tags.
        
        Args:
            text: Text to extract features from
            tags: Optional list of tags
            
        Returns:
            List of features
        """
        features = []
        
        # Extract tokens from text
        tokens = text.lower().split()
        tokens = [token for token in tokens if len(token) > 3]  # Filter short tokens
        tokens = list(set(tokens))  # Remove duplicates
        
        if tokens:
            features.append({
                "name": "tokens",
                "value": tokens,
                "weight": 1.0
            })
        
        # Use tags for context
        if tags:
            context = " ".join(tags)
            features.append({
                "name": "context",
                "value": context,
                "weight": 0.8
            })
        
        return features
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cur = None
