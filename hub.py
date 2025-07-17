#!/usr/bin/env python3
"""
hub.py
──────
Hub API for the Neural Matrix system.

This module provides a FastAPI-based API for interacting with the Neural Matrix,
including pattern access, similarity calculation, and weight management.
"""

import os
import json
import time
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural_matrix_hub")

# Default database path
DB_PATH = os.environ.get("NEURAL_MATRIX_DB", ".triangulum/neural_matrix/patterns/patterns.db")

# Create FastAPI app
app = FastAPI(
    title="Neural Matrix Hub API",
    description="API for accessing Neural Matrix patterns and weights",
    version="1.0.0"
)

# Database connection
conn = None
cur = None

# ───────────────────────────────────────────────────────────────────────────
# Pydantic models for request and response validation
# ───────────────────────────────────────────────────────────────────────────

class Feature(BaseModel):
    """Feature in a neural pattern."""
    name: str
    value: Union[List[str], str]
    weight: float = 1.0

class PatternCreate(BaseModel):
    """Request model for creating a pattern."""
    pattern_id: str
    bug_type: str
    tags: List[str]
    features: List[Feature]
    success_rate: float = 0.0
    sample_count: int = 0

class PatternResponse(BaseModel):
    """Response model for a pattern."""
    pattern_id: str
    bug_type: str
    tags: List[str]
    features: List[Dict[str, Any]]
    success_rate: float
    sample_count: int
    created_at: float
    updated_at: float

class SimilarityRequest(BaseModel):
    """Request model for similarity calculation."""
    bug_description: str
    tags: Optional[List[str]] = []
    feature_weights: Optional[Dict[str, float]] = {}

class WeightCreate(BaseModel):
    """Request model for creating a weight."""
    weight_key: str
    weight_value: float
    category: str
    description: Optional[str] = None

class WeightResponse(BaseModel):
    """Response model for a weight."""
    weight_key: str
    weight_value: float
    category: str
    description: Optional[str] = None
    created_at: float
    updated_at: float

# ───────────────────────────────────────────────────────────────────────────
# Database initialization
# ───────────────────────────────────────────────────────────────────────────

def init_db():
    """Initialize the database connection."""
    global conn, cur
    
    try:
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(DB_PATH)
        os.makedirs(db_dir, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Create tables if they don't exist
        cur.execute("""
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
        
        cur.execute("""
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
        
        conn.commit()
        logger.info(f"Initialized neural matrix database: {DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        if conn:
            conn.close()
            conn = None
            cur = None

# Initialize database on startup
init_db()

# Ensure database is initialized before handling requests
@app.on_event("startup")
async def startup_event():
    if not conn:
        init_db()

# Close database connection on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    if conn:
        conn.close()

# ───────────────────────────────────────────────────────────────────────────
# API endpoints
# ───────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Neural Matrix Hub API",
        "version": "1.0.0",
        "description": "API for accessing Neural Matrix patterns and weights"
    }

@app.post("/neural/patterns", status_code=201, response_model=Dict[str, Any])
async def create_pattern(pattern: PatternCreate):
    """Create a new neural pattern."""
    if not conn:
        init_db()
    
    try:
        # Check if pattern already exists
        cur.execute("SELECT pattern_id FROM neural_patterns WHERE pattern_id = ?", (pattern.pattern_id,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail=f"Pattern with ID {pattern.pattern_id} already exists")
        
        # Prepare data
        now = time.time()
        pattern_data = (
            pattern.pattern_id,
            pattern.bug_type,
            json.dumps(pattern.tags),
            json.dumps([feature.dict() for feature in pattern.features]),
            pattern.success_rate,
            pattern.sample_count,
            now,
            now
        )
        
        # Insert pattern
        cur.execute(
            """
            INSERT INTO neural_patterns
            (pattern_id, bug_type, tags, features, success_rate, sample_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pattern_data
        )
        
        conn.commit()
        
        return {
            "success": True,
            "pattern_id": pattern.pattern_id,
            "created_at": now
        }
    except Exception as e:
        logger.error(f"Error creating pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neural/patterns/{pattern_id}", response_model=PatternResponse)
async def get_pattern(pattern_id: str):
    """Get a pattern by ID."""
    if not conn:
        init_db()
    
    try:
        # Get pattern from database
        cur.execute("SELECT * FROM neural_patterns WHERE pattern_id = ?", (pattern_id,))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Pattern with ID {pattern_id} not found")
        
        # Convert row to dictionary
        pattern = dict(row)
        
        # Parse JSON fields
        pattern["tags"] = json.loads(pattern["tags"])
        pattern["features"] = json.loads(pattern["features"])
        
        return pattern
    except Exception as e:
        logger.error(f"Error getting pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neural/patterns", response_model=List[PatternResponse])
async def list_patterns(limit: int = 100, offset: int = 0):
    """List all patterns."""
    if not conn:
        init_db()
    
    try:
        # Get patterns from database
        cur.execute("SELECT * FROM neural_patterns LIMIT ? OFFSET ?", (limit, offset))
        rows = cur.fetchall()
        
        patterns = []
        for row in rows:
            # Convert row to dictionary
            pattern = dict(row)
            
            # Parse JSON fields
            pattern["tags"] = json.loads(pattern["tags"])
            pattern["features"] = json.loads(pattern["features"])
            
            patterns.append(pattern)
        
        return patterns
    except Exception as e:
        logger.error(f"Error listing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neural/similarity", response_model=List[Dict[str, Any]])
async def calculate_similarity(request: SimilarityRequest):
    """Calculate similarity between a bug description and all patterns."""
    if not conn:
        init_db()
    
    try:
        # Get all patterns
        cur.execute("SELECT * FROM neural_patterns")
        rows = cur.fetchall()
        
        if not rows:
            return []
        
        # Calculate similarity for each pattern
        similarities = []
        for row in rows:
            # Convert row to dictionary
            pattern = dict(row)
            
            # Parse JSON fields
            pattern["tags"] = json.loads(pattern["tags"])
            pattern["features"] = json.loads(pattern["features"])
            
            # Calculate similarity
            similarity = _calculate_pattern_similarity(pattern, request.dict())
            
            similarities.append({
                "pattern_id": pattern["pattern_id"],
                "bug_type": pattern["bug_type"],
                "similarity": similarity,
                "success_rate": pattern["success_rate"]
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neural/weights", status_code=201, response_model=Dict[str, Any])
async def create_weight(weight: WeightCreate):
    """Create a new neural weight."""
    if not conn:
        init_db()
    
    try:
        # Check if weight already exists
        cur.execute("SELECT weight_key FROM neural_weights WHERE weight_key = ? AND category = ?", 
                  (weight.weight_key, weight.category))
        if cur.fetchone():
            raise HTTPException(
                status_code=400, 
                detail=f"Weight with key {weight.weight_key} in category {weight.category} already exists"
            )
        
        # Prepare data
        now = time.time()
        weight_data = (
            weight.weight_key,
            weight.weight_value,
            weight.category,
            weight.description or "",
            now,
            now
        )
        
        # Insert weight
        cur.execute(
            """
            INSERT INTO neural_weights
            (weight_key, weight_value, category, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            weight_data
        )
        
        conn.commit()
        
        return {
            "success": True,
            "weight_key": weight.weight_key,
            "category": weight.category,
            "created_at": now
        }
    except Exception as e:
        logger.error(f"Error creating weight: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neural/weights", response_model=List[WeightResponse])
async def list_weights(category: Optional[str] = None, limit: int = 100, offset: int = 0):
    """List all weights, optionally filtered by category."""
    if not conn:
        init_db()
    
    try:
        # Prepare query based on whether category is provided
        if category:
            cur.execute("SELECT * FROM neural_weights WHERE category = ? LIMIT ? OFFSET ?", 
                      (category, limit, offset))
        else:
            cur.execute("SELECT * FROM neural_weights LIMIT ? OFFSET ?", (limit, offset))
        
        rows = cur.fetchall()
        
        weights = []
        for row in rows:
            # Convert row to dictionary
            weight = dict(row)
            weights.append(weight)
        
        return weights
    except Exception as e:
        logger.error(f"Error listing weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neural/weights/{weight_key}", response_model=WeightResponse)
async def get_weight(weight_key: str, category: Optional[str] = None):
    """Get a weight by key and optional category."""
    if not conn:
        init_db()
    
    try:
        # Prepare query based on whether category is provided
        if category:
            cur.execute("SELECT * FROM neural_weights WHERE weight_key = ? AND category = ?", (weight_key, category))
        else:
            cur.execute("SELECT * FROM neural_weights WHERE weight_key = ?", (weight_key,))
        
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Weight with key {weight_key} not found")
        
        # Convert row to dictionary
        weight = dict(row)
        
        return weight
    except Exception as e:
        logger.error(f"Error getting weight: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/neural/weights/{weight_key}", response_model=Dict[str, Any])
async def update_weight(weight_key: str, weight_value: float, category: Optional[str] = None):
    """Update a neural weight."""
    if not conn:
        init_db()
    
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
        cur.execute(query, params)
        conn.commit()
        
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Weight with key {weight_key} not found")
        
        return {
            "success": True,
            "weight_key": weight_key,
            "updated": True,
            "updated_at": now
        }
    except Exception as e:
        logger.error(f"Error updating weight: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/neural/patterns/{pattern_id}/success_rate", response_model=Dict[str, Any])
async def update_pattern_success_rate(pattern_id: str, success: bool):
    """Update the success rate of a pattern based on a new result."""
    if not conn:
        init_db()
    
    try:
        # Get current pattern
        cur.execute("SELECT success_rate, sample_count FROM neural_patterns WHERE pattern_id = ?", (pattern_id,))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Pattern with ID {pattern_id} not found")
        
        # Calculate new success rate
        current_rate = row["success_rate"]
        current_count = row["sample_count"]
        
        new_count = current_count + 1
        new_rate = ((current_rate * current_count) + (1 if success else 0)) / new_count
        
        # Update pattern
        now = time.time()
        cur.execute(
            """
            UPDATE neural_patterns 
            SET success_rate = ?, sample_count = ?, updated_at = ? 
            WHERE pattern_id = ?
            """,
            (new_rate, new_count, now, pattern_id)
        )
        
        conn.commit()
        
        return {
            "success": True,
            "pattern_id": pattern_id,
            "old_rate": current_rate,
            "new_rate": new_rate,
            "sample_count": new_count,
            "updated_at": now
        }
    except Exception as e:
        logger.error(f"Error updating pattern success rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neural/health")
async def health_check():
    """Health check endpoint."""
    if not conn:
        init_db()
    
    try:
        # Check if database is accessible
        cur.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# ───────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────

def _calculate_pattern_similarity(pattern: Dict[str, Any], request: Dict[str, Any]) -> float:
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
            if isinstance(pattern_context, str) and pattern_context.lower() in request_description.lower():
                context_similarity = feature["weight"]
            
            # Apply feature weight if provided
            if "context" in request_feature_weights:
                context_similarity *= request_feature_weights["context"]
            
            break
    
    # Calculate overall similarity
    similarity = (tag_similarity * 0.3) + (token_similarity * 0.5) + (context_similarity * 0.2)
    
    return min(similarity, 1.0)

# ───────────────────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    # Run API server
    uvicorn.run(app, host="0.0.0.0", port=8001)
