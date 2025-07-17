#!/usr/bin/env python3
"""
integration_module.py
───────────────────
Integration module for connecting the Neural Matrix with other system components.

This module provides the NeuralMatrixIntegration class, which serves as a bridge
between the Neural Matrix and other components like the triangulation engine.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from neural_matrix.core.neural_matrix import NeuralMatrix
from triangulation_engine import TriangulationEngine, BugState
from verification_engine import VerificationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural_matrix_integration")

class NeuralMatrixIntegration:
    """
    Integration class for connecting the Neural Matrix with other system components.
    
    This class serves as a bridge between the Neural Matrix and components like
    the triangulation engine, enabling pattern recognition and learning.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the neural matrix integration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("neural_matrix_integration")
        
        # Configuration options
        base_dir = self.config.get("base_dir", ".triangulum/neural_matrix")
        self.base_dir = Path(base_dir)
        
        # Initialize neural matrix
        neural_config = {
            "base_dir": str(self.base_dir),
            "db_path": self.config.get("db_path", str(self.base_dir / "patterns" / "patterns.db"))
        }
        self.neural_matrix = NeuralMatrix(neural_config)
        
        # Track initialization status
        self.initialized = True
        self.initialization_time = time.time()
        
        self.logger.info(f"Neural Matrix Integration initialized at {self.base_dir}")
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the neural matrix integration.
        
        Returns:
            Dictionary with initialization result
        """
        try:
            # Ensure directories exist
            os.makedirs(self.base_dir, exist_ok=True)
            os.makedirs(self.base_dir / "patterns", exist_ok=True)
            os.makedirs(self.base_dir / "weights", exist_ok=True)
            
            # Create default weights file if it doesn't exist
            weights_file = self.base_dir / "weights" / "default_weights.json"
            if not weights_file.exists():
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
                
                with open(weights_file, 'w') as f:
                    json.dump(default_weights, f, indent=2)
            
            self.initialized = True
            self.initialization_time = time.time()
            
            return {
                "success": True,
                "base_dir": str(self.base_dir),
                "initialization_time": self.initialization_time
            }
        except Exception as e:
            self.logger.error(f"Error initializing neural matrix integration: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def calculate_pattern_similarity(self, bug_description: str, tags: Optional[List[str]] = None,
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
        return self.neural_matrix.calculate_similarity(bug_description, tags, feature_weights)
    
    def integrate_with_triangulation_engine(self, engine: TriangulationEngine) -> Dict[str, Any]:
        """
        Integrate with the triangulation engine.
        
        Args:
            engine: TriangulationEngine instance
            
        Returns:
            Dictionary with integration result
        """
        try:
            # Check if neural matrix is enabled in the engine
            if not engine.neural_enabled:
                return {
                    "success": False,
                    "error": "Neural matrix is not enabled in the triangulation engine"
                }
            
            # Get neural weights
            weights = {}
            for category in ["agent", "feature"]:
                category_weights = self.neural_matrix.list_weights(category=category)
                for weight in category_weights:
                    key = weight["weight_key"]
                    value = weight["weight_value"]
                    weights[key] = value
            
            # Update engine's neural weights cache
            engine.neural_weights_cache.update(weights)
            engine.neural_weight_update_time = time.time()
            
            # Log the integration
            self.logger.info(f"Integrated with triangulation engine: {len(weights)} weights loaded")
            
            return {
                "success": True,
                "weights_loaded": len(weights),
                "updated_at": engine.neural_weight_update_time
            }
        except Exception as e:
            self.logger.error(f"Error integrating with triangulation engine: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def integrate_with_verification_engine(self, engine: VerificationEngine) -> Dict[str, Any]:
        """
        Integrate with the verification engine.
        
        Args:
            engine: VerificationEngine instance
            
        Returns:
            Dictionary with integration result
        """
        try:
            # Check if neural matrix is enabled in the engine
            if not engine.neural_enabled:
                return {
                    "success": False,
                    "error": "Neural matrix is not enabled in the verification engine"
                }
            
            # Trigger neural matrix integrity check
            check_result = engine._check_neural_matrix_integrity()
            
            # Return the result
            return {
                "success": check_result["success"],
                "checks": check_result.get("checks", []),
                "hash": check_result.get("hash", None),
                "hash_changed": check_result.get("hash_changed", False)
            }
        except Exception as e:
            self.logger.error(f"Error integrating with verification engine: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_pattern_success_rate(self, pattern_id: str, success: bool) -> Dict[str, Any]:
        """
        Update the success rate of a pattern based on a new result.
        
        Args:
            pattern_id: ID of the pattern to update
            success: Whether the pattern was successful
            
        Returns:
            Dictionary with update result
        """
        return self.neural_matrix.update_pattern_success_rate(pattern_id, success)
    
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
        return self.neural_matrix.recommend_solutions(bug_description, tags, feature_weights, limit)
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: The pattern ID to look for
            
        Returns:
            Pattern dictionary if found, None otherwise
        """
        return self.neural_matrix.get_pattern(pattern_id)
    
    def hub_api_pattern_access(self, pattern_id: str) -> Dict[str, Any]:
        """
        Access a pattern via the hub API.
        
        Args:
            pattern_id: The pattern ID to look for
            
        Returns:
            Pattern dictionary if found, error otherwise
        """
        pattern = self.neural_matrix.get_pattern(pattern_id)
        
        if pattern:
            return pattern
        else:
            return {
                "success": False,
                "error": f"Pattern with ID {pattern_id} not found"
            }
    
    def close(self) -> None:
        """Close the neural matrix connection."""
        self.neural_matrix.close()
