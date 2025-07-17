#!/usr/bin/env python3
"""
Neural Matrix Core

This module implements the core Neural Matrix functionality for pattern recognition,
learning from historical fixes, adaptive solution path selection, and weight-based
optimization.

The Neural Matrix is a critical component of the FixWurx system that enhances
the Triangulation Engine with machine learning capabilities.
"""

import os
import sys
import json
import time
import uuid
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(".triangulum/neural_matrix.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("NeuralMatrix")

class NeuralMatrix:
    """
    Neural Matrix core implementation.
    
    The NeuralMatrix provides:
    1. Pattern recognition for bugs and solutions
    2. Learning from historical fixes
    3. Adaptive solution path selection
    4. Weight-based optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Matrix.
        
        Args:
            config: Matrix configuration
        """
        self.config = config or {}
        
        # Create directories
        self.matrix_dir = Path(".triangulum/neural_matrix")
        self.matrix_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize feature matrices
        self.bug_features = {}  # bug_id -> feature vector
        self.solution_features = {}  # solution_id -> feature vector
        self.pattern_weights = {}  # pattern_id -> weight vector
        
        # Initialize learning parameters
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.75)
        self.pattern_count = self.config.get("pattern_count", 10)
        self.weight_decay = self.config.get("weight_decay", 0.001)
        
        # Load state if available
        self.state_file = self.matrix_dir / "matrix_state.json"
        self.load_state()
        
        logger.info("Neural Matrix initialized")
    
    def load_state(self) -> None:
        """Load Neural Matrix state from disk."""
        try:
            if not self.state_file.exists():
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Load feature matrices
            self.bug_features = {k: np.array(v) for k, v in state.get("bug_features", {}).items()}
            self.solution_features = {k: np.array(v) for k, v in state.get("solution_features", {}).items()}
            self.pattern_weights = {k: np.array(v) for k, v in state.get("pattern_weights", {}).items()}
            
            # Load learning parameters
            self.learning_rate = state.get("learning_rate", self.learning_rate)
            self.similarity_threshold = state.get("similarity_threshold", self.similarity_threshold)
            self.pattern_count = state.get("pattern_count", self.pattern_count)
            self.weight_decay = state.get("weight_decay", self.weight_decay)
            
            logger.info(f"Loaded Neural Matrix state: {len(self.bug_features)} bugs, {len(self.solution_features)} solutions, {len(self.pattern_weights)} patterns")
        except Exception as e:
            logger.error(f"Error loading Neural Matrix state: {e}")
    
    def save_state(self) -> None:
        """Save Neural Matrix state to disk."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            state = {
                "bug_features": {k: v.tolist() for k, v in self.bug_features.items()},
                "solution_features": {k: v.tolist() for k, v in self.solution_features.items()},
                "pattern_weights": {k: v.tolist() for k, v in self.pattern_weights.items()},
                "learning_rate": self.learning_rate,
                "similarity_threshold": self.similarity_threshold,
                "pattern_count": self.pattern_count,
                "weight_decay": self.weight_decay
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug("Neural Matrix state saved")
        except Exception as e:
            logger.error(f"Error saving Neural Matrix state: {e}")
    
    def extract_bug_features(self, bug_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a bug.
        
        Args:
            bug_data: Bug data
            
        Returns:
            np.ndarray: Feature vector
        """
        # Initialize feature vector (in a real implementation, this would be more sophisticated)
        features = np.zeros(20)
        
        # Extract basic features
        bug_id = bug_data.get("bug_id", "")
        title = bug_data.get("title", "")
        description = bug_data.get("description", "")
        severity = bug_data.get("severity", "medium")
        
        # Feature 0: Severity (critical=1.0, high=0.75, medium=0.5, low=0.25)
        if severity == "critical":
            features[0] = 1.0
        elif severity == "high":
            features[0] = 0.75
        elif severity == "medium":
            features[0] = 0.5
        elif severity == "low":
            features[0] = 0.25
        
        # Feature 1: Has description
        features[1] = 1.0 if description else 0.0
        
        # Feature 2: Description length (normalized)
        if description:
            features[2] = min(len(description) / 1000, 1.0)
        
        # Feature 3-7: Word counts for common error terms
        error_terms = ["error", "exception", "fail", "bug", "crash"]
        for i, term in enumerate(error_terms):
            count = description.lower().count(term) if description else 0
            features[3 + i] = min(count / 5, 1.0)  # Normalize
        
        # Feature 8-12: Word counts for common solution terms
        solution_terms = ["fix", "solve", "patch", "update", "resolve"]
        for i, term in enumerate(solution_terms):
            count = description.lower().count(term) if description else 0
            features[8 + i] = min(count / 5, 1.0)  # Normalize
        
        # Features 13-19: Reserved for future use
        
        return features
    
    def extract_solution_features(self, solution_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a solution.
        
        Args:
            solution_data: Solution data
            
        Returns:
            np.ndarray: Feature vector
        """
        # Initialize feature vector (in a real implementation, this would be more sophisticated)
        features = np.zeros(20)
        
        # Extract basic features
        solution_id = solution_data.get("solution_id", "")
        bug_id = solution_data.get("bug_id", "")
        steps = solution_data.get("steps", [])
        successful = solution_data.get("successful", False)
        
        # Feature 0: Solution success
        features[0] = 1.0 if successful else 0.0
        
        # Feature 1: Number of steps (normalized)
        features[1] = min(len(steps) / 10, 1.0)
        
        # Feature 2-6: Step type counts
        step_types = ["analyze", "plan", "implement", "verify", "learn"]
        for i, step_type in enumerate(step_types):
            count = sum(1 for step in steps if step.get("type") == step_type)
            features[2 + i] = min(count / 3, 1.0)  # Normalize
        
        # Feature 7: Has fallbacks
        fallbacks = solution_data.get("fallbacks", [])
        features[7] = 1.0 if fallbacks else 0.0
        
        # Feature 8: Number of fallbacks (normalized)
        features[8] = min(len(fallbacks) / 5, 1.0)
        
        # Feature 9: Execution time (normalized)
        execution_time = solution_data.get("execution_time", 0)
        features[9] = min(execution_time / 3600, 1.0)  # Normalize to hours
        
        # Features 10-19: Reserved for future use
        
        return features
    
    def register_bug(self, bug_data: Dict[str, Any]) -> None:
        """
        Register a bug with the Neural Matrix.
        
        Args:
            bug_data: Bug data
        """
        bug_id = bug_data.get("bug_id")
        if not bug_id:
            logger.warning("Bug ID not provided")
            return
        
        # Extract features
        features = self.extract_bug_features(bug_data)
        
        # Store in bug features dictionary
        self.bug_features[bug_id] = features
        
        # Save state
        self.save_state()
        
        logger.info(f"Registered bug {bug_id} with Neural Matrix")
    
    def register_solution(self, solution_data: Dict[str, Any]) -> None:
        """
        Register a solution with the Neural Matrix.
        
        Args:
            solution_data: Solution data
        """
        solution_id = solution_data.get("solution_id")
        if not solution_id:
            logger.warning("Solution ID not provided")
            return
        
        # Extract features
        features = self.extract_solution_features(solution_data)
        
        # Store in solution features dictionary
        self.solution_features[solution_id] = features
        
        # Save state
        self.save_state()
        
        logger.info(f"Registered solution {solution_id} with Neural Matrix")
    
    def find_similar_bugs(self, bug_data: Dict[str, Any], threshold: float = None) -> List[Tuple[str, float]]:
        """
        Find bugs similar to the given bug.
        
        Args:
            bug_data: Bug data
            threshold: Similarity threshold (optional, defaults to self.similarity_threshold)
            
        Returns:
            List[Tuple[str, float]]: List of similar bugs with similarity scores
        """
        if not self.bug_features:
            return []
        
        # Extract features for the query bug
        query_features = self.extract_bug_features(bug_data)
        
        # Use provided threshold or default
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        # Calculate similarity scores
        similar_bugs = []
        for bug_id, features in self.bug_features.items():
            # Skip if same bug
            if bug_id == bug_data.get("bug_id"):
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_features, features)
            
            # Add to similar bugs if above threshold
            if similarity >= threshold:
                similar_bugs.append((bug_id, similarity))
        
        # Sort by similarity (descending)
        similar_bugs.sort(key=lambda x: x[1], reverse=True)
        
        return similar_bugs
    
    def find_solution_patterns(self, bug_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Find solution patterns for the given bug.
        
        Args:
            bug_data: Bug data
            
        Returns:
            List[Tuple[str, float]]: List of solution patterns with confidence scores
        """
        # Find similar bugs
        similar_bugs = self.find_similar_bugs(bug_data)
        
        if not similar_bugs:
            return []
        
        # Extract features for the query bug
        query_features = self.extract_bug_features(bug_data)
        
        # Calculate pattern scores
        pattern_scores = {}
        for bug_id, similarity in similar_bugs:
            # Find solutions for this bug
            for solution_id, solution_features in self.solution_features.items():
                if solution_id.startswith(f"{bug_id}-"):
                    # Calculate pattern score (similarity * solution success)
                    pattern_score = similarity * solution_features[0]  # Feature 0 is success
                    
                    if solution_id not in pattern_scores or pattern_score > pattern_scores[solution_id]:
                        pattern_scores[solution_id] = pattern_score
        
        # Sort by score (descending)
        patterns = [(solution_id, score) for solution_id, score in pattern_scores.items()]
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        return patterns[:self.pattern_count]  # Return top N patterns
    
    def optimize_solution_path(self, bug_data: Dict[str, Any], paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize solution paths for a bug using neural matrix.
        
        Args:
            bug_data: Bug data
            paths: List of solution paths
            
        Returns:
            List[Dict[str, Any]]: Optimized paths with confidence scores
        """
        if not paths:
            return []
        
        # Extract features for the query bug
        query_features = self.extract_bug_features(bug_data)
        
        # Find similar bugs and their solutions
        similar_bugs = self.find_similar_bugs(bug_data)
        
        # Calculate path scores
        path_scores = []
        for path in paths:
            # Initialize base score
            base_score = path.get("score", 0.5)
            
            # Extract path features
            steps = path.get("steps", [])
            fallbacks = path.get("fallbacks", [])
            
            # Apply neural weights based on similar bugs
            neural_score = 0.0
            weight_sum = 0.0
            
            for bug_id, similarity in similar_bugs:
                for solution_id, solution_features in self.solution_features.items():
                    if solution_id.startswith(f"{bug_id}-"):
                        # Calculate weighted score
                        weight = similarity * solution_features[0]  # Feature 0 is success
                        
                        # Compare path features to solution
                        step_count_match = abs(len(steps) - solution_features[1] * 10) / 10
                        fallback_match = 1.0 if (bool(fallbacks) == bool(solution_features[7] > 0.5)) else 0.0
                        
                        # Calculate feature match score
                        feature_match = 1.0 - (step_count_match * 0.5 + (1.0 - fallback_match) * 0.5)
                        
                        # Add to neural score
                        neural_score += weight * feature_match
                        weight_sum += weight
            
            # Calculate final score
            if weight_sum > 0:
                final_score = base_score * 0.5 + (neural_score / weight_sum) * 0.5
            else:
                final_score = base_score
            
            # Add to path scores
            path_scores.append((path, final_score))
        
        # Sort by score (descending)
        path_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return paths with scores
        return [{**path, "neural_score": score} for path, score in path_scores]
    
    def learn_from_fix(self, bug_data: Dict[str, Any], solution_data: Dict[str, Any]) -> None:
        """
        Learn from a successful or failed fix.
        
        Args:
            bug_data: Bug data
            solution_data: Solution data
        """
        bug_id = bug_data.get("bug_id")
        solution_id = solution_data.get("solution_id")
        
        if not bug_id or not solution_id:
            logger.warning("Bug ID or Solution ID not provided")
            return
        
        # Register bug and solution (if not already registered)
        if bug_id not in self.bug_features:
            self.register_bug(bug_data)
        
        if solution_id not in self.solution_features:
            self.register_solution(solution_data)
        
        # Update pattern weights
        self._update_pattern_weights(bug_data, solution_data)
        
        logger.info(f"Learned from fix: bug={bug_id}, solution={solution_id}")
    
    def _update_pattern_weights(self, bug_data: Dict[str, Any], solution_data: Dict[str, Any]) -> None:
        """
        Update pattern weights based on a fix.
        
        Args:
            bug_data: Bug data
            solution_data: Solution data
        """
        bug_id = bug_data.get("bug_id")
        solution_id = solution_data.get("solution_id")
        successful = solution_data.get("successful", False)
        
        # Extract features
        bug_features = self.bug_features.get(bug_id)
        solution_features = self.solution_features.get(solution_id)
        
        if bug_features is None or solution_features is None:
            return
        
        # Create pattern ID
        pattern_id = f"pattern-{str(uuid.uuid4())[:8]}"
        
        # Initialize pattern weights if not exists
        if pattern_id not in self.pattern_weights:
            self.pattern_weights[pattern_id] = np.zeros(40)  # 20 for bug, 20 for solution
        
        # Concatenate bug and solution features
        features = np.concatenate([bug_features, solution_features])
        
        # Update weights
        weights = self.pattern_weights[pattern_id]
        
        # Apply learning
        if successful:
            # Reward successful patterns
            self.pattern_weights[pattern_id] = weights * (1 - self.learning_rate) + features * self.learning_rate
        else:
            # Penalize failed patterns
            self.pattern_weights[pattern_id] = weights * (1 - self.learning_rate) - features * self.learning_rate
        
        # Apply weight decay
        self.pattern_weights[pattern_id] *= (1 - self.weight_decay)
        
        # Save state
        self.save_state()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Compute dot product
        dot_product = np.dot(vec1, vec2)
        
        # Compute magnitudes
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)
        
        # Compute cosine similarity
        if mag1 > 0 and mag2 > 0:
            return dot_product / (mag1 * mag2)
        else:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get Neural Matrix statistics.
        
        Returns:
            Dict[str, Any]: Statistics
        """
        return {
            "bugs": len(self.bug_features),
            "solutions": len(self.solution_features),
            "patterns": len(self.pattern_weights),
            "learning_rate": self.learning_rate,
            "similarity_threshold": self.similarity_threshold,
            "pattern_count": self.pattern_count,
            "weight_decay": self.weight_decay
        }

# Singleton instance for the Neural Matrix
_matrix = None

def get_matrix(config: Dict[str, Any] = None) -> NeuralMatrix:
    """
    Get the singleton instance of the Neural Matrix.
    
    Args:
        config: Matrix configuration (used only if matrix is not initialized)
        
    Returns:
        NeuralMatrix: The matrix instance
    """
    global _matrix
    
    if _matrix is None:
        _matrix = NeuralMatrix(config)
    
    return _matrix

# API Functions for integration with Triangulation Engine

def register_bug(bug_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register a bug with the Neural Matrix.
    
    Args:
        bug_data: Bug data
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        matrix = get_matrix()
        matrix.register_bug(bug_data)
        
        return {
            "success": True,
            "bug_id": bug_data.get("bug_id")
        }
    except Exception as e:
        logger.error(f"Error registering bug with Neural Matrix: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def find_similar_bugs(bug_data: Dict[str, Any], threshold: float = None) -> Dict[str, Any]:
    """
    Find bugs similar to the given bug.
    
    Args:
        bug_data: Bug data
        threshold: Similarity threshold
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        matrix = get_matrix()
        similar_bugs = matrix.find_similar_bugs(bug_data, threshold)
        
        return {
            "success": True,
            "similar_bugs": similar_bugs
        }
    except Exception as e:
        logger.error(f"Error finding similar bugs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def find_solution_patterns(bug_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find solution patterns for the given bug.
    
    Args:
        bug_data: Bug data
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        matrix = get_matrix()
        patterns = matrix.find_solution_patterns(bug_data)
        
        return {
            "success": True,
            "patterns": patterns
        }
    except Exception as e:
        logger.error(f"Error finding solution patterns: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def optimize_solution_paths(bug_data: Dict[str, Any], paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Optimize solution paths for a bug.
    
    Args:
        bug_data: Bug data
        paths: List of solution paths
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        matrix = get_matrix()
        optimized_paths = matrix.optimize_solution_path(bug_data, paths)
        
        return {
            "success": True,
            "paths": optimized_paths
        }
    except Exception as e:
        logger.error(f"Error optimizing solution paths: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def learn_from_fix(bug_data: Dict[str, Any], solution_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Learn from a successful or failed fix.
    
    Args:
        bug_data: Bug data
        solution_data: Solution data
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        matrix = get_matrix()
        matrix.learn_from_fix(bug_data, solution_data)
        
        return {
            "success": True,
            "bug_id": bug_data.get("bug_id"),
            "solution_id": solution_data.get("solution_id")
        }
    except Exception as e:
        logger.error(f"Error learning from fix: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_matrix_stats() -> Dict[str, Any]:
    """
    Get Neural Matrix statistics.
    
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        matrix = get_matrix()
        stats = matrix.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting matrix stats: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Simple CLI for testing
    print("Neural Matrix Core - Run tests or import as module")
