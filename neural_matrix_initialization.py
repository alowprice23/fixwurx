#!/usr/bin/env python3
"""
neural_matrix_initialization.py
────────────────────────────────
Implements initialization procedures for the Neural Matrix component of FixWurx.

This module provides functionality for initializing, loading, saving, and validating
the Neural Matrix. It supports bootstrapping a new neural matrix with default weights,
loading from pre-trained models, initializing with various strategies, and ensuring
the matrix is properly configured before use.
"""

import os
import sys
import json
import logging
import numpy as np
import time
import uuid
import hashlib
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Import core components
from neural_matrix_core import NeuralMatrix
from storage_manager import StorageManager

# Configure logging
logger = logging.getLogger("NeuralMatrixInitialization")

class NeuralMatrixInitialization:
    """
    Implements initialization procedures for the Neural Matrix.
    
    This class provides methods for initializing, loading, saving, and validating
    the Neural Matrix. It supports various initialization strategies and ensures
    the matrix is properly configured before use.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Matrix initialization component.
        
        Args:
            config: Configuration for the Neural Matrix initialization.
        """
        self.config = config or {}
        self.neural_matrix = NeuralMatrix()
        self.storage_manager = StorageManager()
        
        # Default paths
        self.data_dir = self.config.get("data_dir", os.path.join(os.getcwd(), "data", "neural_matrix"))
        self.weights_file = self.config.get("weights_file", os.path.join(self.data_dir, "weights.npz"))
        self.config_file = self.config.get("config_file", os.path.join(self.data_dir, "config.json"))
        self.metadata_file = self.config.get("metadata_file", os.path.join(self.data_dir, "metadata.json"))
        self.backup_dir = self.config.get("backup_dir", os.path.join(self.data_dir, "backups"))
        
        # Initialization parameters
        self.initialization_method = self.config.get("initialization_method", "xavier")
        self.seed = self.config.get("seed", 42)
        self.default_layers = self.config.get("default_layers", [128, 256, 128, 64])
        self.weight_decay = self.config.get("weight_decay", 0.001)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.momentum = self.config.get("momentum", 0.9)
        self.use_bias = self.config.get("use_bias", True)
        self.activation = self.config.get("activation", "relu")
        
        # Model versioning
        self.version_format = self.config.get("version_format", "v{major}.{minor}.{patch}")
        self.current_version = self.config.get("current_version", "v0.1.0")
        
        # Metrics tracking
        self.track_metrics = self.config.get("track_metrics", True)
        self.metrics_file = self.config.get("metrics_file", os.path.join(self.data_dir, "metrics.json"))
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        logger.info("Neural Matrix Initialization component initialized")
    
    def initialize_matrix(self, force_reinit: bool = False) -> bool:
        """
        Initialize the Neural Matrix.
        
        Args:
            force_reinit: Whether to force reinitialization even if a matrix already exists.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        # Check if matrix already exists
        if self._matrix_exists() and not force_reinit:
            logger.info("Neural Matrix already exists, loading existing matrix")
            return self.load_matrix()
        
        logger.info("Initializing new Neural Matrix")
        
        try:
            # Create a new neural matrix configuration
            matrix_config = self._create_matrix_config()
            
            # Initialize weights
            weights = self._initialize_weights(matrix_config)
            
            # Create metadata
            metadata = self._create_metadata(matrix_config)
            
            # Apply configuration and weights to neural matrix
            success = self.neural_matrix.configure(matrix_config)
            if not success:
                logger.error("Failed to configure Neural Matrix")
                return False
            
            success = self.neural_matrix.set_weights(weights)
            if not success:
                logger.error("Failed to set Neural Matrix weights")
                return False
            
            # Save configuration, weights, and metadata
            self._save_config(matrix_config)
            self._save_weights(weights)
            self._save_metadata(metadata)
            
            logger.info("Neural Matrix initialized successfully")
            
            # Initialize metrics tracking
            if self.track_metrics:
                self._initialize_metrics()
            
            return True
        
        except Exception as e:
            logger.error(f"Error initializing Neural Matrix: {e}")
            return False
    
    def _matrix_exists(self) -> bool:
        """
        Check if a Neural Matrix already exists.
        
        Returns:
            True if a Neural Matrix exists, False otherwise.
        """
        return (
            os.path.exists(self.weights_file) and
            os.path.exists(self.config_file) and
            os.path.exists(self.metadata_file)
        )
    
    def _create_matrix_config(self) -> Dict[str, Any]:
        """
        Create a new Neural Matrix configuration.
        
        Returns:
            Neural Matrix configuration.
        """
        # Basic configuration
        config = {
            "version": self.current_version,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "initialization_method": self.initialization_method,
            "seed": self.seed,
            "layers": self.default_layers.copy(),
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "input_dim": self.config.get("input_dim", 64),
            "output_dim": self.config.get("output_dim", 32)
        }
        
        # Add any additional configuration from the provided config
        for key, value in self.config.items():
            if key not in config and key not in [
                "data_dir", "weights_file", "config_file", "metadata_file",
                "backup_dir", "initialization_method", "seed", "default_layers",
                "weight_decay", "learning_rate", "momentum", "use_bias",
                "activation", "version_format", "current_version", "track_metrics",
                "metrics_file"
            ]:
                config[key] = value
        
        return config
    
    def _initialize_weights(self, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Initialize Neural Matrix weights.
        
        Args:
            config: Neural Matrix configuration.
            
        Returns:
            Dictionary mapping layer names to weight matrices.
        """
        weights = {}
        layers = config["layers"]
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        use_bias = config["use_bias"]
        
        # Determine layer dimensions
        dims = [input_dim] + layers + [output_dim]
        
        # Initialize weights for each layer
        for i in range(len(dims) - 1):
            layer_name = f"layer_{i}"
            input_size = dims[i]
            output_size = dims[i + 1]
            
            # Initialize weights based on the specified method
            if config["initialization_method"] == "xavier":
                # Xavier/Glorot initialization
                weights[f"{layer_name}_weights"] = np.random.normal(
                    0, np.sqrt(2 / (input_size + output_size)),
                    (input_size, output_size)
                )
            elif config["initialization_method"] == "he":
                # He initialization
                weights[f"{layer_name}_weights"] = np.random.normal(
                    0, np.sqrt(2 / input_size),
                    (input_size, output_size)
                )
            elif config["initialization_method"] == "uniform":
                # Uniform initialization
                scale = np.sqrt(6 / (input_size + output_size))
                weights[f"{layer_name}_weights"] = np.random.uniform(
                    -scale, scale,
                    (input_size, output_size)
                )
            else:
                # Default to Xavier initialization
                weights[f"{layer_name}_weights"] = np.random.normal(
                    0, np.sqrt(2 / (input_size + output_size)),
                    (input_size, output_size)
                )
            
            # Initialize biases if enabled
            if use_bias:
                weights[f"{layer_name}_biases"] = np.zeros(output_size)
        
        # Add special weights for pattern recognition and bug fixing
        weights["pattern_recognition"] = np.random.normal(0, 0.01, (64, 128))
        weights["bug_fixing"] = np.random.normal(0, 0.01, (128, 64))
        weights["solution_generation"] = np.random.normal(0, 0.01, (64, 32))
        
        return weights
    
    def _create_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for the Neural Matrix.
        
        Args:
            config: Neural Matrix configuration.
            
        Returns:
            Neural Matrix metadata.
        """
        return {
            "id": str(uuid.uuid4()),
            "version": config["version"],
            "created_at": config["created_at"],
            "updated_at": config["updated_at"],
            "description": "Neural Matrix for FixWurx",
            "architecture": {
                "type": "multilayer_perceptron",
                "layers": config["layers"],
                "input_dim": config["input_dim"],
                "output_dim": config["output_dim"],
                "activation": config["activation"]
            },
            "training": {
                "epochs": 0,
                "samples": 0,
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "momentum": config["momentum"]
            },
            "performance": {
                "accuracy": 0.0,
                "loss": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            },
            "hash": "",  # Will be set after saving weights
            "tags": ["initial", "untrained"]
        }
    
    def _save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save Neural Matrix configuration.
        
        Args:
            config: Neural Matrix configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Neural Matrix configuration saved to {self.config_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving Neural Matrix configuration: {e}")
            return False
    
    def _save_weights(self, weights: Dict[str, np.ndarray]) -> bool:
        """
        Save Neural Matrix weights.
        
        Args:
            weights: Neural Matrix weights.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            np.savez_compressed(self.weights_file, **weights)
            
            logger.info(f"Neural Matrix weights saved to {self.weights_file}")
            
            # Update metadata with weights hash
            self._update_weights_hash()
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving Neural Matrix weights: {e}")
            return False
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Save Neural Matrix metadata.
        
        Args:
            metadata: Neural Matrix metadata.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Neural Matrix metadata saved to {self.metadata_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving Neural Matrix metadata: {e}")
            return False
    
    def _update_weights_hash(self) -> None:
        """
        Update the hash of the weights file in the metadata.
        """
        try:
            # Calculate hash of weights file
            hasher = hashlib.sha256()
            with open(self.weights_file, "rb") as f:
                buf = f.read()
                hasher.update(buf)
            
            weights_hash = hasher.hexdigest()
            
            # Update metadata with hash
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                
                metadata["hash"] = weights_hash
                metadata["updated_at"] = int(time.time())
                
                with open(self.metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error updating weights hash: {e}")
    
    def _initialize_metrics(self) -> bool:
        """
        Initialize metrics tracking.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            metrics = {
                "version": self.current_version,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
                "training": {
                    "epochs": [],
                    "loss": [],
                    "accuracy": []
                },
                "validation": {
                    "epochs": [],
                    "loss": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "f1_score": []
                },
                "performance": {
                    "inference_time": [],
                    "memory_usage": []
                }
            }
            
            with open(self.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Neural Matrix metrics initialized and saved to {self.metrics_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing Neural Matrix metrics: {e}")
            return False
    
    def load_matrix(self) -> bool:
        """
        Load an existing Neural Matrix.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Check if matrix exists
            if not self._matrix_exists():
                logger.error("Neural Matrix files not found")
                return False
            
            # Load configuration
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            # Validate configuration
            if not self._validate_config(config):
                logger.error("Neural Matrix configuration validation failed")
                return False
            
            # Load weights
            try:
                weights_npz = np.load(self.weights_file)
                weights = {key: weights_npz[key] for key in weights_npz.files}
            except Exception as e:
                logger.error(f"Error loading Neural Matrix weights: {e}")
                return False
            
            # Validate weights against configuration
            if not self._validate_weights(weights, config):
                logger.error("Neural Matrix weights validation failed")
                return False
            
            # Apply configuration and weights to neural matrix
            success = self.neural_matrix.configure(config)
            if not success:
                logger.error("Failed to configure Neural Matrix")
                return False
            
            success = self.neural_matrix.set_weights(weights)
            if not success:
                logger.error("Failed to set Neural Matrix weights")
                return False
            
            # Load metadata
            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Verify weights hash
                if not self._verify_weights_hash(metadata.get("hash", "")):
                    logger.warning("Neural Matrix weights hash verification failed")
            except Exception as e:
                logger.error(f"Error loading Neural Matrix metadata: {e}")
            
            logger.info("Neural Matrix loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading Neural Matrix: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate Neural Matrix configuration.
        
        Args:
            config: Neural Matrix configuration.
            
        Returns:
            True if valid, False otherwise.
        """
        # Check required fields
        required_fields = ["version", "created_at", "updated_at", "layers", "input_dim", "output_dim"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in configuration: {field}")
                return False
        
        # Validate layers
        layers = config.get("layers", [])
        if not isinstance(layers, list) or len(layers) == 0:
            logger.error("Invalid layers configuration")
            return False
        
        # Validate dimensions
        input_dim = config.get("input_dim")
        output_dim = config.get("output_dim")
        if not isinstance(input_dim, int) or not isinstance(output_dim, int):
            logger.error("Invalid input or output dimensions")
            return False
        
        # Validate activation function
        activation = config.get("activation")
        valid_activations = ["relu", "sigmoid", "tanh", "linear"]
        if activation not in valid_activations:
            logger.error(f"Invalid activation function: {activation}")
            return False
        
        return True
    
    def _validate_weights(self, weights: Dict[str, np.ndarray], config: Dict[str, Any]) -> bool:
        """
        Validate Neural Matrix weights against configuration.
        
        Args:
            weights: Neural Matrix weights.
            config: Neural Matrix configuration.
            
        Returns:
            True if valid, False otherwise.
        """
        # Get layer dimensions
        layers = config["layers"]
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        use_bias = config.get("use_bias", True)
        
        # Determine expected layer dimensions
        dims = [input_dim] + layers + [output_dim]
        
        # Check each layer's weights
        for i in range(len(dims) - 1):
            layer_name = f"layer_{i}"
            input_size = dims[i]
            output_size = dims[i + 1]
            
            # Check weights
            weight_name = f"{layer_name}_weights"
            if weight_name not in weights:
                logger.error(f"Missing weights for layer: {layer_name}")
                return False
            
            if weights[weight_name].shape != (input_size, output_size):
                logger.error(f"Invalid shape for {weight_name}: expected {(input_size, output_size)}, got {weights[weight_name].shape}")
                return False
            
            # Check biases if enabled
            if use_bias:
                bias_name = f"{layer_name}_biases"
                if bias_name not in weights:
                    logger.error(f"Missing biases for layer: {layer_name}")
                    return False
                
                if weights[bias_name].shape != (output_size,):
                    logger.error(f"Invalid shape for {bias_name}: expected {(output_size,)}, got {weights[bias_name].shape}")
                    return False
        
        # Check special weights
        special_weights = ["pattern_recognition", "bug_fixing", "solution_generation"]
        for weight_name in special_weights:
            if weight_name not in weights:
                logger.warning(f"Missing special weights: {weight_name}")
        
        return True
    
    def _verify_weights_hash(self, expected_hash: str) -> bool:
        """
        Verify the hash of the weights file.
        
        Args:
            expected_hash: Expected hash value.
            
        Returns:
            True if hash matches, False otherwise.
        """
        if not expected_hash:
            return False
        
        try:
            # Calculate hash of weights file
            hasher = hashlib.sha256()
            with open(self.weights_file, "rb") as f:
                buf = f.read()
                hasher.update(buf)
            
            weights_hash = hasher.hexdigest()
            
            # Compare hashes
            return weights_hash == expected_hash
        
        except Exception as e:
            logger.error(f"Error verifying weights hash: {e}")
            return False
    
    def save_matrix(self) -> bool:
        """
        Save the current state of the Neural Matrix.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get current configuration and weights
            config = self.neural_matrix.get_config()
            weights = self.neural_matrix.get_weights()
            
            if not config or not weights:
                logger.error("Failed to get current Neural Matrix state")
                return False
            
            # Update timestamps
            config["updated_at"] = int(time.time())
            
            # Save configuration and weights
            self._save_config(config)
            self._save_weights(weights)
            
            # Update metadata
            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                
                metadata["updated_at"] = config["updated_at"]
                
                # Update training metrics if available
                training_metrics = self.neural_matrix.get_training_metrics()
                if training_metrics:
                    metadata["training"] = {
                        "epochs": training_metrics.get("epochs", 0),
                        "samples": training_metrics.get("samples", 0),
                        "learning_rate": training_metrics.get("learning_rate", self.learning_rate),
                        "weight_decay": training_metrics.get("weight_decay", self.weight_decay),
                        "momentum": training_metrics.get("momentum", self.momentum)
                    }
                
                # Update performance metrics if available
                performance_metrics = self.neural_matrix.get_performance_metrics()
                if performance_metrics:
                    metadata["performance"] = {
                        "accuracy": performance_metrics.get("accuracy", 0.0),
                        "loss": performance_metrics.get("loss", 0.0),
                        "precision": performance_metrics.get("precision", 0.0),
                        "recall": performance_metrics.get("recall", 0.0),
                        "f1_score": performance_metrics.get("f1_score", 0.0)
                    }
                
                self._save_metadata(metadata)
            except Exception as e:
                logger.error(f"Error updating metadata: {e}")
            
            logger.info("Neural Matrix saved successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error saving Neural Matrix: {e}")
            return False
    
    def backup_matrix(self, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of the current Neural Matrix.
        
        Args:
            backup_name: Name for the backup (optional).
            
        Returns:
            Path to the backup directory or empty string if backup failed.
        """
        try:
            # Generate backup name if not provided
            if not backup_name:
                timestamp = int(time.time())
                backup_name = f"backup_{timestamp}"
            
            # Create backup directory
            backup_dir = os.path.join(self.backup_dir, backup_name)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy files to backup directory
            backup_config_file = os.path.join(backup_dir, "config.json")
            backup_weights_file = os.path.join(backup_dir, "weights.npz")
            backup_metadata_file = os.path.join(backup_dir, "metadata.json")
            
            # Copy configuration
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as src_f:
                    config = json.load(src_f)
                
                with open(backup_config_file, "w") as dst_f:
                    json.dump(config, dst_f, indent=2)
            
            # Copy weights
            if os.path.exists(self.weights_file):
                weights_npz = np.load(self.weights_file)
                weights = {key: weights_npz[key] for key in weights_npz.files}
                np.savez_compressed(backup_weights_file, **weights)
            
            # Copy metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as src_f:
                    metadata = json.load(src_f)
                
                # Update backup information
                metadata["backup_info"] = {
                    "created_at": int(time.time()),
                    "name": backup_name,
                    "source_version": metadata.get("version", "unknown")
                }
                
                with open(backup_metadata_file, "w") as dst_f:
                    json.dump(metadata, dst_f, indent=2)
            
            logger.info(f"Neural Matrix backup created: {backup_dir}")
            return backup_dir
        
        except Exception as e:
            logger.error(f"Error creating Neural Matrix backup: {e}")
            return ""
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """
        Restore Neural Matrix from a backup.
        
        Args:
            backup_name: Name of the backup to restore from.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Build backup directory path
            backup_dir = os.path.join(self.backup_dir, backup_name)
            
            # Check if backup exists
            if not os.path.exists(backup_dir):
                logger.error(f"Backup not found: {backup_dir}")
                return False
            
            # Build backup file paths
            backup_config_file = os.path.join(backup_dir, "config.json")
            backup_weights_file = os.path.join(backup_dir, "weights.npz")
            backup_metadata_file = os.path.join(backup_dir, "metadata.json")
            
            # Check if backup files exist
            if not os.path.exists(backup_config_file) or not os.path.exists(backup_weights_file):
                logger.error(f"Incomplete backup: {backup_dir}")
                return False
            
            # Backup current state before restoring
            current_backup_name = f"pre_restore_{int(time.time())}"
            self.backup_matrix(current_backup_name)
            
            # Copy backup files to main location
            # Copy configuration
            with open(backup_config_file, "r") as src_f:
                config = json.load(src_f)
            
            with open(self.config_file, "w") as dst_f:
                json.dump(config, dst_f, indent=2)
            
            # Copy weights
            weights_npz = np.load(backup_weights_file)
            weights = {key: weights_npz[key] for key in weights_npz.files}
            np.savez_compressed(self.weights_file, **weights)
            
            # Copy metadata if exists
            if os.path.exists(backup_metadata_file):
                with open(backup_metadata_file, "r") as src_f:
                    metadata = json.load(src_f)
                
                # Update restoration information
                metadata["restoration_info"] = {
                    "restored_at": int(time.time()),
                    "backup_name": backup_name
                }
                
                with open(self.metadata_file, "w") as dst_f:
                    json.dump(metadata, dst_f, indent=2)
            
            # Load the restored matrix
            success = self.load_matrix()
            
            if success:
                logger.info(f"Neural Matrix restored from backup: {backup_name}")
            else:
                logger.error(f"Failed to load restored Neural Matrix from backup: {backup_name}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error restoring Neural Matrix from backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available Neural Matrix backups.
        
        Returns:
            List of backup information dictionaries.
        """
        backups = []
        
        try:
            # Check if backup directory exists
            if not os.path.exists(self.backup_dir):
                return backups
            
            # List subdirectories in backup directory
            for backup_name in os.listdir(self.backup_dir):
                backup_dir = os.path.join(self.backup_dir, backup_name)
                
                if not os.path.isdir(backup_dir):
                    continue
                
                # Check for required backup files
                backup_config_file = os.path.join(backup_dir, "config.json")
                backup_weights_file = os.path.join(backup_dir, "weights.npz")
                backup_metadata_file = os.path.join(backup_dir, "metadata.json")
                
                if not os.path.exists(backup_config_file) or not os.path.exists(backup_weights_file):
                    # Skip incomplete backups
                    continue
                
                # Extract backup information
                backup_info = {
                    "name": backup_name,
                    "path": backup_dir,
                    "created_at": os.path.getctime(backup_dir)
                }
                
                # Add metadata if available
                if os.path.exists(backup_metadata_file):
                    try:
                        with open(backup_metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        backup_info["version"] = metadata.get("version", "unknown")
                        backup_info["created_at"] = metadata.get("backup_info", {}).get("created_at", backup_info["created_at"])
                        backup_info["source_version"] = metadata.get("backup_info", {}).get("source_version", "unknown")
                    except Exception as e:
                        logger.error(f"Error loading backup metadata: {e}")
                
                backups.append(backup_info)
            
            # Sort backups by creation time (newest first)
            backups.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            
            return backups
        
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return backups
    
    def get_matrix_info(self) -> Dict[str, Any]:
        """
        Get information about the current Neural Matrix.
        
        Returns:
            Dictionary with Neural Matrix information.
        """
        info = {
            "exists": self._matrix_exists(),
            "size": {
                "weights_file": os.path.getsize(self.weights_file) if os.path.exists(self.weights_file) else 0,
                "config_file": os.path.getsize(self.config_file) if os.path.exists(self.config_file) else 0,
                "metadata_file": os.path.getsize(self.metadata_file) if os.path.exists(self.metadata_file) else 0
            },
            "timestamps": {
                "created_at": 0,
                "updated_at": 0
            },
            "version": "unknown",
            "architecture": {},
            "backups": len(self.list_backups())
        }
        
        # Add metadata if available
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                
                info["version"] = metadata.get("version", "unknown")
                info["timestamps"]["created_at"] = metadata.get("created_at", 0)
                info["timestamps"]["updated_at"] = metadata.get("updated_at", 0)
                info["architecture"] = metadata.get("architecture", {})
                
                # Add performance metrics if available
                if "performance" in metadata:
                    info["performance"] = metadata["performance"]
                
                # Add training information if available
                if "training" in metadata:
                    info["training"] = metadata["training"]
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        return info
    
    def load_pretrained_model(self, model_path: str) -> bool:
        """
        Load a pre-trained model into the Neural Matrix.
        
        Args:
            model_path: Path to the pre-trained model.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Pre-trained model not found: {model_path}")
                return False
            
            # Create a backup of current state before loading
            self.backup_matrix("pre_pretrained_load")
            
            # Determine model format
            if model_path.endswith(".npz"):
                # Load weights directly
                weights_npz = np.load(model_path)
                weights = {key: weights_npz[key] for key in weights_npz.files}
                
                # Apply weights to neural matrix
                success = self.neural_matrix.set_weights(weights)
                if not success:
                    logger.error("Failed to set Neural Matrix weights from pre-trained model")
                    return False
                
            elif os.path.isdir(model_path):
                # Treat as a backup directory
                model_config_file = os.path.join(model_path, "config.json")
                model_weights_file = os.path.join(model_path, "weights.npz")
                
                if not os.path.exists(model_config_file) or not os.path.exists(model_weights_file):
                    logger.error(f"Incomplete model directory: {model_path}")
                    return False
                
                # Load configuration
                with open(model_config_file, "r") as f:
                    config = json.load(f)
                
                # Validate configuration
                if not self._validate_config(config):
                    logger.error("Pre-trained model configuration validation failed")
                    return False
                
                # Load weights
                weights_npz = np.load(model_weights_file)
                weights = {key: weights_npz[key] for key in weights_npz.files}
                
                # Validate weights against configuration
                if not self._validate_weights(weights, config):
                    logger.error("Pre-trained model weights validation failed")
                    return False
                
                # Apply configuration and weights to neural matrix
                success = self.neural_matrix.configure(config)
                if not success:
                    logger.error("Failed to configure Neural Matrix with pre-trained model")
                    return False
                
                success = self.neural_matrix.set_weights(weights)
                if not success:
                    logger.error("Failed to set Neural Matrix weights from pre-trained model")
                    return False
                
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return False
            
            # Save the updated matrix
            self.save_matrix()
            
            logger.info(f"Pre-trained model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {e}")
            return False
    
    def export_model(self, output_path: str, format: str = "npz") -> bool:
        """
        Export the Neural Matrix model.
        
        Args:
            output_path: Path to export the model to.
            format: Export format ('npz', 'json', 'dir').
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get current weights and configuration
            weights = self.neural_matrix.get_weights()
            config = self.neural_matrix.get_config()
            
            if not weights or not config:
                logger.error("Failed to get Neural Matrix state for export")
                return False
            
            # Export based on format
            if format == "npz":
                # Export weights as NPZ file
                np.savez_compressed(output_path, **weights)
                logger.info(f"Neural Matrix weights exported to {output_path}")
                
            elif format == "json":
                # Export configuration as JSON file
                with open(output_path, "w") as f:
                    json.dump({
                        "config": config,
                        "metadata": {
                            "exported_at": int(time.time()),
                            "version": config.get("version", "unknown")
                        }
                    }, f, indent=2)
                logger.info(f"Neural Matrix configuration exported to {output_path}")
                
            elif format == "dir":
                # Export as directory with weights, config, and metadata
                os.makedirs(output_path, exist_ok=True)
                
                # Export weights
                weights_path = os.path.join(output_path, "weights.npz")
                np.savez_compressed(weights_path, **weights)
                
                # Export configuration
                config_path = os.path.join(output_path, "config.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                
                # Export metadata
                metadata_path = os.path.join(output_path, "metadata.json")
                with open(self.metadata_file, "r") as src_f:
                    metadata = json.load(src_f)
                
                metadata["export_info"] = {
                    "exported_at": int(time.time()),
                    "format": format
                }
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Neural Matrix exported to directory {output_path}")
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Neural Matrix: {e}")
            return False
    
    def update_version(self, version: Optional[str] = None, increment: str = "patch") -> str:
        """
        Update the Neural Matrix version.
        
        Args:
            version: New version string (optional).
            increment: Version part to increment if version not provided ('major', 'minor', 'patch').
            
        Returns:
            New version string.
        """
        try:
            # Get current version
            current_version = self.current_version
            
            # Parse current version
            if current_version.startswith("v"):
                current_version = current_version[1:]
            
            major, minor, patch = map(int, current_version.split("."))
            
            # Set new version
            if version:
                # Use provided version
                new_version = version
                if not new_version.startswith("v"):
                    new_version = f"v{new_version}"
            else:
                # Increment version
                if increment == "major":
                    major += 1
                    minor = 0
                    patch = 0
                elif increment == "minor":
                    minor += 1
                    patch = 0
                else:  # patch
                    patch += 1
                
                new_version = f"v{major}.{minor}.{patch}"
            
            # Update current version
            self.current_version = new_version
            
            # Update configuration if matrix exists
            if self._matrix_exists():
                try:
                    # Load configuration
                    with open(self.config_file, "r") as f:
                        config = json.load(f)
                    
                    # Update version
                    config["version"] = new_version
                    config["updated_at"] = int(time.time())
                    
                    # Save configuration
                    with open(self.config_file, "w") as f:
                        json.dump(config, f, indent=2)
                    
                    # Update metadata
                    with open(self.metadata_file, "r") as f:
                        metadata = json.load(f)
                    
                    metadata["version"] = new_version
                    metadata["updated_at"] = config["updated_at"]
                    
                    with open(self.metadata_file, "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Update neural matrix
                    self.neural_matrix.configure(config)
                    
                except Exception as e:
                    logger.error(f"Error updating version in files: {e}")
            
            logger.info(f"Neural Matrix version updated to {new_version}")
            return new_version
            
        except Exception as e:
            logger.error(f"Error updating Neural Matrix version: {e}")
            return self.current_version

# Main entry point
def initialize_neural_matrix(config_path: Optional[str] = None, force_reinit: bool = False) -> bool:
    """
    Initialize the Neural Matrix.
    
    Args:
        config_path: Path to configuration file (optional).
        force_reinit: Whether to force reinitialization even if a matrix already exists.
        
    Returns:
        True if initialization was successful, False otherwise.
    """
    try:
        # Load configuration if provided
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        
        # Create initializer
        initializer = NeuralMatrixInitialization(config)
        
        # Initialize matrix
        success = initializer.initialize_matrix(force_reinit)
        
        return success
    
    except Exception as e:
        logger.error(f"Error initializing Neural Matrix: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Matrix Initialization")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force reinitialization")
    parser.add_argument("--backup", action="store_true", help="Create a backup before initialization")
    parser.add_argument("--restore", help="Restore from backup")
    parser.add_argument("--list-backups", action="store_true", help="List available backups")
    parser.add_argument("--export", help="Export model to file")
    parser.add_argument("--export-format", choices=["npz", "json", "dir"], default="npz", help="Export format")
    parser.add_argument("--load-pretrained", help="Load pre-trained model")
    parser.add_argument("--version", help="Set version")
    
    args = parser.parse_args()
    
    # Create initializer
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    
    initializer = NeuralMatrixInitialization(config)
    
    if args.list_backups:
        # List backups
        backups = initializer.list_backups()
        print(f"Found {len(backups)} backups:")
        for backup in backups:
            print(f"  - {backup['name']}: {backup.get('version', 'unknown')} (created at {time.ctime(backup['created_at'])})")
    
    elif args.restore:
        # Restore from backup
        success = initializer.restore_from_backup(args.restore)
        if success:
            print(f"Neural Matrix restored from backup: {args.restore}")
        else:
            print(f"Failed to restore Neural Matrix from backup: {args.restore}")
            sys.exit(1)
    
    elif args.export:
        # Export model
        success = initializer.export_model(args.export, args.export_format)
        if success:
            print(f"Neural Matrix exported to {args.export} (format: {args.export_format})")
        else:
            print(f"Failed to export Neural Matrix to {args.export}")
            sys.exit(1)
    
    elif args.load_pretrained:
        # Load pre-trained model
        success = initializer.load_pretrained_model(args.load_pretrained)
        if success:
            print(f"Pre-trained model loaded successfully from {args.load_pretrained}")
        else:
            print(f"Failed to load pre-trained model from {args.load_pretrained}")
            sys.exit(1)
    
    else:
        # Create backup if requested
        if args.backup:
            backup_dir = initializer.backup_matrix()
            if backup_dir:
                print(f"Created backup: {backup_dir}")
        
        # Initialize matrix
        success = initializer.initialize_matrix(args.force)
        
        if success:
            print("Neural Matrix initialized successfully")
            
            # Get matrix info
            info = initializer.get_matrix_info()
            
            print(f"Version: {info.get('version', 'unknown')}")
            print(f"Architecture: {len(info.get('architecture', {}).get('layers', []))} layers")
            print(f"Created: {time.ctime(info.get('timestamps', {}).get('created_at', 0))}")
            print(f"Updated: {time.ctime(info.get('timestamps', {}).get('updated_at', 0))}")
            print(f"Backups: {info.get('backups', 0)}")
        else:
            print("Failed to initialize Neural Matrix")
            sys.exit(1)
        
        # Set version if provided
        if args.version:
            new_version = initializer.update_version(args.version)
            print(f"Version updated to {new_version}")
