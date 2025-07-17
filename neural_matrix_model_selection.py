#!/usr/bin/env python3
"""
neural_matrix_model_selection.py
────────────────────────────────
Implements model selection and activation for the Neural Matrix component of FixWurx.

This module provides functionality for selecting, configuring, and activating 
neural matrix models. It supports multiple selection strategies, auto-tuning,
and environment-aware configuration to optimize performance based on the specific
task and available resources.
"""

import os
import sys
import json
import logging
import numpy as np
import time
import psutil
import tempfile
import shutil
import hashlib
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path

# Import core components
from neural_matrix_core import NeuralMatrix
from neural_matrix_initialization import NeuralMatrixInitialization, initialize_neural_matrix
from storage_manager import StorageManager

# Configure logging
logger = logging.getLogger("NeuralMatrixModelSelection")

class NeuralMatrixModelSelection:
    """
    Implements model selection and activation for the Neural Matrix.
    
    This class provides methods for selecting, configuring, and activating
    neural matrix models based on task requirements and available resources.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Matrix model selection component.
        
        Args:
            config: Configuration for model selection.
        """
        self.config = config or {}
        self.neural_matrix = NeuralMatrix()
        self.storage_manager = StorageManager()
        self.initializer = NeuralMatrixInitialization(config)
        
        # Default paths
        self.data_dir = self.config.get("data_dir", os.path.join(os.getcwd(), "data", "neural_matrix"))
        self.models_dir = self.config.get("models_dir", os.path.join(self.data_dir, "models"))
        self.active_model_file = self.config.get("active_model_file", os.path.join(self.data_dir, "active_model.json"))
        self.registry_file = self.config.get("registry_file", os.path.join(self.models_dir, "registry.json"))
        
        # Selection parameters
        self.selection_method = self.config.get("selection_method", "performance")
        self.default_model = self.config.get("default_model", "default")
        self.auto_tune = self.config.get("auto_tune", True)
        self.performance_threshold = self.config.get("performance_threshold", 0.8)
        self.resource_aware = self.config.get("resource_aware", True)
        self.cache_models = self.config.get("cache_models", True)
        self.validation_required = self.config.get("validation_required", True)
        
        # Task category mapping
        self.task_categories = self.config.get("task_categories", {
            "bug_detection": ["logical_error", "syntax_error", "runtime_error", "security_vulnerability"],
            "code_analysis": ["complexity", "style", "performance", "security"],
            "fix_generation": ["automatic_fix", "guided_fix", "refactoring"],
            "verification": ["test_generation", "regression_testing", "validation"]
        })
        
        # Resource thresholds
        self.resource_thresholds = self.config.get("resource_thresholds", {
            "cpu_percent": 80,
            "memory_percent": 80,
            "disk_percent": 90
        })
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize model registry if it doesn't exist
        if not os.path.exists(self.registry_file):
            self._initialize_registry()
        
        logger.info("Neural Matrix Model Selection component initialized")
    
    def _initialize_registry(self) -> None:
        """
        Initialize the model registry.
        """
        registry = {
            "version": "1.0.0",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "models": {
                "default": {
                    "id": "default",
                    "name": "Default Model",
                    "description": "Default Neural Matrix model",
                    "version": "v0.1.0",
                    "path": os.path.join(self.models_dir, "default"),
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                    "task_categories": ["bug_detection", "code_analysis", "fix_generation", "verification"],
                    "performance": {
                        "accuracy": 0.75,
                        "precision": 0.8,
                        "recall": 0.7,
                        "f1_score": 0.75
                    },
                    "resources": {
                        "cpu_usage": 20,
                        "memory_usage": 256,
                        "disk_usage": 10
                    },
                    "metadata": {
                        "architecture": {
                            "type": "multilayer_perceptron",
                            "layers": [128, 256, 128, 64],
                            "input_dim": 64,
                            "output_dim": 32,
                            "activation": "relu"
                        },
                        "training": {
                            "epochs": 0,
                            "samples": 0
                        }
                    },
                    "tags": ["default", "general-purpose"]
                }
            },
            "active_model": "default"
        }
        
        # Save registry
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)
        
        logger.info("Initialized model registry")
        
        # Create default model directory
        default_model_dir = os.path.join(self.models_dir, "default")
        os.makedirs(default_model_dir, exist_ok=True)
        
        # Initialize default model
        config = {
            "data_dir": default_model_dir,
            "weights_file": os.path.join(default_model_dir, "weights.npz"),
            "config_file": os.path.join(default_model_dir, "config.json"),
            "metadata_file": os.path.join(default_model_dir, "metadata.json"),
            "backup_dir": os.path.join(default_model_dir, "backups"),
            "default_layers": [128, 256, 128, 64],
            "input_dim": 64,
            "output_dim": 32
        }
        
        default_initializer = NeuralMatrixInitialization(config)
        default_initializer.initialize_matrix(force_reinit=True)
        
        logger.info("Initialized default model")
        
        # Set active model
        self._set_active_model("default")
    
    def _set_active_model(self, model_id: str) -> None:
        """
        Set the active model.
        
        Args:
            model_id: ID of the model to set as active.
        """
        # Get model from registry
        registry = self._get_registry()
        
        if model_id not in registry["models"]:
            logger.error(f"Model {model_id} not found in registry")
            return
        
        # Set active model in registry
        registry["active_model"] = model_id
        registry["updated_at"] = int(time.time())
        
        # Save registry
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)
        
        # Save active model info
        active_model = registry["models"][model_id]
        
        with open(self.active_model_file, "w") as f:
            json.dump({
                "id": model_id,
                "name": active_model["name"],
                "description": active_model["description"],
                "version": active_model["version"],
                "path": active_model["path"],
                "activated_at": int(time.time())
            }, f, indent=2)
        
        logger.info(f"Set active model to {model_id}")
    
    def _get_registry(self) -> Dict[str, Any]:
        """
        Get the model registry.
        
        Returns:
            Model registry.
        """
        if not os.path.exists(self.registry_file):
            self._initialize_registry()
        
        with open(self.registry_file, "r") as f:
            registry = json.load(f)
        
        return registry
    
    def _get_active_model(self) -> Dict[str, Any]:
        """
        Get the active model.
        
        Returns:
            Active model info.
        """
        registry = self._get_registry()
        active_model_id = registry.get("active_model", self.default_model)
        
        if active_model_id not in registry["models"]:
            logger.warning(f"Active model {active_model_id} not found in registry, using default")
            active_model_id = self.default_model
            
            if active_model_id not in registry["models"]:
                logger.error(f"Default model {active_model_id} not found in registry")
                return {}
        
        return registry["models"][active_model_id]
    
    def select_model(self, task: Dict[str, Any]) -> str:
        """
        Select the best model for a given task.
        
        Args:
            task: Task information.
            
        Returns:
            ID of the selected model.
        """
        # Get task category
        task_category = task.get("category")
        if not task_category:
            logger.warning("No task category provided, using active model")
            return self._get_active_model_id()
        
        # Get registry
        registry = self._get_registry()
        
        # Filter models by task category
        suitable_models = []
        for model_id, model in registry["models"].items():
            model_categories = model.get("task_categories", [])
            
            if task_category in model_categories:
                suitable_models.append(model)
        
        if not suitable_models:
            logger.warning(f"No models found for task category {task_category}, using active model")
            return self._get_active_model_id()
        
        # Select model based on selection method
        if self.selection_method == "performance":
            # Select model with highest performance
            if task.get("metric"):
                # Use specific metric if provided
                metric = task["metric"]
                suitable_models.sort(
                    key=lambda m: m.get("performance", {}).get(metric, 0),
                    reverse=True
                )
            else:
                # Use F1 score by default
                suitable_models.sort(
                    key=lambda m: m.get("performance", {}).get("f1_score", 0),
                    reverse=True
                )
        
        elif self.selection_method == "resource":
            # Select model with lowest resource usage
            if task.get("resource_constraint"):
                # Use specific resource constraint if provided
                resource = task["resource_constraint"]
                suitable_models.sort(
                    key=lambda m: m.get("resources", {}).get(resource, 0)
                )
            else:
                # Use memory usage by default
                suitable_models.sort(
                    key=lambda m: m.get("resources", {}).get("memory_usage", 0)
                )
        
        elif self.selection_method == "version":
            # Select most recent version
            suitable_models.sort(
                key=lambda m: m.get("updated_at", 0),
                reverse=True
            )
        
        else:  # "default"
            # Use active model
            return self._get_active_model_id()
        
        # If resource aware, check resource constraints
        if self.resource_aware:
            # Get system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Check each model against resource constraints
            for model in suitable_models:
                model_resources = model.get("resources", {})
                
                # Check if model resource usage is acceptable
                if cpu_percent + model_resources.get("cpu_usage", 0) / 100 < self.resource_thresholds["cpu_percent"] and \
                   memory_percent + model_resources.get("memory_usage", 0) / (psutil.virtual_memory().total / (1024 * 1024)) < self.resource_thresholds["memory_percent"] and \
                   disk_percent < self.resource_thresholds["disk_percent"]:
                    return model["id"]
            
            # If no model meets resource constraints, use default
            logger.warning("No model meets resource constraints, using default")
            return self.default_model
        
        # Return ID of best model
        return suitable_models[0]["id"]
    
    def _get_active_model_id(self) -> str:
        """
        Get the ID of the active model.
        
        Returns:
            Active model ID.
        """
        registry = self._get_registry()
        return registry.get("active_model", self.default_model)
    
    def activate_model(self, model_id: str) -> bool:
        """
        Activate a model by loading it into the Neural Matrix.
        
        Args:
            model_id: ID of the model to activate.
            
        Returns:
            True if successful, False otherwise.
        """
        # Get model from registry
        registry = self._get_registry()
        
        if model_id not in registry["models"]:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        model = registry["models"][model_id]
        model_path = model["path"]
        
        # Check if model files exist
        model_config_file = os.path.join(model_path, "config.json")
        model_weights_file = os.path.join(model_path, "weights.npz")
        
        if not os.path.exists(model_config_file) or not os.path.exists(model_weights_file):
            logger.error(f"Model files not found for {model_id}")
            return False
        
        try:
            # Load configuration
            with open(model_config_file, "r") as f:
                config = json.load(f)
            
            # Load weights
            weights_npz = np.load(model_weights_file)
            weights = {key: weights_npz[key] for key in weights_npz.files}
            
            # Configure neural matrix
            success = self.neural_matrix.configure(config)
            if not success:
                logger.error(f"Failed to configure Neural Matrix with model {model_id}")
                return False
            
            # Set weights
            success = self.neural_matrix.set_weights(weights)
            if not success:
                logger.error(f"Failed to set Neural Matrix weights for model {model_id}")
                return False
            
            # Set active model
            self._set_active_model(model_id)
            
            logger.info(f"Activated model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error activating model {model_id}: {e}")
            return False
    
    def register_model(self, 
                       model_path: str, 
                       model_info: Dict[str, Any]) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to the model directory.
            model_info: Model information.
            
        Returns:
            ID of the registered model, or empty string if registration failed.
        """
        try:
            # Validate model files
            model_config_file = os.path.join(model_path, "config.json")
            model_weights_file = os.path.join(model_path, "weights.npz")
            
            if not os.path.exists(model_config_file) or not os.path.exists(model_weights_file):
                logger.error(f"Model files not found at {model_path}")
                return ""
            
            # Load model config to extract metadata
            with open(model_config_file, "r") as f:
                model_config = json.load(f)
            
            # Generate model ID if not provided
            model_id = model_info.get("id")
            if not model_id:
                # Use name as basis for ID if provided
                if model_info.get("name"):
                    model_id = model_info["name"].lower().replace(" ", "_")
                else:
                    # Generate random ID
                    model_id = f"model_{str(uuid.uuid4())[:8]}"
            
            # Get registry
            registry = self._get_registry()
            
            # Check if model ID already exists
            if model_id in registry["models"]:
                logger.warning(f"Model ID {model_id} already exists, updating")
            
            # Create model entry
            model_entry = {
                "id": model_id,
                "name": model_info.get("name", model_id),
                "description": model_info.get("description", ""),
                "version": model_config.get("version", "v0.1.0"),
                "path": model_path,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
                "task_categories": model_info.get("task_categories", []),
                "performance": model_info.get("performance", {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                }),
                "resources": model_info.get("resources", {
                    "cpu_usage": 0,
                    "memory_usage": 0,
                    "disk_usage": 0
                }),
                "metadata": {
                    "architecture": {
                        "type": model_info.get("architecture_type", "multilayer_perceptron"),
                        "layers": model_config.get("layers", []),
                        "input_dim": model_config.get("input_dim", 0),
                        "output_dim": model_config.get("output_dim", 0),
                        "activation": model_config.get("activation", "relu")
                    },
                    "training": {
                        "epochs": 0,
                        "samples": 0
                    }
                },
                "tags": model_info.get("tags", [])
            }
            
            # Add to registry
            registry["models"][model_id] = model_entry
            registry["updated_at"] = int(time.time())
            
            # Save registry
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"Registered model {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return ""
    
    def unregister_model(self, model_id: str, delete_files: bool = False) -> bool:
        """
        Unregister a model from the registry.
        
        Args:
            model_id: ID of the model to unregister.
            delete_files: Whether to delete model files.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get registry
            registry = self._get_registry()
            
            # Check if model exists
            if model_id not in registry["models"]:
                logger.error(f"Model {model_id} not found in registry")
                return False
            
            # Check if model is active
            if registry.get("active_model") == model_id:
                logger.error(f"Cannot unregister active model {model_id}")
                return False
            
            # Get model path
            model_path = registry["models"][model_id]["path"]
            
            # Remove from registry
            del registry["models"][model_id]
            registry["updated_at"] = int(time.time())
            
            # Save registry
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2)
            
            # Delete files if requested
            if delete_files and os.path.exists(model_path):
                shutil.rmtree(model_path)
            
            logger.info(f"Unregistered model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering model {model_id}: {e}")
            return False
    
    def list_models(self, 
                   filter_category: Optional[str] = None, 
                   sort_by: str = "updated_at", 
                   ascending: bool = False) -> List[Dict[str, Any]]:
        """
        List registered models.
        
        Args:
            filter_category: Filter by task category (optional).
            sort_by: Sort by field.
            ascending: Sort in ascending order.
            
        Returns:
            List of model information dictionaries.
        """
        # Get registry
        registry = self._get_registry()
        
        # Get models
        models = list(registry["models"].values())
        
        # Filter by category if provided
        if filter_category:
            models = [
                model for model in models
                if filter_category in model.get("task_categories", [])
            ]
        
        # Sort models
        if sort_by == "performance":
            # Sort by F1 score
            models.sort(
                key=lambda m: m.get("performance", {}).get("f1_score", 0),
                reverse=not ascending
            )
        elif sort_by == "resources":
            # Sort by memory usage
            models.sort(
                key=lambda m: m.get("resources", {}).get("memory_usage", 0),
                reverse=not ascending
            )
        elif sort_by in ["created_at", "updated_at"]:
            # Sort by timestamp
            models.sort(
                key=lambda m: m.get(sort_by, 0),
                reverse=not ascending
            )
        else:
            # Sort by name
            models.sort(
                key=lambda m: m.get("name", ""),
                reverse=not ascending
            )
        
        return models
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_id: ID of the model (optional, default to active model).
            
        Returns:
            Model information.
        """
        # Get model ID
        if not model_id:
            model_id = self._get_active_model_id()
        
        # Get registry
        registry = self._get_registry()
        
        # Check if model exists
        if model_id not in registry["models"]:
            logger.error(f"Model {model_id} not found in registry")
            return {}
        
        return registry["models"][model_id]
    
    def update_model_info(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        Update model information.
        
        Args:
            model_id: ID of the model to update.
            model_info: New model information.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get registry
            registry = self._get_registry()
            
            # Check if model exists
            if model_id not in registry["models"]:
                logger.error(f"Model {model_id} not found in registry")
                return False
            
            # Update model info
            model = registry["models"][model_id]
            
            # Update fields
            for key, value in model_info.items():
                if key not in ["id", "path", "created_at"]:
                    if key in ["performance", "resources", "metadata"] and isinstance(value, dict):
                        # Merge dictionaries
                        if key not in model:
                            model[key] = {}
                        model[key].update(value)
                    else:
                        model[key] = value
            
            # Update timestamp
            model["updated_at"] = int(time.time())
            registry["updated_at"] = int(time.time())
            
            # Save registry
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"Updated model info for {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model info for {model_id}: {e}")
            return False
    
    def clone_model(self, 
                   source_model_id: str, 
                   target_model_id: Optional[str] = None, 
                   model_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Clone a model.
        
        Args:
            source_model_id: ID of the source model.
            target_model_id: ID for the new model (optional).
            model_info: Additional model information (optional).
            
        Returns:
            ID of the new model, or empty string if cloning failed.
        """
        try:
            # Get registry
            registry = self._get_registry()
            
            # Check if source model exists
            if source_model_id not in registry["models"]:
                logger.error(f"Source model {source_model_id} not found in registry")
                return ""
            
            # Get source model
            source_model = registry["models"][source_model_id]
            source_path = source_model["path"]
            
            # Generate target model ID if not provided
            if not target_model_id:
                target_model_id = f"{source_model_id}_clone_{str(uuid.uuid4())[:8]}"
            
            # Check if target model ID already exists
            if target_model_id in registry["models"]:
                logger.error(f"Target model ID {target_model_id} already exists")
                return ""
            
            # Create target directory
            target_path = os.path.join(self.models_dir, target_model_id)
            os.makedirs(target_path, exist_ok=True)
            
            # Copy model files
            for filename in ["config.json", "weights.npz", "metadata.json"]:
                source_file = os.path.join(source_path, filename)
                target_file = os.path.join(target_path, filename)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
            
            # Create model entry
            model_entry = source_model.copy()
            model_entry.update({
                "id": target_model_id,
                "name": model_info.get("name", f"{source_model['name']} (Clone)"),
                "description": model_info.get("description", f"Clone of {source_model['name']}"),
                "path": target_path,
                "created_at": int(time.time()),
                "updated_at": int(time.time())
            })
            
            # Update with provided model info
            if model_info:
                for key, value in model_info.items():
                    if key not in ["id", "path", "created_at"]:
                        model_entry[key] = value
            
            # Add to registry
            registry["models"][target_model_id] = model_entry
            registry["updated_at"] = int(time.time())
            
            # Save registry
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"Cloned model {source_model_id} to {target_model_id}")
            return target_model_id
            
        except Exception as e:
            logger.error(f"Error cloning model {source_model_id}: {e}")
            return ""
    
    def export_model(self, model_id: str, output_path: str, format: str = "directory") -> bool:
        """
        Export a model.
        
        Args:
            model_id: ID of the model to export.
            output_path: Path to export the model to.
            format: Export format ('directory', 'zip', 'tar.gz').
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get registry
            registry = self._get_registry()
            
            # Check if model exists
            if model_id not in registry["models"]:
                logger.error(f"Model {model_id} not found in registry")
                return False
            
            # Get model
            model = registry["models"][model_id]
            model_path = model["path"]
            
            # Export based on format
            if format == "directory":
                # Create output directory
                os.makedirs(output_path, exist_ok=True)
                
                # Copy model files
                for filename in ["config.json", "weights.npz", "metadata.json"]:
                    source_file = os.path.join(model_path, filename)
                    target_file = os.path.join(output_path, filename)
                    
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, target_file)
                
                # Create export info file
                with open(os.path.join(output_path, "export_info.json"), "w") as f:
                    json.dump({
                        "model_id": model_id,
                        "model_name": model["name"],
                        "model_description": model["description"],
                        "model_version": model["version"],
                        "exported_at": int(time.time()),
                        "format": format,
                        "task_categories": model.get("task_categories", []),
                        "performance": model.get("performance", {}),
                        "resources": model.get("resources", {}),
                        "tags": model.get("tags", [])
                    }, f, indent=2)
                
            elif format == "zip":
                # Create a zip file
                import zipfile
                
                with zipfile.ZipFile(output_path, "w") as zipf:
                    # Add model files
                    for filename in ["config.json", "weights.npz", "metadata.json"]:
                        source_file = os.path.join(model_path, filename)
                        
                        if os.path.exists(source_file):
                            zipf.write(source_file, os.path.basename(source_file))
                    
                    # Add export info
                    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                        json.dump({
                            "model_id": model_id,
                            "model_name": model["name"],
                            "model_description": model["description"],
                            "model_version": model["version"],
                            "exported_at": int(time.time()),
                            "format": format,
                            "task_categories": model.get("task_categories", []),
                            "performance": model.get("performance", {}),
                            "resources": model.get("resources", {}),
                            "tags": model.get("tags", [])
                        }, f, indent=2)
                    
                    zipf.write(f.name, "export_info.json")
                    os.unlink(f.name)
                
            elif format == "tar.gz":
                # Create a tar.gz file
                import tarfile
                
                with tarfile.open(output_path, "w:gz") as tar:
                    # Add model files
                    for filename in ["config.json", "weights.npz", "metadata.json"]:
                        source_file = os.path.join(model_path, filename)
                        
                        if os.path.exists(source_file):
                            tar.add(source_file, arcname=os.path.basename(source_file))
                    
                    # Add export info
                    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                        json.dump({
                            "model_id": model_id,
                            "model_name": model["name"],
                            "model_description": model["description"],
                            "model_version": model["version"],
                            "exported_at": int(time.time()),
                            "format": format,
                            "task_categories": model.get("task_categories", []),
                            "performance": model.get("performance", {}),
                            "resources": model.get("resources", {}),
                            "tags": model.get("tags", [])
                        }, f, indent=2)
                    
                    tar.add(f.name, arcname="export_info.json")
                    os.unlink(f.name)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported model {model_id} to {output_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model {model_id}: {e}")
            return False
    
    def import_model(self, model_path: str, model_id: Optional[str] = None) -> str:
        """
        Import a model.
        
        Args:
            model_path: Path to the model to import.
            model_id: ID to use for the imported model (optional).
            
        Returns:
            ID of the imported model, or empty string if import failed.
        """
        try:
            # Determine import format
            if os.path.isdir(model_path):
                # Import from directory
                model_config_file = os.path.join(model_path, "config.json")
                model_weights_file = os.path.join(model_path, "weights.npz")
                model_metadata_file = os.path.join(model_path, "metadata.json")
                model_export_info_file = os.path.join(model_path, "export_info.json")
                
                if not os.path.exists(model_config_file) or not os.path.exists(model_weights_file):
                    logger.error(f"Model files not found at {model_path}")
                    return ""
                
                # Generate model ID if not provided
                if not model_id:
                    if os.path.exists(model_export_info_file):
                        # Use ID from export info if available
                        with open(model_export_info_file, "r") as f:
                            export_info = json.load(f)
                        
                        model_id = export_info.get("model_id", "")
                    
                    if not model_id:
                        # Generate random ID
                        model_id = f"imported_{str(uuid.uuid4())[:8]}"
                
                # Create model directory
                target_path = os.path.join(self.models_dir, model_id)
                os.makedirs(target_path, exist_ok=True)
                
                # Copy model files
                for filename in ["config.json", "weights.npz", "metadata.json"]:
                    source_file = os.path.join(model_path, filename)
                    target_file = os.path.join(target_path, filename)
                    
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, target_file)
                
                # Load model information
                with open(model_config_file, "r") as f:
                    model_config = json.load(f)
                
                # Create model info
                model_info = {
                    "name": "Imported Model",
                    "description": "Imported Neural Matrix model",
                    "task_categories": ["bug_detection", "code_analysis", "fix_generation", "verification"],
                    "performance": {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0
                    },
                    "resources": {
                        "cpu_usage": 0,
                        "memory_usage": 0,
                        "disk_usage": 0
                    },
                    "tags": ["imported"]
                }
                
                # Update with export info if available
                if os.path.exists(model_export_info_file):
                    with open(model_export_info_file, "r") as f:
                        export_info = json.load(f)
                    
                    model_info.update({
                        "name": export_info.get("model_name", model_info["name"]),
                        "description": export_info.get("model_description", model_info["description"]),
                        "task_categories": export_info.get("task_categories", model_info["task_categories"]),
                        "performance": export_info.get("performance", model_info["performance"]),
                        "resources": export_info.get("resources", model_info["resources"]),
                        "tags": export_info.get("tags", model_info["tags"])
                    })
                
                # Register model
                registered_model_id = self.register_model(target_path, model_info)
                
                if not registered_model_id:
                    logger.error(f"Failed to register imported model")
                    return ""
                
                logger.info(f"Imported model from {model_path} as {registered_model_id}")
                return registered_model_id
                
            elif model_path.endswith(".zip"):
                # Import from zip file
                import zipfile
                
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract zip file
                    with zipfile.ZipFile(model_path, "r") as zipf:
                        zipf.extractall(temp_dir)
                    
                    # Import from extracted directory
                    return self.import_model(temp_dir, model_id)
                
            elif model_path.endswith(".tar.gz") or model_path.endswith(".tgz"):
                # Import from tar.gz file
                import tarfile
                
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract tar.gz file
                    with tarfile.open(model_path, "r:gz") as tar:
                        tar.extractall(temp_dir)
                    
                    # Import from extracted directory
                    return self.import_model(temp_dir, model_id)
                
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return ""
                
        except Exception as e:
            logger.error(f"Error importing model: {e}")
            return ""

# Main entry point
def select_and_activate_model(task: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Select and activate the best model for a given task.
    
    Args:
        task: Task information.
        config_path: Path to configuration file (optional).
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Load configuration if provided
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        
        # Create model selection
        model_selection = NeuralMatrixModelSelection(config)
        
        # Select model
        model_id = model_selection.select_model(task)
        
        # Activate model
        success = model_selection.activate_model(model_id)
        
        return success
    
    except Exception as e:
        logger.error(f"Error selecting and activating model: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Matrix Model Selection")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--info", help="Get information about a model")
    parser.add_argument("--activate", help="Activate a model")
    parser.add_argument("--select", help="Select a model for a task category")
    parser.add_argument("--export", help="Export a model")
    parser.add_argument("--export-format", choices=["directory", "zip", "tar.gz"], default="directory", help="Export format")
    parser.add_argument("--output", help="Output path for export")
    parser.add_argument("--import", dest="import_model", help="Import a model")
    parser.add_argument("--clone", help="Clone a model")
    parser.add_argument("--target", help="Target model ID for clone")
    parser.add_argument("--unregister", help="Unregister a model")
    parser.add_argument("--delete-files", action="store_true", help="Delete model files when unregistering")
    
    args = parser.parse_args()
    
    # Create model selection
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    
    model_selection = NeuralMatrixModelSelection(config)
    
    if args.list:
        # List models
        models = model_selection.list_models()
        print(f"Found {len(models)} models:")
        for model in models:
            active = "*" if model["id"] == model_selection._get_active_model_id() else " "
            print(f"  {active} {model['id']}: {model['name']} ({model['version']})")
            print(f"      Description: {model['description']}")
            print(f"      Task categories: {', '.join(model.get('task_categories', []))}")
            print(f"      Tags: {', '.join(model.get('tags', []))}")
            print(f"      Performance: F1={model.get('performance', {}).get('f1_score', 0):.2f}")
            print()
    
    elif args.info:
        # Get model info
        model_info = model_selection.get_model_info(args.info)
        
        if model_info:
            print(f"Model: {model_info['name']} ({model_info['id']})")
            print(f"Version: {model_info['version']}")
            print(f"Description: {model_info['description']}")
            print(f"Path: {model_info['path']}")
            print(f"Created: {time.ctime(model_info['created_at'])}")
            print(f"Updated: {time.ctime(model_info['updated_at'])}")
            print(f"Task categories: {', '.join(model_info.get('task_categories', []))}")
            print(f"Tags: {', '.join(model_info.get('tags', []))}")
            
            print("\nPerformance:")
            performance = model_info.get('performance', {})
            for metric, value in performance.items():
                print(f"  {metric}: {value:.4f}")
            
            print("\nResources:")
            resources = model_info.get('resources', {})
            for resource, value in resources.items():
                print(f"  {resource}: {value}")
            
            print("\nArchitecture:")
            architecture = model_info.get('metadata', {}).get('architecture', {})
            for key, value in architecture.items():
                print(f"  {key}: {value}")
        else:
            print(f"Model {args.info} not found")
    
    elif args.activate:
        # Activate model
        success = model_selection.activate_model(args.activate)
        
        if success:
            print(f"Activated model {args.activate}")
        else:
            print(f"Failed to activate model {args.activate}")
            sys.exit(1)
    
    elif args.select:
        # Select model for task category
        model_id = model_selection.select_model({"category": args.select})
        print(f"Selected model {model_id} for task category {args.select}")
        
        # Activate selected model
        success = model_selection.activate_model(model_id)
        
        if success:
            print(f"Activated model {model_id}")
        else:
            print(f"Failed to activate model {model_id}")
            sys.exit(1)
    
    elif args.export:
        # Check output path
        if not args.output:
            print("Error: Output path required for export")
            sys.exit(1)
        
        # Export model
        success = model_selection.export_model(args.export, args.output, args.export_format)
        
        if success:
            print(f"Exported model {args.export} to {args.output} in {args.export_format} format")
        else:
            print(f"Failed to export model {args.export}")
            sys.exit(1)
    
    elif args.import_model:
        # Import model
        model_id = model_selection.import_model(args.import_model)
        
        if model_id:
            print(f"Imported model as {model_id}")
        else:
            print(f"Failed to import model from {args.import_model}")
            sys.exit(1)
    
    elif args.clone:
        # Clone model
        cloned_model_id = model_selection.clone_model(args.clone, args.target)
        
        if cloned_model_id:
            print(f"Cloned model {args.clone} to {cloned_model_id}")
        else:
            print(f"Failed to clone model {args.clone}")
            sys.exit(1)
    
    elif args.unregister:
        # Unregister model
        success = model_selection.unregister_model(args.unregister, args.delete_files)
        
        if success:
            print(f"Unregistered model {args.unregister}")
        else:
            print(f"Failed to unregister model {args.unregister}")
            sys.exit(1)
    
    else:
        # Show active model
        active_model = model_selection._get_active_model()
        
        print(f"Active model: {active_model.get('name', 'Unknown')} ({active_model.get('id', 'Unknown')})")
        print(f"Version: {active_model.get('version', 'Unknown')}")
        print(f"Description: {active_model.get('description', 'Unknown')}")
