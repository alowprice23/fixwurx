#!/usr/bin/env python3
"""
neural_matrix_configuration.py
───────────────────────────────
Implements advanced configuration for the Neural Matrix system.

This module provides specialized configuration management for Neural Matrix
parameters, including model selection, weight management, learning rate
optimization, and integration with the core system configuration.
"""

import os
import json
import yaml
import logging
import math
import numpy as np
import copy
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path

# Import system configuration
from system_configuration import get_config, ConfigurationError, Configuration

# Configure logging
logger = logging.getLogger("NeuralMatrixConfiguration")

# Default Neural Matrix configuration directory and files
DEFAULT_MATRIX_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_matrix/config")
DEFAULT_MATRIX_CONFIG_FILE = os.path.join(DEFAULT_MATRIX_CONFIG_DIR, "neural_matrix_config.yaml")
DEFAULT_WEIGHTS_DIR = os.path.join(DEFAULT_MATRIX_CONFIG_DIR, "weights")

# Neural Matrix models
MATRIX_MODELS = {
    "lightweight": {
        "description": "Lightweight model for resource-constrained environments",
        "layers": 3,
        "neurons_per_layer": [32, 16, 8],
        "activation": "relu",
        "dropout": 0.2,
        "optimizer": "adam",
        "batch_size": 16,
        "max_memory_mb": 128
    },
    "standard": {
        "description": "Standard model with balanced performance and resource usage",
        "layers": 5,
        "neurons_per_layer": [64, 48, 32, 24, 16],
        "activation": "relu",
        "dropout": 0.3,
        "optimizer": "adam",
        "batch_size": 32,
        "max_memory_mb": 256
    },
    "advanced": {
        "description": "Advanced model with highest accuracy and performance",
        "layers": 7,
        "neurons_per_layer": [128, 96, 64, 48, 32, 24, 16],
        "activation": "relu",
        "dropout": 0.4,
        "optimizer": "adam",
        "batch_size": 64,
        "max_memory_mb": 512
    }
}

# Configuration schema for Neural Matrix
NEURAL_MATRIX_CONFIG_SCHEMA = {
    "model": {
        "type": {
            "type": "string",
            "enum": ["lightweight", "standard", "advanced", "custom"],
            "default": "standard",
            "description": "Neural Matrix model type"
        },
        "custom_config": {
            "type": "object",
            "default": None,
            "description": "Custom model configuration (if type is 'custom')"
        },
        "initialization": {
            "type": "string",
            "enum": ["random", "xavier", "he", "orthogonal"],
            "default": "he",
            "description": "Weight initialization method"
        },
        "regularization": {
            "type": "string",
            "enum": ["none", "l1", "l2", "elastic_net"],
            "default": "l2",
            "description": "Regularization method to prevent overfitting"
        },
        "pruning_threshold": {
            "type": "number",
            "default": 0.01,
            "description": "Threshold for pruning low-weight connections"
        }
    },
    "training": {
        "learning_rate": {
            "type": "number",
            "default": 0.001,
            "description": "Base learning rate for training"
        },
        "adaptive_learning": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use adaptive learning rate"
        },
        "min_learning_rate": {
            "type": "number",
            "default": 0.0001,
            "description": "Minimum learning rate"
        },
        "max_learning_rate": {
            "type": "number",
            "default": 0.01,
            "description": "Maximum learning rate"
        },
        "momentum": {
            "type": "number",
            "default": 0.9,
            "description": "Momentum coefficient for gradient descent"
        },
        "weight_decay": {
            "type": "number",
            "default": 0.0001,
            "description": "Weight decay for regularization"
        },
        "batch_size": {
            "type": "integer",
            "default": 32,
            "description": "Batch size for training"
        },
        "max_epochs": {
            "type": "integer",
            "default": 1000,
            "description": "Maximum number of training epochs"
        },
        "early_stopping": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use early stopping"
        },
        "patience": {
            "type": "integer",
            "default": 10,
            "description": "Number of epochs with no improvement before early stopping"
        }
    },
    "architecture": {
        "activation": {
            "type": "string",
            "enum": ["relu", "sigmoid", "tanh", "leaky_relu", "elu"],
            "default": "relu",
            "description": "Activation function for hidden layers"
        },
        "output_activation": {
            "type": "string",
            "enum": ["softmax", "sigmoid", "tanh", "linear"],
            "default": "softmax",
            "description": "Activation function for output layer"
        },
        "dropout_rate": {
            "type": "number",
            "default": 0.3,
            "description": "Dropout rate for regularization"
        },
        "batch_normalization": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use batch normalization"
        },
        "skip_connections": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use skip connections"
        },
        "layer_normalization": {
            "type": "boolean",
            "default": False,
            "description": "Whether to use layer normalization"
        }
    },
    "features": {
        "attention_mechanism": {
            "type": "boolean",
            "default": False,
            "description": "Whether to use attention mechanism"
        },
        "feature_selection": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use feature selection"
        },
        "pattern_recognition": {
            "type": "string",
            "enum": ["basic", "advanced", "expert"],
            "default": "advanced",
            "description": "Level of pattern recognition capabilities"
        },
        "context_aware": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use context-aware learning"
        },
        "transfer_learning": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use transfer learning from previous solutions"
        }
    },
    "inference": {
        "confidence_threshold": {
            "type": "number",
            "default": 0.7,
            "description": "Confidence threshold for predictions"
        },
        "max_inference_time_ms": {
            "type": "integer",
            "default": 500,
            "description": "Maximum inference time in milliseconds"
        },
        "fallback_strategy": {
            "type": "string",
            "enum": ["conservative", "aggressive", "balanced"],
            "default": "balanced",
            "description": "Strategy to use when confidence is below threshold"
        },
        "batch_inference": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use batch inference"
        }
    },
    "storage": {
        "weights_dir": {
            "type": "string",
            "default": DEFAULT_WEIGHTS_DIR,
            "description": "Directory for storing neural weights"
        },
        "checkpoint_interval": {
            "type": "integer",
            "default": 10,
            "description": "Interval (in epochs) for checkpointing weights"
        },
        "max_checkpoints": {
            "type": "integer",
            "default": 5,
            "description": "Maximum number of checkpoints to keep"
        },
        "save_best_only": {
            "type": "boolean",
            "default": True,
            "description": "Whether to save only the best weights"
        },
        "compress_weights": {
            "type": "boolean",
            "default": True,
            "description": "Whether to compress stored weights"
        }
    },
    "optimization": {
        "auto_tune": {
            "type": "boolean",
            "default": True,
            "description": "Whether to automatically tune hyperparameters"
        },
        "search_strategy": {
            "type": "string",
            "enum": ["grid", "random", "bayesian", "genetic"],
            "default": "bayesian",
            "description": "Strategy for hyperparameter search"
        },
        "max_trials": {
            "type": "integer",
            "default": 10,
            "description": "Maximum number of hyperparameter optimization trials"
        },
        "evaluation_metric": {
            "type": "string",
            "enum": ["accuracy", "f1", "precision", "recall", "auc"],
            "default": "f1",
            "description": "Metric to use for evaluation during optimization"
        },
        "resource_constraint": {
            "type": "string",
            "enum": ["none", "memory", "time", "both"],
            "default": "both",
            "description": "Resource constraint for optimization"
        }
    },
    "advanced": {
        "gradient_clipping": {
            "type": "boolean",
            "default": True,
            "description": "Whether to use gradient clipping"
        },
        "clip_value": {
            "type": "number",
            "default": 5.0,
            "description": "Value for gradient clipping"
        },
        "mixed_precision": {
            "type": "boolean",
            "default": False,
            "description": "Whether to use mixed precision training"
        },
        "dynamic_quantization": {
            "type": "boolean",
            "default": False,
            "description": "Whether to use dynamic quantization"
        },
        "sparse_inference": {
            "type": "boolean",
            "default": False,
            "description": "Whether to use sparse inference"
        },
        "debug_mode": {
            "type": "boolean",
            "default": False,
            "description": "Whether to enable debug mode"
        }
    }
}

class NeuralMatrixConfigurationError(Exception):
    """Exception raised for Neural Matrix configuration errors."""
    pass

class NeuralMatrixConfiguration:
    """
    Neural Matrix configuration manager.
    
    This class manages advanced configuration for the Neural Matrix system,
    including model parameters, training settings, architecture, features,
    inference, storage, and optimization.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the Neural Matrix configuration manager.
        
        Args:
            config_file: Path to the Neural Matrix configuration file. If not provided,
                the default configuration file will be used.
        """
        self._config = {}
        self._default_config = self._generate_default_config()
        self._change_listeners = []
        self._system_config = get_config()
        
        # Load configuration
        if config_file:
            self._config_file = config_file
        else:
            neural_matrix_section = self._system_config.get_section("neural_matrix")
            custom_config_path = neural_matrix_section.get("config_path")
            
            if custom_config_path and os.path.exists(custom_config_path):
                self._config_file = custom_config_path
            else:
                self._config_file = DEFAULT_MATRIX_CONFIG_FILE
        
        self.load()
        
        logger.info(f"Neural Matrix configuration initialized from {self._config_file}")
    
    def _generate_default_config(self) -> Dict[str, Any]:
        """
        Generate the default configuration from the schema.
        
        Returns:
            Default configuration dictionary.
        """
        default_config = {}
        
        for section, section_schema in NEURAL_MATRIX_CONFIG_SCHEMA.items():
            default_config[section] = {}
            
            for key, key_schema in section_schema.items():
                default_config[section][key] = key_schema.get("default")
        
        return default_config
    
    def load(self) -> None:
        """
        Load configuration from the configuration file.
        
        If the configuration file doesn't exist, the default configuration will be used.
        """
        try:
            # Start with default configuration
            self._config = copy.deepcopy(self._default_config)
            
            # If configuration file exists, load it
            if os.path.exists(self._config_file):
                try:
                    file_config = self._load_config_file(self._config_file)
                    self._merge_config(self._config, file_config)
                    logger.debug(f"Loaded Neural Matrix configuration from {self._config_file}")
                except Exception as e:
                    logger.warning(f"Failed to load Neural Matrix configuration: {e}")
            
            # Apply overrides from system configuration
            self._apply_system_config_overrides()
            
            # Validate configuration
            self.validate()
            
            # Initialize model configuration
            self._initialize_model_config()
        except Exception as e:
            logger.error(f"Failed to load Neural Matrix configuration: {e}")
            logger.info("Using default Neural Matrix configuration")
            self._config = copy.deepcopy(self._default_config)
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file.
            
        Returns:
            Configuration dictionary.
            
        Raises:
            NeuralMatrixConfigurationError: If the file cannot be loaded.
        """
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    raise NeuralMatrixConfigurationError(f"Unsupported configuration file format: {file_path}")
        except Exception as e:
            raise NeuralMatrixConfigurationError(f"Failed to load configuration file {file_path}: {e}")
    
    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """
        Merge override configuration into base configuration.
        
        Args:
            base_config: Base configuration to merge into.
            override_config: Override configuration to merge from.
        """
        for section, section_config in override_config.items():
            if section not in base_config:
                base_config[section] = {}
            
            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    base_config[section][key] = value
    
    def _apply_system_config_overrides(self) -> None:
        """
        Apply overrides from system configuration.
        
        This method reads relevant settings from the system configuration and
        applies them to the Neural Matrix configuration.
        """
        neural_matrix_section = self._system_config.get_section("neural_matrix")
        
        # Apply model type from system configuration
        model_type = neural_matrix_section.get("model_type")
        if model_type and model_type in MATRIX_MODELS:
            self._config["model"]["type"] = model_type
        
        # Apply learning rate from system configuration
        learning_rate = neural_matrix_section.get("learning_rate")
        if learning_rate is not None:
            self._config["training"]["learning_rate"] = learning_rate
        
        # Apply weight decay from system configuration
        weight_decay = neural_matrix_section.get("weight_decay")
        if weight_decay is not None:
            self._config["training"]["weight_decay"] = weight_decay
        
        # Apply storage path from system configuration
        storage_path = neural_matrix_section.get("storage_path")
        if storage_path:
            # Expand ~ to user's home directory
            storage_path = os.path.expanduser(storage_path)
            weights_dir = os.path.join(storage_path, "weights")
            self._config["storage"]["weights_dir"] = weights_dir
    
    def validate(self) -> None:
        """
        Validate the configuration against the schema.
        
        Raises:
            NeuralMatrixConfigurationError: If the configuration is invalid.
        """
        for section, section_schema in NEURAL_MATRIX_CONFIG_SCHEMA.items():
            if section not in self._config:
                self._config[section] = {}
            
            for key, key_schema in section_schema.items():
                # If key doesn't exist, use default
                if key not in self._config[section]:
                    self._config[section][key] = key_schema.get("default")
                    continue
                
                # Validate type
                value = self._config[section][key]
                expected_type = key_schema.get("type")
                
                # Skip validation for None values
                if value is None:
                    continue
                
                if expected_type == "string" and not isinstance(value, str):
                    logger.warning(f"Invalid type for {section}.{key}: expected string, got {type(value)}")
                    self._config[section][key] = key_schema.get("default")
                
                elif expected_type == "integer" and not isinstance(value, int):
                    logger.warning(f"Invalid type for {section}.{key}: expected integer, got {type(value)}")
                    self._config[section][key] = key_schema.get("default")
                
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    logger.warning(f"Invalid type for {section}.{key}: expected number, got {type(value)}")
                    self._config[section][key] = key_schema.get("default")
                
                elif expected_type == "boolean" and not isinstance(value, bool):
                    logger.warning(f"Invalid type for {section}.{key}: expected boolean, got {type(value)}")
                    self._config[section][key] = key_schema.get("default")
                
                elif expected_type == "object" and value is not None and not isinstance(value, dict):
                    logger.warning(f"Invalid type for {section}.{key}: expected object, got {type(value)}")
                    self._config[section][key] = key_schema.get("default")
                
                # Validate enum
                if "enum" in key_schema and value not in key_schema["enum"]:
                    logger.warning(f"Invalid value for {section}.{key}: {value} not in {key_schema['enum']}")
                    self._config[section][key] = key_schema.get("default")
        
        # Additional validation for specific parameters
        
        # Learning rate must be positive
        learning_rate = self._config["training"]["learning_rate"]
        if learning_rate <= 0:
            logger.warning(f"Invalid learning rate: {learning_rate}. Must be positive.")
            self._config["training"]["learning_rate"] = NEURAL_MATRIX_CONFIG_SCHEMA["training"]["learning_rate"]["default"]
        
        # Weight decay must be non-negative
        weight_decay = self._config["training"]["weight_decay"]
        if weight_decay < 0:
            logger.warning(f"Invalid weight decay: {weight_decay}. Must be non-negative.")
            self._config["training"]["weight_decay"] = NEURAL_MATRIX_CONFIG_SCHEMA["training"]["weight_decay"]["default"]
        
        # Batch size must be positive
        batch_size = self._config["training"]["batch_size"]
        if batch_size <= 0:
            logger.warning(f"Invalid batch size: {batch_size}. Must be positive.")
            self._config["training"]["batch_size"] = NEURAL_MATRIX_CONFIG_SCHEMA["training"]["batch_size"]["default"]
        
        # Patience must be non-negative
        patience = self._config["training"]["patience"]
        if patience < 0:
            logger.warning(f"Invalid patience: {patience}. Must be non-negative.")
            self._config["training"]["patience"] = NEURAL_MATRIX_CONFIG_SCHEMA["training"]["patience"]["default"]
        
        # Dropout rate must be between 0 and 1
        dropout_rate = self._config["architecture"]["dropout_rate"]
        if dropout_rate < 0 or dropout_rate > 1:
            logger.warning(f"Invalid dropout rate: {dropout_rate}. Must be between 0 and 1.")
            self._config["architecture"]["dropout_rate"] = NEURAL_MATRIX_CONFIG_SCHEMA["architecture"]["dropout_rate"]["default"]
        
        # Confidence threshold must be between 0 and 1
        confidence_threshold = self._config["inference"]["confidence_threshold"]
        if confidence_threshold < 0 or confidence_threshold > 1:
            logger.warning(f"Invalid confidence threshold: {confidence_threshold}. Must be between 0 and 1.")
            self._config["inference"]["confidence_threshold"] = NEURAL_MATRIX_CONFIG_SCHEMA["inference"]["confidence_threshold"]["default"]
    
    def _initialize_model_config(self) -> None:
        """
        Initialize model configuration based on the selected model type.
        
        This method loads the appropriate model configuration based on the
        selected model type and merges it with the current configuration.
        """
        model_type = self._config["model"]["type"]
        
        if model_type == "custom":
            # Custom model configuration is already in the configuration
            return
        
        if model_type in MATRIX_MODELS:
            model_config = MATRIX_MODELS[model_type]
            
            # Apply model configuration to architecture
            if "neurons_per_layer" in model_config:
                self._config["model"]["layers"] = model_config["layers"]
                self._config["model"]["neurons_per_layer"] = model_config["neurons_per_layer"]
            
            if "activation" in model_config:
                self._config["architecture"]["activation"] = model_config["activation"]
            
            if "dropout" in model_config:
                self._config["architecture"]["dropout_rate"] = model_config["dropout"]
            
            if "optimizer" in model_config:
                self._config["training"]["optimizer"] = model_config["optimizer"]
            
            if "batch_size" in model_config:
                self._config["training"]["batch_size"] = model_config["batch_size"]
            
            if "max_memory_mb" in model_config:
                self._config["optimization"]["max_memory_mb"] = model_config["max_memory_mb"]
        else:
            logger.warning(f"Unknown model type: {model_type}. Using default model configuration.")
    
    def save(self, file_path: Optional[str] = None) -> None:
        """
        Save the configuration to a file.
        
        Args:
            file_path: Path to save the configuration to. If not provided,
                the current configuration file will be used.
                
        Raises:
            NeuralMatrixConfigurationError: If the configuration cannot be saved.
        """
        try:
            save_path = file_path or self._config_file
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.json'):
                    json.dump(self._config, f, indent=2)
                elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(self._config, f, default_flow_style=False)
                else:
                    raise NeuralMatrixConfigurationError(f"Unsupported configuration file format: {save_path}")
            
            logger.info(f"Neural Matrix configuration saved to {save_path}")
        except Exception as e:
            raise NeuralMatrixConfigurationError(f"Failed to save configuration: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section.
            key: Configuration key.
            default: Default value to return if the key doesn't exist.
            
        Returns:
            Configuration value or default if not found.
        """
        if section in self._config and key in self._config[section]:
            return self._config[section][key]
        return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section.
            key: Configuration key.
            value: Configuration value.
            
        Raises:
            NeuralMatrixConfigurationError: If the section or key doesn't exist in the schema.
        """
        # Check if section and key exist in schema
        if section not in NEURAL_MATRIX_CONFIG_SCHEMA:
            raise NeuralMatrixConfigurationError(f"Unknown configuration section: {section}")
        
        if key not in NEURAL_MATRIX_CONFIG_SCHEMA[section]:
            raise NeuralMatrixConfigurationError(f"Unknown configuration key: {section}.{key}")
        
        # Create section if it doesn't exist
        if section not in self._config:
            self._config[section] = {}
        
        # Set value and validate
        old_value = self._config[section].get(key)
        self._config[section][key] = value
        
        try:
            self.validate()
        except Exception as e:
            # Restore old value if validation fails
            self._config[section][key] = old_value
            raise NeuralMatrixConfigurationError(f"Invalid configuration value: {e}")
        
        # Notify listeners of change
        if old_value != value:
            self._notify_change(section, key, value, old_value)
        
        # Special handling for model type changes
        if section == "model" and key == "type" and old_value != value:
            self._initialize_model_config()
    
    def add_change_listener(self, listener: Callable[[str, str, Any, Any], None]) -> None:
        """
        Add a listener for configuration changes.
        
        Args:
            listener: Function to call when configuration changes. The function
                should accept four arguments: section, key, new_value, old_value.
        """
        if listener not in self._change_listeners:
            self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[str, str, Any, Any], None]) -> None:
        """
        Remove a listener for configuration changes.
        
        Args:
            listener: Function to remove.
        """
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def _notify_change(self, section: str, key: str, value: Any, old_value: Any) -> None:
        """
        Notify listeners of a configuration change.
        
        Args:
            section: Configuration section.
            key: Configuration key.
            value: New configuration value.
            old_value: Old configuration value.
        """
        for listener in self._change_listeners:
            try:
                listener(section, key, value, old_value)
            except Exception as e:
                logger.warning(f"Error in configuration change listener: {e}")
    
    def reset(self, section: Optional[str] = None, key: Optional[str] = None) -> None:
        """
        Reset configuration to default values.
        
        Args:
            section: Configuration section to reset. If not provided, all sections
                will be reset.
            key: Configuration key to reset. If not provided, all keys in the
                section will be reset.
        """
        if section is None:
            # Reset all sections
            self._config = copy.deepcopy(self._default_config)
            self._initialize_model_config()
        elif section in self._config:
            if key is None:
                # Reset entire section
                self._config[section] = copy.deepcopy(self._default_config.get(section, {}))
                
                # Re-initialize model config if resetting model section
                if section == "model":
                    self._initialize_model_config()
            elif key in self._config[section]:
                # Reset specific key
                default_value = self._default_config.get(section, {}).get(key)
                old_value = self._config[section][key]
                self._config[section][key] = default_value
                
                # Notify listeners of change
                if old_value != default_value:
                    self._notify_change(section, key, default_value, old_value)
                
                # Re-initialize model config if resetting model type
                if section == "model" and key == "type":
                    self._initialize_model_config()
    
    def get_schema(self, section: Optional[str] = None, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the configuration schema.
        
        Args:
            section: Configuration section to get schema for. If not provided,
                the entire schema will be returned.
            key: Configuration key to get schema for. If not provided, the
                entire section schema will be returned.
                
        Returns:
            Configuration schema.
        """
        if section is None:
            return NEURAL_MATRIX_CONFIG_SCHEMA
        
        if section in NEURAL_MATRIX_CONFIG_SCHEMA:
            if key is None:
                return NEURAL_MATRIX_CONFIG_SCHEMA[section]
            
            if key in NEURAL_MATRIX_CONFIG_SCHEMA[section]:
                return NEURAL_MATRIX_CONFIG_SCHEMA[section][key]
        
        return {}
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Complete configuration dictionary.
        """
        return copy.deepcopy(self._config)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section.
        
        Args:
            section: Configuration section.
            
        Returns:
            Configuration section dictionary.
        """
        return copy.deepcopy(self._config.get(section, {}))
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the current model configuration.
        
        Returns:
            Model configuration dictionary with all relevant parameters.
        """
        model_config = {}
        
        # Basic model information
        model_type = self._config["model"]["type"]
        model_config["type"] = model_type
        
        if model_type == "custom":
            # Use custom configuration
            custom_config = self._config["model"]["custom_config"]
            if custom_config:
                model_config.update(custom_config)
        else:
            # Use predefined model configuration
            if model_type in MATRIX_MODELS:
                predefined_config = MATRIX_MODELS[model_type]
                model_config.update(predefined_config)
        
        # Add configuration from other sections
        model_config["initialization"] = self._config["model"]["initialization"]
        model_config["regularization"] = self._config["model"]["regularization"]
        model_config["pruning_threshold"] = self._config["model"]["pruning_threshold"]
        
        # Architecture
        model_config["activation"] = self._config["architecture"]["activation"]
        model_config["output_activation"] = self._config["architecture"]["output_activation"]
        model_config["dropout_rate"] = self._config["architecture"]["dropout_rate"]
        model_config["batch_normalization"] = self._config["architecture"]["batch_normalization"]
        model_config["skip_connections"] = self._config["architecture"]["skip_connections"]
        
        # Training
        model_config["learning_rate"] = self._config["training"]["learning_rate"]
        model_config["adaptive_learning"] = self._config["training"]["adaptive_learning"]
        model_config["weight_decay"] = self._config["training"]["weight_decay"]
        model_config["batch_size"] = self._config["training"]["batch_size"]
        model_config["early_stopping"] = self._config["training"]["early_stopping"]
        
        # Features
        model_config["attention_mechanism"] = self._config["features"]["attention_mechanism"]
        model_config["pattern_recognition"] = self._config["features"]["pattern_recognition"]
        model_config["transfer_learning"] = self._config["features"]["transfer_learning"]
        
        return model_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get the current training configuration.
        
        Returns:
            Training configuration dictionary.
        """
        return self.get_section("training")
    
    def get_inference_config(self) -> Dict[str, Any]:
        """
        Get the current inference configuration.
        
        Returns:
            Inference configuration dictionary.
        """
        return self.get_section("inference")
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get the current storage configuration.
        
        Returns:
            Storage configuration dictionary.
        """
        return self.get_section("storage")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get the current optimization configuration.
        
        Returns:
            Optimization configuration dictionary.
        """
        return self.get_section("optimization")
    
    def generate_model_summary(self) -> str:
        """
        Generate a human-readable summary of the model configuration.
        
        Returns:
            String containing a summary of the model configuration.
        """
        model_type = self._config["model"]["type"]
        
        if model_type == "custom":
            model_description = "Custom Neural Matrix Model"
        else:
            model_description = MATRIX_MODELS[model_type]["description"]
        
        summary = [f"Neural Matrix Model: {model_type}"]
        summary.append(f"Description: {model_description}")
        summary.append("")
        
        if model_type != "custom":
            summary.append("Architecture:")
            summary.append(f"  Layers: {MATRIX_MODELS[model_type]['layers']}")
            summary.append(f"  Neurons: {MATRIX_MODELS[model_type]['neurons_per_layer']}")
        
        summary.append("Training Configuration:")
        summary.append(f"  Learning Rate: {self._config['training']['learning_rate']}")
        summary.append(f"  Weight Decay: {self._config['training']['weight_decay']}")
        summary.append(f"  Batch Size: {self._config['training']['batch_size']}")
        summary.append(f"  Early Stopping: {self._config['training']['early_stopping']}")
        summary.append("")
        
        summary.append("Capabilities:")
        summary.append(f"  Pattern Recognition: {self._config['features']['pattern_recognition']}")
        summary.append(f"  Transfer Learning: {self._config['features']['transfer_learning']}")
        summary.append(f"  Context Awareness: {self._config['features']['context_aware']}")
        summary.append("")
        
        summary.append("Advanced Features:")
        summary.append(f"  Attention Mechanism: {self._config['features']['attention_mechanism']}")
        summary.append(f"  Batch Normalization: {self._config['architecture']['batch_normalization']}")
        summary.append(f"  Skip Connections: {self._config['architecture']['skip_connections']}")
        
        return "\n".join(summary)
    
    def create_model_directory(self) -> str:
        """
        Create the directory for storing model weights.
        
        Returns:
            Path to the created directory.
        """
        weights_dir = self._config["storage"]["weights_dir"]
        
        # Expand ~ to user's home directory
        weights_dir = os.path.expanduser(weights_dir)
        
        # Create directory if it doesn't exist
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            logger.info(f"Created weights directory: {weights_dir}")
        
        return weights_dir
    
    def get_checkpoint_path(self, checkpoint_name: str) -> str:
        """
        Get the path for a checkpoint file.
        
        Args:
            checkpoint_name: Name of the checkpoint.
            
        Returns:
            Path to the checkpoint file.
        """
        weights_dir = self._config["storage"]["weights_dir"]
        
        # Expand ~ to user's home directory
        weights_dir = os.path.expanduser(weights_dir)
        
        return os.path.join(weights_dir, f"{checkpoint_name}.ckpt")
    
    def estimate_memory_usage(self) -> int:
        """
        Estimate the memory usage of the model in MB.
        
        Returns:
            Estimated memory usage in MB.
        """
        model_type = self._config["model"]["type"]
        
        if model_type != "custom":
            # Return predefined memory usage
            return MATRIX_MODELS[model_type]["max_memory_mb"]
        
        # For custom models, calculate memory usage based on layers and neurons
        memory_usage = 0
        
        # Add memory for custom model parameters
        custom_config = self._config["model"]["custom_config"] or {}
        
        if "neurons_per_layer" in custom_config:
            neurons = custom_config["neurons_per_layer"]
            
            # Calculate memory usage based on number of parameters
            prev_layer_size = 0
            for layer_size in neurons:
                if prev_layer_size > 0:
                    # Each connection uses 4 bytes (float32)
                    memory_usage += (prev_layer_size * layer_size * 4) / (1024 * 1024)  # Convert to MB
                prev_layer_size = layer_size
        
        # Add memory overhead for the framework
        memory_usage += 50  # Base memory usage in MB
        
        return int(memory_usage)
    
    def optimize_for_performance(self) -> None:
        """
        Optimize the configuration for performance.
        
        This method adjusts the configuration to optimize for performance,
        potentially at the cost of memory usage.
        """
        # Increase batch size for better parallelization
        self._config["training"]["batch_size"] = max(64, self._config["training"]["batch_size"])
        
        # Enable batch normalization for faster convergence
        self._config["architecture"]["batch_normalization"] = True
        
        # Use faster activation function
        self._config["architecture"]["activation"] = "relu"
        
        # Enable skip connections for better gradient flow
        self._config["architecture"]["skip_connections"] = True
        
        # Increase learning rate slightly
        self._config["training"]["learning_rate"] = max(0.001, self._config["training"]["learning_rate"])
        
        # Disable features that might slow down training
        self._config["features"]["attention_mechanism"] = False
        
        logger.info("Optimized configuration for performance")
    
    def optimize_for_memory(self) -> None:
        """
        Optimize the configuration for lower memory usage.
        
        This method adjusts the configuration to reduce memory usage,
        potentially at the cost of performance.
        """
        # Switch to lightweight model
        self._config["model"]["type"] = "lightweight"
        self._initialize_model_config()
        
        # Reduce batch size
        self._config["training"]["batch_size"] = min(16, self._config["training"]["batch_size"])
        
        # Increase dropout for smaller model
        self._config["architecture"]["dropout_rate"] = 0.4
        
        # Disable memory-intensive features
        self._config["architecture"]["batch_normalization"] = False
        self._config["architecture"]["skip_connections"] = False
        self._config["features"]["attention_mechanism"] = False
        
        # Enable pruning
        self._config["model"]["pruning_threshold"] = 0.05
        
        logger.info("Optimized configuration for lower memory usage")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return self.get_all()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NeuralMatrixConfiguration':
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
            
        Returns:
            NeuralMatrixConfiguration instance.
        """
        config = cls()
        
        for section, section_config in config_dict.items():
            if section in NEURAL_MATRIX_CONFIG_SCHEMA:
                for key, value in section_config.items():
                    if key in NEURAL_MATRIX_CONFIG_SCHEMA[section]:
                        config._config[section][key] = value
        
        config.validate()
        
        return config


# Global instance
neural_matrix_config = NeuralMatrixConfiguration()

def get_neural_matrix_config() -> NeuralMatrixConfiguration:
    """
    Get the global NeuralMatrixConfiguration instance.
    
    Returns:
        Global NeuralMatrixConfiguration instance.
    """
    return neural_matrix_config

def initialize_neural_matrix_config(config_file: Optional[str] = None) -> None:
    """
    Initialize the global NeuralMatrixConfiguration.
    
    Args:
        config_file: Path to the configuration file to load. If not provided,
            the default configuration file will be used.
    """
    global neural_matrix_config
    neural_matrix_config = NeuralMatrixConfiguration(config_file)

def create_default_neural_matrix_config(output_path: str = DEFAULT_MATRIX_CONFIG_FILE) -> None:
    """
    Create a default Neural Matrix configuration file.
    
    Args:
        output_path: Path to save the default configuration to.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate default configuration
    default_config = {}
    for section, section_schema in NEURAL_MATRIX_CONFIG_SCHEMA.items():
        default_config[section] = {}
        
        for key, key_schema in section_schema.items():
            default_config[section][key] = key_schema.get("default")
    
    # Save configuration
    with open(output_path, 'w') as f:
        if output_path.endswith('.json'):
            json.dump(default_config, f, indent=2)
        else:
            yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Default Neural Matrix configuration saved to {output_path}")

def main():
    """Command-line interface for Neural Matrix configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Matrix Configuration Manager")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create default config command
    create_parser = subparsers.add_parser("create", help="Create default configuration")
    create_parser.add_argument("--output", "-o", help="Output file path", default=DEFAULT_MATRIX_CONFIG_FILE)
    
    # Get config command
    get_parser = subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("section", help="Configuration section")
    get_parser.add_argument("key", nargs="?", help="Configuration key")
    get_parser.add_argument("--config", "-c", help="Configuration file path")
    
    # Set config command
    set_parser = subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("section", help="Configuration section")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Configuration value")
    set_parser.add_argument("--config", "-c", help="Configuration file path")
    
    # Reset config command
    reset_parser = subparsers.add_parser("reset", help="Reset configuration to defaults")
    reset_parser.add_argument("section", nargs="?", help="Configuration section to reset")
    reset_parser.add_argument("key", nargs="?", help="Configuration key to reset")
    reset_parser.add_argument("--config", "-c", help="Configuration file path")
    
    # List config command
    list_parser = subparsers.add_parser("list", help="List configuration")
    list_parser.add_argument("section", nargs="?", help="Configuration section to list")
    list_parser.add_argument("--config", "-c", help="Configuration file path")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize configuration")
    optimize_parser.add_argument("--mode", "-m", help="Optimization mode (performance, memory)", choices=["performance", "memory"], required=True)
    optimize_parser.add_argument("--config", "-c", help="Configuration file path")
    optimize_parser.add_argument("--output", "-o", help="Output file path")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Display model summary")
    summary_parser.add_argument("--config", "-c", help="Configuration file path")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "create":
        create_default_neural_matrix_config(args.output)
    
    elif args.command == "get":
        if args.config:
            initialize_neural_matrix_config(args.config)
        
        if args.key:
            value = neural_matrix_config.get(args.section, args.key)
            print(f"{args.section}.{args.key} = {value}")
        else:
            section = neural_matrix_config.get_section(args.section)
            print(f"{args.section}:")
            for key, value in section.items():
                print(f"  {key} = {value}")
    
    elif args.command == "set":
        if args.config:
            initialize_neural_matrix_config(args.config)
        
        # Convert value to appropriate type
        value = args.value
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
            value = float(value)
        
        try:
            neural_matrix_config.set(args.section, args.key, value)
            neural_matrix_config.save()
            print(f"Set {args.section}.{args.key} = {value}")
        except Exception as e:
            print(f"Error: {e}")
            import sys
            sys.exit(1)
    
    elif args.command == "reset":
        if args.config:
            initialize_neural_matrix_config(args.config)
        
        neural_matrix_config.reset(args.section, args.key)
        neural_matrix_config.save()
        
        if args.key:
            print(f"Reset {args.section}.{args.key} to default")
        elif args.section:
            print(f"Reset section {args.section} to defaults")
        else:
            print("Reset all configuration to defaults")
    
    elif args.command == "list":
        if args.config:
            initialize_neural_matrix_config(args.config)
        
        if args.section:
            section = neural_matrix_config.get_section(args.section)
            print(f"{args.section}:")
            for key, value in sorted(section.items()):
                schema = neural_matrix_config.get_schema(args.section, key)
                description = schema.get("description", "")
                print(f"  {key} = {value}")
                if description:
                    print(f"    # {description}")
        else:
            for section, section_config in sorted(neural_matrix_config.get_all().items()):
                print(f"{section}:")
                for key, value in sorted(section_config.items()):
                    print(f"  {key} = {value}")
    
    elif args.command == "optimize":
        if args.config:
            initialize_neural_matrix_config(args.config)
        
        if args.mode == "performance":
            neural_matrix_config.optimize_for_performance()
        elif args.mode == "memory":
            neural_matrix_config.optimize_for_memory()
        
        if args.output:
            neural_matrix_config.save(args.output)
            print(f"Optimized configuration saved to {args.output}")
        else:
            neural_matrix_config.save()
            print(f"Configuration optimized for {args.mode}")
    
    elif args.command == "summary":
        if args.config:
            initialize_neural_matrix_config(args.config)
        
        summary = neural_matrix_config.generate_model_summary()
        print(summary)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
