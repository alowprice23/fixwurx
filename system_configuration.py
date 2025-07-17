#!/usr/bin/env python3
"""
system_configuration.py
───────────────────────
Implements the basic system configuration management for the FixWurx platform.

This module provides functionality for loading, validating, saving, and accessing
system-wide configuration settings. It supports multiple configuration sources
(default, system, user), configuration validation, and change notifications.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import copy
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Configure logging
logger = logging.getLogger("SystemConfiguration")

# Default configuration directory and files
DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "system_config.yaml")
USER_CONFIG_FILE = os.path.expanduser("~/.fixwurx/config.yaml")

# Configuration schema definitions
CONFIG_SCHEMA = {
    "system": {
        "log_level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "default": "INFO",
            "description": "Default logging level for the system"
        },
        "log_file": {
            "type": "string",
            "default": "fixwurx.log",
            "description": "Path to the log file"
        },
        "max_log_size": {
            "type": "integer",
            "default": 10485760,  # 10 MB
            "description": "Maximum log file size in bytes before rotation"
        },
        "backup_count": {
            "type": "integer",
            "default": 5,
            "description": "Number of backup log files to keep"
        },
        "temp_dir": {
            "type": "string",
            "default": "/tmp/fixwurx",
            "description": "Temporary directory for the system"
        }
    },
    "shell": {
        "history_file": {
            "type": "string",
            "default": "~/.fixwurx/history",
            "description": "Path to the shell history file"
        },
        "max_history": {
            "type": "integer",
            "default": 1000,
            "description": "Maximum number of history entries"
        },
        "prompt": {
            "type": "string",
            "default": "fx> ",
            "description": "Shell prompt"
        },
        "completion": {
            "type": "boolean",
            "default": True,
            "description": "Enable tab completion"
        },
        "auto_suggest": {
            "type": "boolean",
            "default": True,
            "description": "Enable auto-suggestions based on history"
        },
        "highlight": {
            "type": "boolean",
            "default": True,
            "description": "Enable syntax highlighting"
        }
    },
    "agents": {
        "default_timeout": {
            "type": "integer",
            "default": 60,
            "description": "Default timeout for agent operations in seconds"
        },
        "max_agents": {
            "type": "integer",
            "default": 10,
            "description": "Maximum number of agents to run simultaneously"
        },
        "agent_log_level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "default": "INFO",
            "description": "Default logging level for agents"
        },
        "persistence": {
            "type": "boolean",
            "default": True,
            "description": "Whether to persist agent state between sessions"
        },
        "auto_start": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "default": ["meta"],
            "description": "Agents to automatically start on system initialization"
        }
    },
    "triangulum": {
        "daemon_port": {
            "type": "integer",
            "default": 8080,
            "description": "Port for the Triangulum daemon"
        },
        "connection_timeout": {
            "type": "integer",
            "default": 5,
            "description": "Connection timeout in seconds"
        },
        "plan_storage": {
            "type": "string",
            "default": "~/.fixwurx/plans",
            "description": "Directory for storing Triangulum plans"
        },
        "default_priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "default": "medium",
            "description": "Default priority for Triangulum plans"
        }
    },
    "neural_matrix": {
        "enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether to enable the Neural Matrix"
        },
        "storage_path": {
            "type": "string",
            "default": "~/.fixwurx/neural_matrix",
            "description": "Path to store Neural Matrix data"
        },
        "learning_rate": {
            "type": "number",
            "default": 0.001,
            "description": "Learning rate for the Neural Matrix"
        },
        "weight_decay": {
            "type": "number",
            "default": 0.0001,
            "description": "Weight decay for the Neural Matrix"
        },
        "model_type": {
            "type": "string",
            "enum": ["lightweight", "standard", "advanced"],
            "default": "standard",
            "description": "Type of Neural Matrix model to use"
        }
    },
    "auditor": {
        "enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether to enable the Auditor"
        },
        "log_level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "default": "INFO",
            "description": "Logging level for the Auditor"
        },
        "sensors": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "default": ["system", "process", "file", "network"],
            "description": "Sensors to enable for the Auditor"
        },
        "retention_days": {
            "type": "integer",
            "default": 30,
            "description": "Number of days to retain audit logs"
        },
        "alert_threshold": {
            "type": "string",
            "enum": ["info", "warning", "error", "critical"],
            "default": "warning",
            "description": "Minimum level for alerts"
        }
    },
    "security": {
        "api_key_rotation": {
            "type": "integer",
            "default": 30,
            "description": "Number of days before API key rotation"
        },
        "encryption_enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether to enable encryption for sensitive data"
        },
        "authentication_required": {
            "type": "boolean",
            "default": False,
            "description": "Whether to require authentication for the shell"
        },
        "permission_level": {
            "type": "string",
            "enum": ["read_only", "standard", "advanced", "admin"],
            "default": "standard",
            "description": "Default permission level"
        }
    }
}

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

class Configuration:
    """
    System configuration manager.
    
    This class manages the system configuration, including loading, validating,
    saving, and accessing configuration settings. It supports multiple configuration
    sources (default, system, user) and provides a unified interface for accessing
    configuration values.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file to load. If not provided,
                the default configuration file will be used.
        """
        self._config = {}
        self._default_config = self._generate_default_config()
        self._change_listeners = []
        
        # Load configuration
        if config_file:
            self._config_file = config_file
        else:
            self._config_file = USER_CONFIG_FILE if os.path.exists(USER_CONFIG_FILE) else DEFAULT_CONFIG_FILE
        
        self.load()
        
        logger.info(f"Configuration initialized from {self._config_file}")
    
    def _generate_default_config(self) -> Dict[str, Any]:
        """
        Generate the default configuration from the schema.
        
        Returns:
            Default configuration dictionary.
        """
        default_config = {}
        
        for section, section_schema in CONFIG_SCHEMA.items():
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
            
            # If system config file exists, load it and merge
            if os.path.exists(DEFAULT_CONFIG_FILE):
                try:
                    system_config = self._load_config_file(DEFAULT_CONFIG_FILE)
                    self._merge_config(self._config, system_config)
                    logger.debug(f"Loaded system configuration from {DEFAULT_CONFIG_FILE}")
                except Exception as e:
                    logger.warning(f"Failed to load system configuration: {e}")
            
            # If user config file exists, load it and merge
            if os.path.exists(self._config_file) and self._config_file != DEFAULT_CONFIG_FILE:
                try:
                    user_config = self._load_config_file(self._config_file)
                    self._merge_config(self._config, user_config)
                    logger.debug(f"Loaded user configuration from {self._config_file}")
                except Exception as e:
                    logger.warning(f"Failed to load user configuration: {e}")
            
            # Validate configuration
            self.validate()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self._config = copy.deepcopy(self._default_config)
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file.
            
        Returns:
            Configuration dictionary.
            
        Raises:
            ConfigurationError: If the file cannot be loaded.
        """
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {file_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file {file_path}: {e}")
    
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
    
    def validate(self) -> None:
        """
        Validate the configuration against the schema.
        
        Raises:
            ConfigurationError: If the configuration is invalid.
        """
        for section, section_schema in CONFIG_SCHEMA.items():
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
                
                elif expected_type == "array" and not isinstance(value, list):
                    logger.warning(f"Invalid type for {section}.{key}: expected array, got {type(value)}")
                    self._config[section][key] = key_schema.get("default")
                
                # Validate enum
                if "enum" in key_schema and value not in key_schema["enum"]:
                    logger.warning(f"Invalid value for {section}.{key}: {value} not in {key_schema['enum']}")
                    self._config[section][key] = key_schema.get("default")
    
    def save(self, file_path: Optional[str] = None) -> None:
        """
        Save the configuration to a file.
        
        Args:
            file_path: Path to save the configuration to. If not provided,
                the current configuration file will be used.
                
        Raises:
            ConfigurationError: If the configuration cannot be saved.
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
                    raise ConfigurationError(f"Unsupported configuration file format: {save_path}")
            
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
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
            ConfigurationError: If the section or key doesn't exist in the schema.
        """
        # Check if section and key exist in schema
        if section not in CONFIG_SCHEMA:
            raise ConfigurationError(f"Unknown configuration section: {section}")
        
        if key not in CONFIG_SCHEMA[section]:
            raise ConfigurationError(f"Unknown configuration key: {section}.{key}")
        
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
            raise ConfigurationError(f"Invalid configuration value: {e}")
        
        # Notify listeners of change
        if old_value != value:
            self._notify_change(section, key, value, old_value)
    
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
        elif section in self._config:
            if key is None:
                # Reset entire section
                self._config[section] = copy.deepcopy(self._default_config.get(section, {}))
            elif key in self._config[section]:
                # Reset specific key
                default_value = self._default_config.get(section, {}).get(key)
                old_value = self._config[section][key]
                self._config[section][key] = default_value
                
                # Notify listeners of change
                if old_value != default_value:
                    self._notify_change(section, key, default_value, old_value)
    
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
            return CONFIG_SCHEMA
        
        if section in CONFIG_SCHEMA:
            if key is None:
                return CONFIG_SCHEMA[section]
            
            if key in CONFIG_SCHEMA[section]:
                return CONFIG_SCHEMA[section][key]
        
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
    
    def has_section(self, section: str) -> bool:
        """
        Check if a configuration section exists.
        
        Args:
            section: Configuration section.
            
        Returns:
            True if the section exists, False otherwise.
        """
        return section in self._config
    
    def has_key(self, section: str, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            section: Configuration section.
            key: Configuration key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        return section in self._config and key in self._config[section]

# Create the global configuration instance
config = Configuration()

def get_config() -> Configuration:
    """
    Get the global configuration instance.
    
    Returns:
        Global configuration instance.
    """
    return config

def initialize_config(config_file: Optional[str] = None) -> None:
    """
    Initialize the global configuration.
    
    Args:
        config_file: Path to the configuration file to load. If not provided,
            the default configuration file will be used.
    """
    global config
    config = Configuration(config_file)

def create_default_config(output_path: str = DEFAULT_CONFIG_FILE) -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path to save the default configuration to.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate default configuration
    default_config = {}
    for section, section_schema in CONFIG_SCHEMA.items():
        default_config[section] = {}
        
        for key, key_schema in section_schema.items():
            default_config[section][key] = key_schema.get("default")
            
            # Add comment with description
            description = key_schema.get("description", "")
            if description:
                # We can't add comments to JSON, but we can add them to YAML
                # For now, we'll just print the descriptions
                print(f"{section}.{key}: {description}")
    
    # Save configuration
    with open(output_path, 'w') as f:
        if output_path.endswith('.json'):
            json.dump(default_config, f, indent=2)
        else:
            yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Default configuration saved to {output_path}")

def main():
    """Command-line interface for configuration management."""
    parser = argparse.ArgumentParser(description="FixWurx Configuration Manager")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create default config command
    create_parser = subparsers.add_parser("create", help="Create default configuration")
    create_parser.add_argument("--output", "-o", help="Output file path", default=DEFAULT_CONFIG_FILE)
    
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "create":
        create_default_config(args.output)
    
    elif args.command == "get":
        if args.config:
            initialize_config(args.config)
        
        if args.key:
            value = config.get(args.section, args.key)
            print(f"{args.section}.{args.key} = {value}")
        else:
            section = config.get_section(args.section)
            print(f"{args.section}:")
            for key, value in section.items():
                print(f"  {key} = {value}")
    
    elif args.command == "set":
        if args.config:
            initialize_config(args.config)
        
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
            config.set(args.section, args.key, value)
            config.save()
            print(f"Set {args.section}.{args.key} = {value}")
        except ConfigurationError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "reset":
        if args.config:
            initialize_config(args.config)
        
        config.reset(args.section, args.key)
        config.save()
        
        if args.key:
            print(f"Reset {args.section}.{args.key} to default")
        elif args.section:
            print(f"Reset section {args.section} to defaults")
        else:
            print("Reset all configuration to defaults")
    
    elif args.command == "list":
        if args.config:
            initialize_config(args.config)
        
        if args.section:
            section = config.get_section(args.section)
            print(f"{args.section}:")
            for key, value in sorted(section.items()):
                schema = config.get_schema(args.section, key)
                description = schema.get("description", "")
                print(f"  {key} = {value}")
                if description:
                    print(f"    # {description}")
        else:
            for section, section_config in sorted(config.get_all().items()):
                print(f"{section}:")
                for key, value in sorted(section_config.items()):
                    print(f"  {key} = {value}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
