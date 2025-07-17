"""
config_manager.py
─────────────────
Configuration management system for FixWurx.

Provides:
- Configuration validation
- Versioning
- Change tracking
- Backup and restore
- Environment-specific configurations

This system works with system_config.yaml to ensure consistent
configuration across environments and deployments.
"""

import os
import sys
import yaml
import json
import time
import shutil
import logging
import argparse
import datetime
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union

# Import modules if available
try:
    from access_control import log_action
    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    AUDIT_LOGGING_AVAILABLE = False
    # Simple logging fallback
    def log_action(username, action, target=None, details=None):
        """Fallback logging function if access_control module is not available."""
        logging.info(f"ACTION: {username} - {action}" + 
                    (f" - Target: {target}" if target else "") +
                    (f" - Details: {details}" if details else ""))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(".triangulum/config_manager.log"),
        logging.StreamHandler()
    ]
)
config_logger = logging.getLogger("config_manager")

# Default paths
TRIANGULUM_DIR = Path(".triangulum")
CONFIG_DIR = TRIANGULUM_DIR / "configs"
CONFIG_HISTORY_DIR = CONFIG_DIR / "history"
CONFIG_SCHEMA_DIR = CONFIG_DIR / "schemas"
CONFIG_BACKUP_DIR = CONFIG_DIR / "backups"

# Default configuration path
DEFAULT_CONFIG_PATH = Path("system_config.yaml")

# Default environment
DEFAULT_ENVIRONMENT = "development"

# Environment variable for configuration environment
ENV_VAR_NAME = "FIXWURX_ENV"

# Schema version compatibility
SCHEMA_COMPATIBILITY = {
    "1.0": ["1.0"],
    "1.1": ["1.0", "1.1"],
    "2.0": ["2.0"],
    "2.1": ["2.0", "2.1"]
}

# Configuration sections and their schemas
CONFIG_SECTIONS = {
    "llm": {
        "preferred": str,
        "temperature": float,
        "cost-budget-usd": float,
        "update_interval_hours": float,
        "check_interval_hours": float,
        "models": {
            "primary": str,
            "fallback": str,
            "offline": str,
            "explanation": str
        }
    },
    "security": {
        "key_rotation_interval_days": int,
        "access_control_enabled": bool,
        "audit_logging_enabled": bool,
        "sensitive_data_encryption": bool
    },
    "storage": {
        "buffer_size_mb": int,
        "compression_enabled": bool,
        "compression_level": int,
        "persistence_enabled": bool
    },
    "logging": {
        "level": str,
        "file_enabled": bool,
        "console_enabled": bool,
        "max_file_size_mb": int,
        "max_files": int,
        "retention_days": int
    },
    "performance": {
        "parallelism": int,
        "max_workers": int,
        "batch_size": int,
        "throttling_enabled": bool
    }
}


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class ConfigVersionError(Exception):
    """Exception raised for configuration version errors."""
    pass


class ConfigManager:
    """
    Manages system configuration.
    
    Features:
    - Configuration validation
    - Versioning
    - Change tracking
    - Backup and restore
    - Environment-specific configurations
    """
    
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        environment: Optional[str] = None,
        auto_create: bool = True,
        auto_upgrade: bool = False
    ) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
            environment: Environment to use (development, staging, production)
            auto_create: Automatically create default config if none exists
            auto_upgrade: Automatically upgrade configuration to latest schema
        """
        # Determine environment
        self.environment = environment or os.environ.get(ENV_VAR_NAME, DEFAULT_ENVIRONMENT)
        
        # Set up paths
        self.base_config_path = Path(config_path)
        self.config_path = self._get_environment_config_path()
        
        # Create directories
        TRIANGULUM_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.config = {}
        self.schema_version = "1.0"
        self.last_updated = 0
        self.update_history = []
        
        # Load configuration
        if self.config_path.exists():
            self._load_config()
        elif auto_create:
            self._create_default_config()
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Validate configuration
        self._validate_config()
        
        # Check for schema upgrades
        if auto_upgrade:
            self._check_for_upgrades()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if the key is not found
            
        Returns:
            The configuration value or the default value
        """
        parts = key_path.split(".")
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any, validate: bool = True) -> bool:
        """
        Set a configuration value by its path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
            validate: Whether to validate the new configuration
            
        Returns:
            True if the value was set, False otherwise
        """
        parts = key_path.split(".")
        parent = self.config
        
        # Traverse to the parent of the target key
        for part in parts[:-1]:
            if part not in parent or not isinstance(parent[part], dict):
                parent[part] = {}
            parent = parent[part]
        
        # Set the value
        parent[parts[-1]] = value
        
        # Validate the configuration
        if validate:
            try:
                self._validate_config()
            except ConfigValidationError as e:
                # Revert the change
                if len(parts) > 1:
                    del parent[parts[-1]]
                config_logger.error(f"Configuration validation failed: {str(e)}")
                return False
        
        # Save the configuration
        self._save_config()
        
        return True
    
    def delete(self, key_path: str, validate: bool = True) -> bool:
        """
        Delete a configuration value by its path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            validate: Whether to validate the new configuration
            
        Returns:
            True if the value was deleted, False otherwise
        """
        parts = key_path.split(".")
        parent = self.config
        
        # Traverse to the parent of the target key
        for part in parts[:-1]:
            if part not in parent or not isinstance(parent[part], dict):
                return False
            parent = parent[part]
        
        # Check if the key exists
        if parts[-1] not in parent:
            return False
        
        # Store the original value for potential rollback
        original_value = parent[parts[-1]]
        
        # Delete the key
        del parent[parts[-1]]
        
        # Validate the configuration
        if validate:
            try:
                self._validate_config()
            except ConfigValidationError as e:
                # Revert the change
                parent[parts[-1]] = original_value
                config_logger.error(f"Configuration validation failed: {str(e)}")
                return False
        
        # Save the configuration
        self._save_config()
        
        return True
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        try:
            self._validate_config()
            return True
        except ConfigValidationError:
            return False
    
    def backup(self, name: Optional[str] = None) -> str:
        """
        Create a backup of the current configuration.
        
        Args:
            name: Optional name for the backup
            
        Returns:
            The path to the backup file
        """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{now}_{self.environment}"
        backup_path = CONFIG_BACKUP_DIR / f"{backup_name}.yaml"
        
        # Create backup
        shutil.copy2(self.config_path, backup_path)
        
        # Log the backup
        config_logger.info(f"Created configuration backup: {backup_path}")
        
        # Audit log
        if AUDIT_LOGGING_AVAILABLE:
            log_action(
                username="system",
                action="CONFIG_BACKUP",
                target=str(backup_path),
                details=f"Created configuration backup for {self.environment} environment"
            )
        
        return str(backup_path)
    
    def restore(self, backup_path: Union[str, Path]) -> bool:
        """
        Restore a configuration from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if the restoration was successful, False otherwise
        """
        backup_path = Path(backup_path)
        
        # Check if the backup exists
        if not backup_path.exists():
            config_logger.error(f"Backup file not found: {backup_path}")
            return False
        
        # Create a backup of the current configuration
        self.backup("pre_restore")
        
        try:
            # Load the backup
            with open(backup_path, 'r') as f:
                backup_config = yaml.safe_load(f)
            
            # Validate the backup
            self._validate_config(backup_config)
            
            # Restore the backup
            shutil.copy2(backup_path, self.config_path)
            
            # Reload the configuration
            self._load_config()
            
            # Log the restoration
            config_logger.info(f"Restored configuration from backup: {backup_path}")
            
            # Audit log
            if AUDIT_LOGGING_AVAILABLE:
                log_action(
                    username="system",
                    action="CONFIG_RESTORE",
                    target=str(backup_path),
                    details=f"Restored configuration from backup for {self.environment} environment"
                )
            
            return True
        except Exception as e:
            config_logger.error(f"Failed to restore configuration from backup: {str(e)}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available configuration backups.
        
        Returns:
            List of dictionaries with backup information
        """
        backups = []
        
        for backup_file in CONFIG_BACKUP_DIR.glob("*.yaml"):
            try:
                # Get file stats
                stats = backup_file.stat()
                created = datetime.datetime.fromtimestamp(stats.st_ctime)
                size = stats.st_size
                
                # Extract environment from filename
                env_match = re.search(r"_(development|staging|production)\.yaml$", backup_file.name)
                environment = env_match.group(1) if env_match else "unknown"
                
                # Load the backup to get schema version
                with open(backup_file, 'r') as f:
                    backup_config = yaml.safe_load(f)
                    schema_version = backup_config.get("schema_version", "unknown")
                
                backups.append({
                    "path": str(backup_file),
                    "name": backup_file.stem,
                    "created": created.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": stats.st_ctime,
                    "size": size,
                    "environment": environment,
                    "schema_version": schema_version
                })
            except Exception as e:
                config_logger.warning(f"Failed to process backup file {backup_file}: {str(e)}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return backups
    
    def list_history(self) -> List[Dict[str, Any]]:
        """
        List the configuration change history.
        
        Returns:
            List of dictionaries with change information
        """
        return self.update_history
    
    def upgrade(self, target_version: Optional[str] = None) -> bool:
        """
        Upgrade the configuration to a newer schema version.
        
        Args:
            target_version: Target schema version, or latest if None
            
        Returns:
            True if the upgrade was successful, False otherwise
        """
        # Determine target version
        latest_version = self._get_latest_schema_version()
        target_version = target_version or latest_version
        
        # Check if upgrade is needed
        if self.schema_version == target_version:
            config_logger.info(f"Configuration already at version {target_version}")
            return True
        
        # Check if upgrade is possible
        if not self._is_upgrade_path_valid(target_version):
            config_logger.error(
                f"Cannot upgrade from {self.schema_version} to {target_version}. "
                f"No valid upgrade path available."
            )
            return False
        
        # Create a backup before upgrading
        self.backup(f"pre_upgrade_to_{target_version}")
        
        try:
            # Apply the upgrade
            self._apply_upgrade(target_version)
            
            # Update the schema version
            self.config["schema_version"] = target_version
            self.schema_version = target_version
            
            # Save the configuration
            self._save_config()
            
            # Log the upgrade
            config_logger.info(f"Upgraded configuration from {self.schema_version} to {target_version}")
            
            # Audit log
            if AUDIT_LOGGING_AVAILABLE:
                log_action(
                    username="system",
                    action="CONFIG_UPGRADE",
                    target=str(self.config_path),
                    details=f"Upgraded configuration from {self.schema_version} to {target_version}"
                )
            
            return True
        except Exception as e:
            config_logger.error(f"Failed to upgrade configuration: {str(e)}")
            return False
    
    def switch_environment(self, environment: str) -> bool:
        """
        Switch to a different environment configuration.
        
        Args:
            environment: Target environment (development, staging, production)
            
        Returns:
            True if the switch was successful, False otherwise
        """
        # Check if the environment is valid
        if environment not in ["development", "staging", "production"]:
            config_logger.error(f"Invalid environment: {environment}")
            return False
        
        # Store the current environment
        previous_environment = self.environment
        
        # Update the environment
        self.environment = environment
        
        # Update the config path
        self.config_path = self._get_environment_config_path()
        
        # Load the configuration for the new environment
        if self.config_path.exists():
            self._load_config()
        else:
            # Create a new configuration for the environment
            self._create_default_config()
        
        # Log the switch
        config_logger.info(f"Switched from {previous_environment} to {environment} environment")
        
        # Audit log
        if AUDIT_LOGGING_AVAILABLE:
            log_action(
                username="system",
                action="CONFIG_ENVIRONMENT_SWITCH",
                target=environment,
                details=f"Switched from {previous_environment} to {environment} environment"
            )
        
        return True
    
    def compare_environments(self, other_environment: str) -> Dict[str, Any]:
        """
        Compare the current environment configuration with another environment.
        
        Args:
            other_environment: Environment to compare with
            
        Returns:
            Dictionary with comparison information
        """
        # Check if the environment is valid
        if other_environment not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {other_environment}")
        
        # Load the other environment configuration
        other_path = self._get_environment_config_path(other_environment)
        
        if not other_path.exists():
            raise FileNotFoundError(f"Configuration for {other_environment} not found")
        
        with open(other_path, 'r') as f:
            other_config = yaml.safe_load(f)
        
        # Compare configurations
        added = []
        removed = []
        modified = []
        same = []
        
        self._compare_dict(
            self.config,
            other_config,
            added,
            removed,
            modified,
            same,
            path=""
        )
        
        # Build comparison result
        return {
            "current_environment": self.environment,
            "other_environment": other_environment,
            "added": sorted(added),
            "removed": sorted(removed),
            "modified": sorted(modified),
            "same": sorted(same),
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "same_count": len(same),
            "total_differences": len(added) + len(removed) + len(modified)
        }
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary with section configuration
        """
        return self.config.get(section, {})
    
    def set_section(self, section: str, value: Dict[str, Any], validate: bool = True) -> bool:
        """
        Set a configuration section.
        
        Args:
            section: Section name
            value: Section configuration
            validate: Whether to validate the new configuration
            
        Returns:
            True if the section was set, False otherwise
        """
        # Store the original section for potential rollback
        original_section = self.config.get(section, {})
        
        # Set the section
        self.config[section] = value
        
        # Validate the configuration
        if validate:
            try:
                self._validate_config()
            except ConfigValidationError as e:
                # Revert the change
                if section in self.config:
                    self.config[section] = original_section
                config_logger.error(f"Configuration validation failed: {str(e)}")
                return False
        
        # Save the configuration
        self._save_config()
        
        return True
    
    def _get_environment_config_path(self, env: Optional[str] = None) -> Path:
        """
        Get the configuration path for an environment.
        
        Args:
            env: Environment to get the path for, or current if None
            
        Returns:
            Path to the environment configuration file
        """
        env = env or self.environment
        
        if env == "development":
            return self.base_config_path
        else:
            filename = self.base_config_path.stem
            return CONFIG_DIR / f"{filename}.{env}.yaml"
    
    def _load_config(self) -> None:
        """Load the configuration from disk."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            
            # Extract schema version
            self.schema_version = self.config.get("schema_version", "1.0")
            
            # Extract last updated timestamp
            self.last_updated = self.config.get("last_updated", 0)
            
            # Load update history
            history_path = CONFIG_HISTORY_DIR / f"{self.config_path.stem}_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    self.update_history = json.load(f)
            else:
                self.update_history = []
            
            config_logger.debug(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            config_logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _save_config(self) -> None:
        """Save the configuration to disk."""
        try:
            # Update metadata
            now = time.time()
            self.config["last_updated"] = now
            self.config["schema_version"] = self.schema_version
            
            # Save configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            # Update history
            self._update_history(now)
            
            # Save history
            history_path = CONFIG_HISTORY_DIR / f"{self.config_path.stem}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.update_history, f, indent=2)
            
            # Update last updated timestamp
            self.last_updated = now
            
            config_logger.debug(f"Saved configuration to {self.config_path}")
        except Exception as e:
            config_logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def _create_default_config(self) -> None:
        """Create a default configuration."""
        default_config = {
            "schema_version": self.schema_version,
            "last_updated": time.time(),
            "llm": {
                "preferred": "openai",
                "temperature": 0.1,
                "cost-budget-usd": 1.0,
                "update_interval_hours": 24.0,
                "check_interval_hours": 12.0,
                "models": {
                    "primary": "gpt-4-turbo",
                    "fallback": "claude-3-sonnet",
                    "offline": "codellama-13b",
                    "explanation": "gpt-3.5-turbo"
                }
            },
            "security": {
                "key_rotation_interval_days": 30,
                "access_control_enabled": True,
                "audit_logging_enabled": True,
                "sensitive_data_encryption": True
            },
            "storage": {
                "buffer_size_mb": 100,
                "compression_enabled": True,
                "compression_level": 6,
                "persistence_enabled": True
            },
            "logging": {
                "level": "INFO",
                "file_enabled": True,
                "console_enabled": True,
                "max_file_size_mb": 10,
                "max_files": 5,
                "retention_days": 30
            },
            "performance": {
                "parallelism": 4,
                "max_workers": 8,
                "batch_size": 16,
                "throttling_enabled": True
            }
        }
        
        # Set the configuration
        self.config = default_config
        
        # Save the configuration
        self._save_config()
        
        config_logger.info(f"Created default configuration at {self.config_path}")
    
    def _validate_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Validate the configuration.
        
        Args:
            config: Configuration to validate, or current if None
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        config = config or self.config
        
        # Check schema version
        schema_version = config.get("schema_version")
        if not schema_version:
            raise ConfigValidationError("Missing schema_version in configuration")
        
        # Check last updated timestamp
        last_updated = config.get("last_updated")
        if not last_updated:
            raise ConfigValidationError("Missing last_updated in configuration")
        
        # Validate sections
        for section, schema in CONFIG_SECTIONS.items():
            if section not in config:
                if section == "llm" or section == "security":
                    raise ConfigValidationError(f"Missing required section: {section}")
                else:
                    continue
            
            self._validate_section(config[section], schema, section)
    
    def _validate_section(
        self,
        section: Dict[str, Any],
        schema: Dict[str, Any],
        path: str
    ) -> None:
        """
        Validate a configuration section against its schema.
        
        Args:
            section: Section to validate
            schema: Schema to validate against
            path: Path to the section
            
        Raises:
            ConfigValidationError: If the section is invalid
        """
        for key, expected_type in schema.items():
            # Build the full path to the key
            full_path = f"{path}.{key}" if path else key
            
            # Check if the key exists
            if key not in section:
                if path in ["llm", "security"] or key in ["primary", "level"]:
                    raise ConfigValidationError(f"Missing required key: {full_path}")
                else:
                    continue
            
            # Get the value
            value = section[key]
            
            # Check if the value is a nested schema
            if isinstance(expected_type, dict):
                # Check if the value is a dictionary
                if not isinstance(value, dict):
                    raise ConfigValidationError(
                        f"Invalid type for {full_path}: expected dict, got {type(value).__name__}"
                    )
                
                # Validate the nested section
                self._validate_section(value, expected_type, full_path)
            else:
                # Check if the value has the expected type
                if not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Invalid type for {full_path}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
    
    def _update_history(self, timestamp: float) -> None:
        """
        Update the configuration change history.
        
        Args:
            timestamp: Timestamp of the change
        """
        # Calculate configuration hash
        config_hash = self._hash_config()
        
        # Check if this is a new change
        if self.update_history and self.update_history[-1]["hash"] == config_hash:
            return
        
        # Create a new history entry
        entry = {
            "timestamp": timestamp,
            "datetime": datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "environment": self.environment,
            "schema_version": self.schema_version,
            "hash": config_hash
        }
        
        # Limit history to 100 entries
        self.update_history.append(entry)
        if len(self.update_history) > 100:
            self.update_history = self.update_history[-100:]
    
    def _hash_config(self) -> str:
        """
        Calculate a hash of the configuration.
        
        Returns:
            Hash of the configuration
        """
        # Create a copy of the configuration without metadata
        config_copy = self.config.copy()
        config_copy.pop("last_updated", None)
        
        # Convert to string and hash
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _get_latest_schema_version(self) -> str:
        """
        Get the latest available schema version.
        
        Returns:
            Latest schema version
        """
        # For now, we'll use a hardcoded value
        return "2.1"
    
    def _is_upgrade_path_valid(self, target_version: str) -> bool:
        """
        Check if an upgrade path is valid.
        
        Args:
            target_version: Target schema version
            
        Returns:
            True if the upgrade path is valid, False otherwise
        """
        # Check if the current and target versions are valid
        if self.schema_version not in SCHEMA_COMPATIBILITY:
            return False
        if target_version not in SCHEMA_COMPATIBILITY:
            return False
        
        # Check if there's a direct upgrade path
        compatible_versions = SCHEMA_COMPATIBILITY[target_version]
        if self.schema_version in compatible_versions:
            return True
        
        # No valid upgrade path
        return False
    
    def _apply_upgrade(self, target_version: str) -> None:
        """
        Apply a schema upgrade.
        
        Args:
            target_version: Target schema version
            
        Raises:
            ConfigVersionError: If the upgrade failed
        """
        # Check upgrade path from 1.0 to 1.1
        if self.schema_version == "1.0" and target_version == "1.1":
            # Add new fields to llm section
            if "llm" in self.config:
                self.config["llm"].setdefault("update_interval_hours", 24.0)
                self.config["llm"].setdefault("check_interval_hours", 12.0)
            return
        
        # Check upgrade path from 1.x to 2.0
        if self.schema_version.startswith("1.") and target_version == "2.0":
            # Add new sections
            self.config.setdefault("security", {
                "key_rotation_interval_days": 30,
                "access_control_enabled": True,
                "audit_logging_enabled": True,
                "sensitive_data_encryption": True
            })
            self.config.setdefault("storage", {
                "buffer_size_mb": 100,
                "compression_enabled": True,
                "compression_level": 6,
                "persistence_enabled": True
            })
            self.config.setdefault("logging", {
                "level": "INFO",
                "file_enabled": True,
                "console_enabled": True,
                "max_file_size_mb": 10,
                "max_files": 5,
                "retention_days": 30
            })
            self.config.setdefault("performance", {
                "parallelism": 4,
                "max_workers": 8,
                "batch_size": 16,
                "throttling_enabled": True
            })
            return
        
        # Check upgrade path from 2.0 to 2.1
        if self.schema_version == "2.0" and target_version == "2.1":
            # Add new fields to existing sections
            if "performance" in self.config:
                self.config["performance"].setdefault("throttling_enabled", True)
            return
        
        # Unsupported upgrade path
        raise ConfigVersionError(
            f"Unsupported upgrade path: {self.schema_version} → {target_version}"
        )
    
    def _check_for_upgrades(self) -> None:
        """Check for available schema upgrades and apply them if possible."""
        latest_version = self._get_latest_schema_version()
        
        # Check if upgrade is needed
        if self.schema_version == latest_version:
            return
        
        # Check if upgrade is possible
        if self._is_upgrade_path_valid(latest_version):
            config_logger.info(
                f"Upgrading configuration from {self.schema_version} to {latest_version}"
            )
            self.upgrade(latest_version)
        else:
            config_logger.warning(
                f"Configuration schema {self.schema_version} cannot be automatically "
                f"upgraded to {latest_version}. Manual upgrade required."
            )
    
    def _compare_dict(
        self,
        current: Dict[str, Any],
        other: Dict[str, Any],
        added: List[str],
        removed: List[str],
        modified: List[str],
        same: List[str],
        path: str
    ) -> None:
        """
        Compare two dictionaries recursively.
        
        Args:
            current: Current dictionary
            other: Other dictionary
            added: List of added keys
            removed: List of removed keys
            modified: List of modified keys
            same: List of unchanged keys
            path: Current path
        """
        # Check keys in current but not in other (removed)
        for key in current:
            if key not in other:
                full_path = f"{path}.{key}" if path else key
                removed.append(full_path)
        
        # Check keys in other but not in current (added)
        for key in other:
            if key not in current:
                full_path = f"{path}.{key}" if path else key
                added.append(full_path)
        
        # Check keys in both (same or modified)
        for key in current:
            if key not in other:
                continue
            
            full_path = f"{path}.{key}" if path else key
            
            # Skip metadata fields
            if key in ["schema_version", "last_updated"]:
                continue
            
            # Get values
            current_value = current[key]
            other_value = other[key]
            
            # Check if values are dictionaries
            if isinstance(current_value, dict) and isinstance(other_value, dict):
                # Recursively compare dictionaries
                self._compare_dict(
                    current_value,
                    other_value,
                    added,
                    removed,
                    modified,
                    same,
                    full_path
                )
            elif current_value != other_value:
                # Values are different
                modified.append(full_path)
            else:
                # Values are the same
                same.append(full_path)


# CLI interface
def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(description="Configuration Management Tool")
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get a configuration value")
    get_parser.add_argument(
        "key",
        type=str,
        help="Key path (e.g., llm.models.primary)"
    )
    get_parser.add_argument(
        "--default",
        type=str,
        help="Default value if the key is not found"
    )
    
    # Set command
    set_parser = subparsers.add_parser("set", help="Set a configuration value")
    set_parser.add_argument(
        "key",
        type=str,
        help="Key path (e.g., llm.models.primary)"
    )
    set_parser.add_argument(
        "value",
        type=str,
        help="Value to set"
    )
    set_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a configuration value")
    delete_parser.add_argument(
        "key",
        type=str,
        help="Key path (e.g., llm.models.primary)"
    )
    delete_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation"
    )
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a configuration backup")
    backup_parser.add_argument(
        "--name",
        type=str,
        help="Name for the backup"
    )
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a configuration backup")
    restore_parser.add_argument(
        "backup",
        type=str,
        help="Path to the backup file or backup name"
    )
    
    # List backups command
    list_backups_parser = subparsers.add_parser(
        "list-backups",
        help="List available configuration backups"
    )
    
    # List history command
    list_history_parser = subparsers.add_parser(
        "list-history",
        help="List configuration change history"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the configuration"
    )
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        help="Upgrade the configuration to a newer schema version"
    )
    upgrade_parser.add_argument(
        "--target",
        type=str,
        help="Target schema version (default: latest)"
    )
    
    # Environment command
    env_parser = subparsers.add_parser(
        "env",
        help="Environment management"
    )
    env_subparsers = env_parser.add_subparsers(dest="env_command", help="Environment command")
    
    # Environment switch command
    env_switch_parser = env_subparsers.add_parser(
        "switch",
        help="Switch to a different environment"
    )
    env_switch_parser.add_argument(
        "environment",
        type=str,
        choices=["development", "staging", "production"],
        help="Target environment"
    )
    
    # Environment compare command
    env_compare_parser = env_subparsers.add_parser(
        "compare",
        help="Compare environments"
    )
    env_compare_parser.add_argument(
        "environment",
        type=str,
        choices=["development", "staging", "production"],
        help="Environment to compare with"
    )
    
    # Environment show command
    env_show_parser = env_subparsers.add_parser(
        "show",
        help="Show current environment"
    )
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Create configuration manager
    try:
        config_manager = ConfigManager()
        
        # Handle command
        if args.command == "get":
            # Convert string value to appropriate type
            default = args.default
            if default is not None:
                if default.lower() == "true":
                    default = True
                elif default.lower() == "false":
                    default = False
                elif default.isdigit():
                    default = int(default)
                elif default.replace(".", "", 1).isdigit() and default.count(".") == 1:
                    default = float(default)
            
            # Get value
            value = config_manager.get(args.key, default)
            
            # Print value
            print(value)
            
        elif args.command == "set":
            # Convert string value to appropriate type
            value = args.value
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                value = float(value)
            
            # Set value
            success = config_manager.set(args.key, value, not args.no_validate)
            
            # Print result
            if success:
                print(f"Set {args.key} to {value}")
            else:
                print(f"Failed to set {args.key}")
                sys.exit(1)
            
        elif args.command == "delete":
            # Delete value
            success = config_manager.delete(args.key, not args.no_validate)
            
            # Print result
            if success:
                print(f"Deleted {args.key}")
            else:
                print(f"Failed to delete {args.key}")
                sys.exit(1)
            
        elif args.command == "backup":
            # Create backup
            backup_path = config_manager.backup(args.name)
            
            # Print result
            print(f"Created backup: {backup_path}")
            
        elif args.command == "restore":
            # Check if the argument is a backup name or path
            backup_path = args.backup
            if not os.path.exists(backup_path):
                # Try to find the backup by name
                backups = config_manager.list_backups()
                for backup in backups:
                    if backup["name"] == args.backup:
                        backup_path = backup["path"]
                        break
            
            # Restore backup
            success = config_manager.restore(backup_path)
            
            # Print result
            if success:
                print(f"Restored configuration from: {backup_path}")
            else:
                print(f"Failed to restore configuration from: {backup_path}")
                sys.exit(1)
            
        elif args.command == "list-backups":
            # List backups
            backups = config_manager.list_backups()
            
            # Print result
            if not backups:
                print("No backups found")
            else:
                print(f"Found {len(backups)} backups:")
                for backup in backups:
                    print(
                        f"  {backup['name']} - "
                        f"{backup['created']} - "
                        f"{backup['environment']} - "
                        f"v{backup['schema_version']} - "
                        f"{backup['size']} bytes"
                    )
            
        elif args.command == "list-history":
            # List history
            history = config_manager.list_history()
            
            # Print result
            if not history:
                print("No history found")
            else:
                print(f"Found {len(history)} changes:")
                for entry in history:
                    print(
                        f"  {entry['datetime']} - "
                        f"{entry['environment']} - "
                        f"v{entry['schema_version']}"
                    )
            
        elif args.command == "validate":
            # Validate configuration
            valid = config_manager.validate()
            
            # Print result
            if valid:
                print("Configuration is valid")
            else:
                print("Configuration is invalid")
                sys.exit(1)
            
        elif args.command == "upgrade":
            # Upgrade configuration
            success = config_manager.upgrade(args.target)
            
            # Print result
            if success:
                print(f"Upgraded configuration to schema version {config_manager.schema_version}")
            else:
                print("Failed to upgrade configuration")
                sys.exit(1)
            
        elif args.command == "env":
            if args.env_command == "switch":
                # Switch environment
                success = config_manager.switch_environment(args.environment)
                
                # Print result
                if success:
                    print(f"Switched to {args.environment} environment")
                else:
                    print(f"Failed to switch to {args.environment} environment")
                    sys.exit(1)
                
            elif args.env_command == "compare":
                # Compare environments
                try:
                    result = config_manager.compare_environments(args.environment)
                    
                    # Print result
                    print(f"Comparing {result['current_environment']} with {result['other_environment']}:")
                    print(f"  Total differences: {result['total_differences']}")
                    print(f"  Added: {result['added_count']}")
                    print(f"  Removed: {result['removed_count']}")
                    print(f"  Modified: {result['modified_count']}")
                    print(f"  Same: {result['same_count']}")
                    
                    if result['added']:
                        print("\nAdded:")
                        for key in result['added']:
                            print(f"  {key}")
                    
                    if result['removed']:
                        print("\nRemoved:")
                        for key in result['removed']:
                            print(f"  {key}")
                    
                    if result['modified']:
                        print("\nModified:")
                        for key in result['modified']:
                            print(f"  {key}")
                except FileNotFoundError as e:
                    print(str(e))
                    sys.exit(1)
                
            elif args.env_command == "show":
                # Show current environment
                print(f"Current environment: {config_manager.environment}")
            
            else:
                env_parser = create_cli_parser()._subparsers._group_actions[0].choices["env"]
                env_parser.print_help()
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
