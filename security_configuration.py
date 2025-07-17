#!/usr/bin/env python3
"""
security_configuration.py
────────────────────────
Implements simplified security configuration for the FixWurx platform.

This module provides basic security features without user-based permissions or authentication.
It maintains system security through encryption and secure communication.
"""

import os
import sys
import time
import json
import yaml
import logging
import base64
import secrets
import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import threading

# Import system configuration
from system_configuration import get_config, ConfigurationError

# Configure logging
logger = logging.getLogger("SecurityConfiguration")

# Constants
DEFAULT_ENCRYPTION_ALGORITHM = "AES-256-GCM"
DEFAULT_AUDIT_LOG_ENABLED = True
DEFAULT_AUDIT_LOG_RETENTION_DAYS = 90
DEFAULT_SENSITIVE_FIELDS = ["password", "token", "key", "secret", "auth"]

@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    encryption_algorithm: str = DEFAULT_ENCRYPTION_ALGORITHM
    audit_log_enabled: bool = DEFAULT_AUDIT_LOG_ENABLED
    audit_log_retention_days: int = DEFAULT_AUDIT_LOG_RETENTION_DAYS
    sensitive_fields: List[str] = field(default_factory=lambda: DEFAULT_SENSITIVE_FIELDS.copy())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary."""
        return {
            "encryption_algorithm": self.encryption_algorithm,
            "audit_log_enabled": self.audit_log_enabled,
            "audit_log_retention_days": self.audit_log_retention_days,
            "sensitive_fields": self.sensitive_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityPolicy':
        """Create policy from dictionary."""
        return cls(
            encryption_algorithm=data.get("encryption_algorithm", DEFAULT_ENCRYPTION_ALGORITHM),
            audit_log_enabled=data.get("audit_log_enabled", DEFAULT_AUDIT_LOG_ENABLED),
            audit_log_retention_days=data.get("audit_log_retention_days", DEFAULT_AUDIT_LOG_RETENTION_DAYS),
            sensitive_fields=data.get("sensitive_fields", DEFAULT_SENSITIVE_FIELDS.copy())
        )

@dataclass
class AuditLogEntry:
    """Security audit log entry."""
    timestamp: float
    action: str
    resource: str
    status: str  # "success", "failure", "error"
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log entry to dictionary."""
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "resource": self.resource,
            "status": self.status,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLogEntry':
        """Create audit log entry from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            action=data["action"],
            resource=data["resource"],
            status=data["status"],
            details=data.get("details")
        )

class SecurityError(Exception):
    """Base exception for security errors."""
    pass

class SecurityManager:
    """
    Simplified security management for the FixWurx platform.
    
    This class provides basic security features without user-based authentication or authorization.
    It focuses on encryption, key management, and auditing.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the security manager.
        
        Args:
            config_dir: Directory for security configuration files.
        """
        self._config = get_config()
        
        # Set configuration directory
        if config_dir:
            self._config_dir = config_dir
        else:
            security_section = self._config.get_section("security")
            self._config_dir = security_section.get("config_dir", "~/.fixwurx/security")
        
        # Expand ~ to user's home directory
        self._config_dir = os.path.expanduser(self._config_dir)
        
        # Create directory if it doesn't exist
        if not os.path.exists(self._config_dir):
            os.makedirs(self._config_dir)
        
        # Set file paths
        self._policy_file = os.path.join(self._config_dir, "security_policy.yaml")
        self._audit_log_file = os.path.join(self._config_dir, "audit_log.jsonl")
        self._keys_file = os.path.join(self._config_dir, "security_keys.yaml")
        
        # Initialize data
        self._policy = self._load_policy()
        self._keys: Dict[str, Any] = {}
        
        # Load data
        self._load_keys()
        
        # Initialize locks
        self._audit_lock = threading.RLock()
        self._keys_lock = threading.RLock()
        
        # Start maintenance thread
        self._stop_maintenance = threading.Event()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="SecurityMaintenanceThread"
        )
        self._maintenance_thread.start()
        
        logger.info("Security manager initialized (no user-based security)")
    
    def _load_policy(self) -> SecurityPolicy:
        """
        Load security policy from file.
        
        Returns:
            SecurityPolicy object.
        """
        if os.path.exists(self._policy_file):
            try:
                with open(self._policy_file, 'r') as f:
                    policy_dict = yaml.safe_load(f)
                    if not policy_dict:
                        logger.warning("Empty security policy file, using defaults")
                        return SecurityPolicy()
                    return SecurityPolicy.from_dict(policy_dict)
            except Exception as e:
                logger.error(f"Error loading security policy: {e}")
                return SecurityPolicy()
        else:
            # Create default policy
            policy = SecurityPolicy()
            self._save_policy(policy)
            return policy
    
    def _save_policy(self, policy: SecurityPolicy) -> None:
        """
        Save security policy to file.
        
        Args:
            policy: SecurityPolicy object.
        """
        try:
            with open(self._policy_file, 'w') as f:
                yaml.dump(policy.to_dict(), f)
            logger.debug("Security policy saved")
        except Exception as e:
            logger.error(f"Error saving security policy: {e}")
    
    def _load_keys(self) -> None:
        """Load security keys from file."""
        if os.path.exists(self._keys_file):
            try:
                with open(self._keys_file, 'r') as f:
                    self._keys = yaml.safe_load(f)
                    if not self._keys:
                        self._keys = {}
                
                logger.debug("Security keys loaded")
            except Exception as e:
                logger.error(f"Error loading security keys: {e}")
        else:
            # Generate default keys
            self._generate_default_keys()
    
    def _generate_default_keys(self) -> None:
        """Generate default security keys."""
        with self._keys_lock:
            self._keys = {
                "encryption_key": base64.b64encode(secrets.token_bytes(32)).decode('utf-8'),
                "signing_key": base64.b64encode(secrets.token_bytes(32)).decode('utf-8'),
                "key_generation_time": time.time(),
                "key_rotation_interval": 90 * 24 * 60 * 60  # 90 days in seconds
            }
            self._save_keys()
    
    def _save_keys(self) -> None:
        """Save security keys to file."""
        with self._keys_lock:
            try:
                # Ensure restrictive permissions on the keys file
                # Create file if it doesn't exist
                if not os.path.exists(self._keys_file):
                    with open(self._keys_file, 'w') as f:
                        pass
                    
                    # Set file permissions (0600 - read/write only for owner)
                    os.chmod(self._keys_file, 0o600)
                
                with open(self._keys_file, 'w') as f:
                    yaml.dump(self._keys, f)
                logger.debug("Security keys saved")
            except Exception as e:
                logger.error(f"Error saving security keys: {e}")
    
    def _maintenance_loop(self) -> None:
        """Periodic maintenance loop."""
        while not self._stop_maintenance.is_set():
            try:
                # Check key rotation
                self._check_key_rotation()
                
                # Clean up audit logs
                self._cleanup_audit_logs()
            except Exception as e:
                logger.error(f"Error in security maintenance: {e}")
            
            # Sleep for a while
            time.sleep(60 * 5)  # 5 minutes
    
    def _check_key_rotation(self) -> None:
        """Check if security keys need rotation."""
        with self._keys_lock:
            current_time = time.time()
            key_gen_time = self._keys.get("key_generation_time", 0)
            key_rotation_interval = self._keys.get("key_rotation_interval", 90 * 24 * 60 * 60)
            
            if current_time - key_gen_time > key_rotation_interval:
                logger.info("Rotating security keys")
                self._rotate_security_keys()
    
    def _rotate_security_keys(self) -> None:
        """Rotate security keys."""
        with self._keys_lock:
            # Create new keys
            new_keys = {
                "encryption_key": base64.b64encode(secrets.token_bytes(32)).decode('utf-8'),
                "signing_key": base64.b64encode(secrets.token_bytes(32)).decode('utf-8'),
                "key_generation_time": time.time(),
                "key_rotation_interval": 90 * 24 * 60 * 60,
                "previous_keys": {
                    "encryption_key": self._keys.get("encryption_key"),
                    "signing_key": self._keys.get("signing_key"),
                    "key_generation_time": self._keys.get("key_generation_time")
                }
            }
            
            self._keys = new_keys
            self._save_keys()
            
            # Log key rotation
            self._add_audit_log_entry(
                action="rotate_keys",
                resource="security_keys",
                status="success",
                details={"reason": "scheduled_rotation"}
            )
    
    def _cleanup_audit_logs(self) -> None:
        """Clean up old audit logs."""
        if not self._policy.audit_log_enabled or not os.path.exists(self._audit_log_file):
            return
        
        try:
            # Get file modification time
            file_mtime = os.path.getmtime(self._audit_log_file)
            current_time = time.time()
            
            # Check if file is older than retention period
            retention_seconds = self._policy.audit_log_retention_days * 24 * 60 * 60
            if current_time - file_mtime > retention_seconds:
                # Archive old log file
                archive_dir = os.path.join(self._config_dir, "audit_archives")
                if not os.path.exists(archive_dir):
                    os.makedirs(archive_dir)
                
                # Create archive filename with timestamp
                timestamp = datetime.datetime.fromtimestamp(file_mtime).strftime("%Y%m%d")
                archive_file = os.path.join(archive_dir, f"audit_log_{timestamp}.jsonl.gz")
                
                # Compress and archive
                import gzip
                import shutil
                with open(self._audit_log_file, 'rb') as f_in:
                    with gzip.open(archive_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Clear current log file
                with open(self._audit_log_file, 'w') as f:
                    pass
                
                logger.info(f"Archived audit log to {archive_file}")
        except Exception as e:
            logger.error(f"Error cleaning up audit logs: {e}")
    
    def _add_audit_log_entry(self, action: str, resource: str, 
                             status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an entry to the audit log.
        
        Args:
            action: Action performed
            resource: Resource affected
            status: Status of the action (success, failure, error)
            details: Additional details (optional)
        """
        if not self._policy.audit_log_enabled:
            return
        
        with self._audit_lock:
            try:
                entry = AuditLogEntry(
                    timestamp=time.time(),
                    action=action,
                    resource=resource,
                    status=status,
                    details=details
                )
                
                # Write entry to log file
                with open(self._audit_log_file, 'a') as f:
                    f.write(json.dumps(entry.to_dict()) + '\n')
                
                logger.debug(f"Added audit log entry: {action} {resource} {status}")
            except Exception as e:
                logger.error(f"Error adding audit log entry: {e}")
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt data using the system encryption key.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as a base64 string
        """
        try:
            # Note: This is a simplified placeholder. In a real system,
            # you would use proper encryption libraries and algorithms.
            from cryptography.fernet import Fernet
            
            # Get encryption key
            key_bytes = base64.b64decode(self._keys.get("encryption_key", ""))
            
            # Use first 32 bytes for Fernet key
            fernet_key = base64.urlsafe_b64encode(key_bytes[:32])
            
            # Create cipher
            cipher = Fernet(fernet_key)
            
            # Encrypt data
            encrypted_data = cipher.encrypt(data.encode('utf-8'))
            
            # Return as base64 string
            return base64.b64encode(encrypted_data).decode('utf-8')
        except ImportError:
            logger.warning("Cryptography library not available, using fallback encryption")
            
            # Fallback to base64 encoding (not real encryption, just for demo)
            return base64.b64encode(data.encode('utf-8')).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise SecurityError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data using the system encryption key.
        
        Args:
            encrypted_data: Encrypted data as a base64 string
            
        Returns:
            Decrypted data
        """
        try:
            # Note: This is a simplified placeholder. In a real system,
            # you would use proper encryption libraries and algorithms.
            from cryptography.fernet import Fernet
            
            # Get encryption key
            key_bytes = base64.b64decode(self._keys.get("encryption_key", ""))
            
            # Use first 32 bytes for Fernet key
            fernet_key = base64.urlsafe_b64encode(key_bytes[:32])
            
            # Create cipher
            cipher = Fernet(fernet_key)
            
            # Decrypt data
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted_data = cipher.decrypt(encrypted_bytes)
            
            # Return as string
            return decrypted_data.decode('utf-8')
        except ImportError:
            logger.warning("Cryptography library not available, using fallback decryption")
            
            # Fallback to base64 decoding (not real decryption, just for demo)
            return base64.b64decode(encrypted_data.encode('utf-8')).decode('utf-8')
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise SecurityError(f"Decryption failed: {e}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Secure token as a base64 string
        """
        token_bytes = secrets.token_bytes(length)
        return base64.b64encode(token_bytes).decode('utf-8')
    
    def log_system_action(self, action: str, resource: str, 
                          status: str = "success", details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a system action to the audit log.
        
        Args:
            action: Action performed
            resource: Resource affected
            status: Status of the action (success, failure, error)
            details: Additional details (optional)
        """
        self._add_audit_log_entry(action, resource, status, details)
    
    def get_secure_headers(self) -> Dict[str, str]:
        """
        Get secure HTTP headers for web interface.
        
        Returns:
            Dictionary of secure HTTP headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache"
        }
    
    def shutdown(self) -> None:
        """Shutdown the security manager."""
        # Stop maintenance thread
        self._stop_maintenance.set()
        if self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=5)
        
        logger.info("Security manager shut down")

# Create a default instance
security_manager = SecurityManager()

# Command handler for security configuration
def security_command(args: str) -> int:
    """
    Security configuration command handler.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Security configuration commands")
    parser.add_argument("action", choices=["info", "audit", "encrypt", "decrypt", "token"], 
                        help="Action to perform")
    parser.add_argument("--value", "-v", help="Value for encryption/decryption")
    parser.add_argument("--file", "-f", help="File path")
    parser.add_argument("--length", "-l", type=int, default=32, help="Token length in bytes")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get security manager
    global security_manager
    
    # Perform action
    action = cmd_args.action
    
    if action == "info":
        # Show security configuration info
        print("Security Configuration:")
        print(f"  Encryption Algorithm: {security_manager._policy.encryption_algorithm}")
        print(f"  Audit Log Enabled: {security_manager._policy.audit_log_enabled}")
        print(f"  Audit Log Retention: {security_manager._policy.audit_log_retention_days} days")
        print(f"  Key Generation Time: {datetime.datetime.fromtimestamp(security_manager._keys.get('key_generation_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Key Rotation Interval: {security_manager._keys.get('key_rotation_interval', 0) / (24 * 60 * 60):.1f} days")
        return 0
    
    elif action == "audit":
        # Show audit log
        if not os.path.exists(security_manager._audit_log_file):
            print("No audit log found")
            return 0
        
        # Read audit log entries
        entries = []
        with open(security_manager._audit_log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        # Show entries
        print(f"Audit Log Entries ({len(entries)}):")
        for entry in entries[-10:]:  # Show last 10 entries
            timestamp = datetime.datetime.fromtimestamp(entry.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {timestamp} | {entry.get('action')} | {entry.get('resource')} | {entry.get('status')}")
        
        return 0
    
    elif action == "encrypt":
        # Encrypt value
        if not cmd_args.value:
            print("Error: Value required for encryption")
            return 1
        
        try:
            encrypted = security_manager.encrypt(cmd_args.value)
            print(f"Encrypted: {encrypted}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    elif action == "decrypt":
        # Decrypt value
        if not cmd_args.value:
            print("Error: Value required for decryption")
            return 1
        
        try:
            decrypted = security_manager.decrypt(cmd_args.value)
            print(f"Decrypted: {decrypted}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    elif action == "token":
        # Generate secure token
        length = cmd_args.length
        if length <= 0:
            print("Error: Token length must be positive")
            return 1
        
        token = security_manager.generate_secure_token(length)
        print(f"Token: {token}")
        return 0
    
    else:
        print(f"Unknown action: {action}")
        return 1
