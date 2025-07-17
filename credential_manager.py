"""
credential_manager.py
────────────────────
Secure credential management for the FixWurx shell.

This module provides secure storage and retrieval of credentials for the
FixWurx command execution environment, including encryption.
"""

import os
import json
import logging
import base64
import secrets
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("credential_manager.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("CredentialManager")

# Singleton instance
_instance = None

def get_instance(registry=None):
    """
    Get the singleton instance of the CredentialManager.
    
    Args:
        registry: Optional component registry
        
    Returns:
        CredentialManager instance
    """
    global _instance
    if _instance is None:
        _instance = CredentialManager(registry=registry)
    return _instance


class CredentialManager:
    """
    Manages secure storage and retrieval of credentials.
    """
    
    def __init__(self, config: Dict[str, Any] = None, registry=None):
        """
        Initialize credential manager.
        
        Args:
            config: Optional configuration dictionary
            registry: Optional component registry
        """
        self.config = config or {}
        self.registry = registry
        
        # Credential storage
        self.credentials = {}
        
        # Secure storage
        self.storage_dir = self.config.get("storage_dir", "security/credentials")
        self.credential_file = os.path.join(self.storage_dir, "credentials.json")
        
        # Encryption key
        self.master_key = self.config.get("master_key", None)
        if not self.master_key:
            # Generate a random key
            self.master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            logger.warning("No master key provided, generated a random key. This key will not persist across restarts.")
        
        # Initialize
        self.initialized = False
        
        # Register with registry if provided
        if self.registry:
            self.registry.register_component("credential_manager", self)
            
        logger.info("Credential Manager initialized with default settings")
    
    def initialize(self):
        """
        Initialize the credential manager.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_dir, exist_ok=True)
            
            # Load credentials from file if it exists
            if os.path.exists(self.credential_file):
                self._load_credentials()
            
            self.initialized = True
            logger.info("Credential Manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing credential manager: {e}")
            return False
    
    def shutdown(self):
        """
        Shutdown the credential manager.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        try:
            # Save credentials to file
            self._save_credentials()
            
            # Clear credentials from memory
            self.credentials = {}
            
            self.initialized = False
            logger.info("Credential Manager shut down successfully")
            return True
        except Exception as e:
            logger.error(f"Error shutting down credential manager: {e}")
            return False
    
    def set_credential(self, name: str, value: str) -> bool:
        """
        Set a credential.
        
        Args:
            name: Credential name
            value: Credential value
            
        Returns:
            True if the credential was set successfully, False otherwise
        """
        try:
            # Encrypt the credential value
            encrypted_value = self._encrypt(value)
            
            # Store the credential
            self.credentials[name] = encrypted_value
            
            # Save credentials to file
            self._save_credentials()
            
            logger.info(f"Set credential: {name}")
            return True
        except Exception as e:
            logger.error(f"Error setting credential {name}: {e}")
            return False
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Get a credential.
        
        Args:
            name: Credential name
            
        Returns:
            Credential value, or None if not found
        """
        try:
            # Get the encrypted credential value
            encrypted_value = self.credentials.get(name)
            if not encrypted_value:
                logger.warning(f"Credential {name} not found")
                return None
            
            # Decrypt the credential value
            value = self._decrypt(encrypted_value)
            
            return value
        except Exception as e:
            logger.error(f"Error getting credential {name}: {e}")
            return None
    
    def delete_credential(self, name: str) -> bool:
        """
        Delete a credential.
        
        Args:
            name: Credential name
            
        Returns:
            True if the credential was deleted successfully, False otherwise
        """
        try:
            # Delete the credential
            if name in self.credentials:
                del self.credentials[name]
                
                # Save credentials to file
                self._save_credentials()
                
                logger.info(f"Deleted credential: {name}")
                return True
            else:
                logger.warning(f"Credential {name} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting credential {name}: {e}")
            return False
    
    def list_credentials(self) -> List[str]:
        """
        List all credential names.
        
        Returns:
            List of credential names
        """
        return list(self.credentials.keys())
    
    def _encrypt(self, value: str) -> str:
        """
        Encrypt a value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            Encrypted value
        """
        # This is a simplified implementation for testing
        # In a real implementation, use a proper encryption library
        try:
            from cryptography.fernet import Fernet
            key = base64.urlsafe_b64encode(self.master_key.ljust(32)[:32].encode())
            f = Fernet(key)
            return f.encrypt(value.encode()).decode()
        except ImportError:
            # Fallback to basic encoding if cryptography is not available
            return base64.b64encode(value.encode()).decode()
    
    def _decrypt(self, encrypted_value: str) -> str:
        """
        Decrypt a value.
        
        Args:
            encrypted_value: Encrypted value
            
        Returns:
            Decrypted value
        """
        # This is a simplified implementation for testing
        # In a real implementation, use a proper encryption library
        try:
            from cryptography.fernet import Fernet
            key = base64.urlsafe_b64encode(self.master_key.ljust(32)[:32].encode())
            f = Fernet(key)
            return f.decrypt(encrypted_value.encode()).decode()
        except ImportError:
            # Fallback to basic decoding if cryptography is not available
            return base64.b64decode(encrypted_value.encode()).decode()
    
    def _save_credentials(self) -> None:
        """
        Save credentials to file.
        """
        try:
            # Create credentials dictionary
            data = {
                "credentials": self.credentials
            }
            
            # Save to file
            with open(self.credential_file, 'w') as f:
                json.dump(data, f)
                
            logger.info(f"Saved credentials to {self.credential_file}")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
    
    def _load_credentials(self) -> None:
        """
        Load credentials from file.
        """
        try:
            # Load from file
            with open(self.credential_file, 'r') as f:
                data = json.load(f)
                
            # Extract credentials
            self.credentials = data.get("credentials", {})
                
            logger.info(f"Loaded credentials from {self.credential_file}")
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
