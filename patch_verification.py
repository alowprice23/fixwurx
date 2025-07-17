#!/usr/bin/env python3
"""
Patch Verification Module

This module provides cryptographic verification for patches, ensuring their
integrity and authenticity before they are applied to the codebase.
"""

import os
import sys
import json
import hashlib
import base64
import logging
import time
import hmac
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.exceptions import InvalidSignature

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("patch_verification.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PatchVerification")

class KeyManager:
    """
    Manage cryptographic keys for patch signing and verification.
    """
    
    def __init__(self, keys_dir: str = None):
        """
        Initialize key manager.
        
        Args:
            keys_dir: Directory for storing keys
        """
        self.keys_dir = os.path.abspath(keys_dir or "keys")
        
        # Create keys directory if it doesn't exist
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Default key paths
        self.private_key_path = os.path.join(self.keys_dir, "private_key.pem")
        self.public_key_path = os.path.join(self.keys_dir, "public_key.pem")
        
        # Load or generate keys
        self.private_key, self.public_key = self._load_or_generate_keys()
        
        logger.info(f"Key manager initialized with keys in {self.keys_dir}")
    
    def _load_or_generate_keys(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Load existing keys or generate new ones if they don't exist.
        
        Returns:
            Tuple of (private_key, public_key)
        """
        private_key = None
        public_key = None
        
        # Check if keys exist
        if os.path.exists(self.private_key_path) and os.path.exists(self.public_key_path):
            try:
                # Load private key
                with open(self.private_key_path, "rb") as f:
                    private_key_data = f.read()
                    private_key = load_pem_private_key(private_key_data, password=None)
                
                # Load public key
                with open(self.public_key_path, "rb") as f:
                    public_key_data = f.read()
                    public_key = load_pem_public_key(public_key_data)
                
                logger.info("Loaded existing keys")
                return private_key, public_key
            except Exception as e:
                logger.error(f"Error loading keys: {e}")
        
        # Generate new keys
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Save keys
        try:
            # Save private key
            private_key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open(self.private_key_path, "wb") as f:
                f.write(private_key_data)
            
            # Save public key
            public_key_data = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            with open(self.public_key_path, "wb") as f:
                f.write(public_key_data)
            
            logger.info("Generated and saved new keys")
        except Exception as e:
            logger.error(f"Error saving keys: {e}")
        
        return private_key, public_key
    
    def sign_data(self, data: bytes) -> bytes:
        """
        Sign data using the private key.
        
        Args:
            data: Data to sign
            
        Returns:
            Signature
        """
        if not self.private_key:
            raise ValueError("Private key not available")
        
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """
        Verify a signature using the public key.
        
        Args:
            data: Data that was signed
            signature: Signature to verify
            
        Returns:
            Whether the signature is valid
        """
        if not self.public_key:
            raise ValueError("Public key not available")
        
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def export_public_key(self) -> bytes:
        """
        Export the public key in PEM format.
        
        Returns:
            Public key bytes
        """
        if not self.public_key:
            raise ValueError("Public key not available")
        
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def import_public_key(self, key_data: bytes) -> bool:
        """
        Import a public key in PEM format.
        
        Args:
            key_data: Public key bytes
            
        Returns:
            Whether the key was imported successfully
        """
        try:
            self.public_key = load_pem_public_key(key_data)
            
            # Save the key
            with open(self.public_key_path, "wb") as f:
                f.write(key_data)
            
            logger.info("Imported public key")
            return True
        except Exception as e:
            logger.error(f"Error importing public key: {e}")
            return False

class PatchVerifier:
    """
    Verify patches using cryptographic signatures.
    """
    
    def __init__(self, key_manager: KeyManager = None):
        """
        Initialize patch verifier.
        
        Args:
            key_manager: Key manager instance
        """
        self.key_manager = key_manager or KeyManager()
        logger.info("Patch verifier initialized")
    
    def verify_patch(self, patch_data: bytes, signature: bytes) -> bool:
        """
        Verify a patch using its signature.
        
        Args:
            patch_data: Patch data
            signature: Signature of the patch
            
        Returns:
            Whether the patch is verified
        """
        return self.key_manager.verify_signature(patch_data, signature)
    
    def verify_patch_file(self, patch_file: str, signature_file: str = None) -> bool:
        """
        Verify a patch file using its signature file.
        
        Args:
            patch_file: Path to the patch file
            signature_file: Path to the signature file (or None to use patch_file + ".sig")
            
        Returns:
            Whether the patch is verified
        """
        if not signature_file:
            signature_file = patch_file + ".sig"
        
        if not os.path.exists(patch_file):
            logger.error(f"Patch file not found: {patch_file}")
            return False
        
        if not os.path.exists(signature_file):
            logger.error(f"Signature file not found: {signature_file}")
            return False
        
        try:
            # Read patch data
            with open(patch_file, "rb") as f:
                patch_data = f.read()
            
            # Read signature
            with open(signature_file, "rb") as f:
                signature = f.read()
            
            # Verify signature
            return self.verify_patch(patch_data, signature)
        except Exception as e:
            logger.error(f"Error verifying patch file: {e}")
            return False

class PatchSigner:
    """
    Sign patches using cryptographic signatures.
    """
    
    def __init__(self, key_manager: KeyManager = None):
        """
        Initialize patch signer.
        
        Args:
            key_manager: Key manager instance
        """
        self.key_manager = key_manager or KeyManager()
        logger.info("Patch signer initialized")
    
    def sign_patch(self, patch_data: bytes) -> bytes:
        """
        Sign a patch.
        
        Args:
            patch_data: Patch data
            
        Returns:
            Signature
        """
        return self.key_manager.sign_data(patch_data)
    
    def sign_patch_file(self, patch_file: str, signature_file: str = None) -> bool:
        """
        Sign a patch file and save the signature.
        
        Args:
            patch_file: Path to the patch file
            signature_file: Path to save the signature (or None to use patch_file + ".sig")
            
        Returns:
            Whether the patch was signed successfully
        """
        if not signature_file:
            signature_file = patch_file + ".sig"
        
        if not os.path.exists(patch_file):
            logger.error(f"Patch file not found: {patch_file}")
            return False
        
        try:
            # Read patch data
            with open(patch_file, "rb") as f:
                patch_data = f.read()
            
            # Sign patch
            signature = self.sign_patch(patch_data)
            
            # Save signature
            with open(signature_file, "wb") as f:
                f.write(signature)
            
            logger.info(f"Signed patch file {patch_file} and saved signature to {signature_file}")
            return True
        except Exception as e:
            logger.error(f"Error signing patch file: {e}")
            return False

class PatchMetadata:
    """
    Metadata for a patch, including its signature and verification status.
    """
    
    def __init__(self, patch_id: str = None, author: str = None, timestamp: float = None,
                 description: str = None, target_files: List[str] = None, 
                 signature: bytes = None, verification_status: bool = None):
        """
        Initialize patch metadata.
        
        Args:
            patch_id: Unique patch identifier
            author: Author of the patch
            timestamp: Creation timestamp
            description: Description of the patch
            target_files: Files affected by the patch
            signature: Cryptographic signature
            verification_status: Whether the patch has been verified
        """
        self.patch_id = patch_id or str(uuid.uuid4())
        self.author = author
        self.timestamp = timestamp or time.time()
        self.description = description
        self.target_files = target_files or []
        self.signature = signature
        self.verification_status = verification_status
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "patch_id": self.patch_id,
            "author": self.author,
            "timestamp": self.timestamp,
            "description": self.description,
            "target_files": self.target_files,
            "signature": base64.b64encode(self.signature).decode('utf-8') if self.signature else None,
            "verification_status": self.verification_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatchMetadata':
        """
        Create patch metadata from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Patch metadata
        """
        signature = None
        if data.get("signature"):
            signature = base64.b64decode(data["signature"])
        
        return cls(
            patch_id=data.get("patch_id"),
            author=data.get("author"),
            timestamp=data.get("timestamp"),
            description=data.get("description"),
            target_files=data.get("target_files"),
            signature=signature,
            verification_status=data.get("verification_status")
        )

class SecurePatch:
    """
    A secure patch with verification capabilities.
    """
    
    def __init__(self, patch_data: bytes, metadata: PatchMetadata = None):
        """
        Initialize secure patch.
        
        Args:
            patch_data: Patch data
            metadata: Patch metadata
        """
        self.patch_data = patch_data
        self.metadata = metadata or PatchMetadata()
        self.hash = hashlib.sha256(patch_data).hexdigest()
    
    def sign(self, signer: PatchSigner, author: str = None) -> bool:
        """
        Sign the patch.
        
        Args:
            signer: Patch signer
            author: Author of the patch
            
        Returns:
            Whether the patch was signed successfully
        """
        try:
            self.metadata.author = author
            self.metadata.signature = signer.sign_patch(self.patch_data)
            logger.info(f"Signed patch {self.metadata.patch_id}")
            return True
        except Exception as e:
            logger.error(f"Error signing patch: {e}")
            return False
    
    def verify(self, verifier: PatchVerifier) -> bool:
        """
        Verify the patch.
        
        Args:
            verifier: Patch verifier
            
        Returns:
            Whether the patch is verified
        """
        if not self.metadata.signature:
            logger.warning(f"Patch {self.metadata.patch_id} has no signature")
            self.metadata.verification_status = False
            return False
        
        try:
            result = verifier.verify_patch(self.patch_data, self.metadata.signature)
            self.metadata.verification_status = result
            logger.info(f"Verified patch {self.metadata.patch_id}: {'success' if result else 'failed'}")
            return result
        except Exception as e:
            logger.error(f"Error verifying patch: {e}")
            self.metadata.verification_status = False
            return False
    
    def save(self, patch_file: str, save_signature: bool = True) -> bool:
        """
        Save the patch and its metadata.
        
        Args:
            patch_file: Path to save the patch
            save_signature: Whether to save the signature separately
            
        Returns:
            Whether the patch was saved successfully
        """
        try:
            # Save patch data
            with open(patch_file, "wb") as f:
                f.write(self.patch_data)
            
            # Save metadata
            metadata_file = patch_file + ".meta"
            with open(metadata_file, "w") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            
            # Save signature separately if requested
            if save_signature and self.metadata.signature:
                signature_file = patch_file + ".sig"
                with open(signature_file, "wb") as f:
                    f.write(self.metadata.signature)
            
            logger.info(f"Saved patch to {patch_file} with metadata")
            return True
        except Exception as e:
            logger.error(f"Error saving patch: {e}")
            return False
    
    @classmethod
    def load(cls, patch_file: str) -> Optional['SecurePatch']:
        """
        Load a patch and its metadata.
        
        Args:
            patch_file: Path to the patch file
            
        Returns:
            Secure patch, or None if loading failed
        """
        if not os.path.exists(patch_file):
            logger.error(f"Patch file not found: {patch_file}")
            return None
        
        try:
            # Load patch data
            with open(patch_file, "rb") as f:
                patch_data = f.read()
            
            # Load metadata if available
            metadata = None
            metadata_file = patch_file + ".meta"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)
                    metadata = PatchMetadata.from_dict(metadata_dict)
            
            # If no metadata, check for separate signature
            if not metadata:
                metadata = PatchMetadata()
                signature_file = patch_file + ".sig"
                if os.path.exists(signature_file):
                    with open(signature_file, "rb") as f:
                        metadata.signature = f.read()
            
            return cls(patch_data, metadata)
        except Exception as e:
            logger.error(f"Error loading patch: {e}")
            return None

def create_key_manager(keys_dir: str = None) -> KeyManager:
    """
    Create a key manager.
    
    Args:
        keys_dir: Directory for storing keys
        
    Returns:
        Key manager
    """
    return KeyManager(keys_dir)

def create_patch_verifier(key_manager: KeyManager = None) -> PatchVerifier:
    """
    Create a patch verifier.
    
    Args:
        key_manager: Key manager instance
        
    Returns:
        Patch verifier
    """
    return PatchVerifier(key_manager)

def create_patch_signer(key_manager: KeyManager = None) -> PatchSigner:
    """
    Create a patch signer.
    
    Args:
        key_manager: Key manager instance
        
    Returns:
        Patch signer
    """
    return PatchSigner(key_manager)

def create_secure_patch(patch_data: bytes, author: str = None, 
                       description: str = None) -> SecurePatch:
    """
    Create a secure patch.
    
    Args:
        patch_data: Patch data
        author: Author of the patch
        description: Description of the patch
        
    Returns:
        Secure patch
    """
    metadata = PatchMetadata(
        author=author,
        description=description
    )
    return SecurePatch(patch_data, metadata)

def sign_patch_file(patch_file: str, author: str = None, 
                   description: str = None) -> bool:
    """
    Sign a patch file.
    
    Args:
        patch_file: Path to the patch file
        author: Author of the patch
        description: Description of the patch
        
    Returns:
        Whether the patch was signed successfully
    """
    if not os.path.exists(patch_file):
        logger.error(f"Patch file not found: {patch_file}")
        return False
    
    try:
        # Read patch data
        with open(patch_file, "rb") as f:
            patch_data = f.read()
        
        # Create secure patch
        patch = create_secure_patch(patch_data, author, description)
        
        # Sign patch
        signer = create_patch_signer()
        if not patch.sign(signer, author):
            return False
        
        # Save patch with metadata and signature
        return patch.save(patch_file)
    except Exception as e:
        logger.error(f"Error signing patch file: {e}")
        return False

def verify_patch_file(patch_file: str) -> bool:
    """
    Verify a patch file.
    
    Args:
        patch_file: Path to the patch file
        
    Returns:
        Whether the patch is verified
    """
    # Load patch
    patch = SecurePatch.load(patch_file)
    if not patch:
        return False
    
    # Verify patch
    verifier = create_patch_verifier()
    return patch.verify(verifier)

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Patch Verification")
    parser.add_argument("--sign", help="Sign a patch file")
    parser.add_argument("--verify", help="Verify a patch file")
    parser.add_argument("--author", help="Author of the patch")
    parser.add_argument("--description", help="Description of the patch")
    
    args = parser.parse_args()
    
    if args.sign:
        success = sign_patch_file(args.sign, args.author, args.description)
        print(f"Signing patch: {'Success' if success else 'Failed'}")
    elif args.verify:
        success = verify_patch_file(args.verify)
        print(f"Verifying patch: {'Success' if success else 'Failed'}")
    else:
        parser.print_help()
