#!/usr/bin/env python3
"""
Test Script for Patch Verification Module

This script tests the functionality of the patch verification module by creating
keys, signing patches, and verifying patch signatures.
"""

import os
import sys
import json
import tempfile
import shutil
import base64
from pathlib import Path
from typing import Dict, Any

# Ensure the patch_verification module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from patch_verification import (
        KeyManager, PatchVerifier, PatchSigner, PatchMetadata, SecurePatch,
        create_key_manager, create_patch_verifier, create_patch_signer,
        create_secure_patch, sign_patch_file, verify_patch_file
    )
except ImportError:
    print("Error: Could not import patch_verification module")
    sys.exit(1)

def setup_test_dir():
    """Create a temporary test directory."""
    test_dir = tempfile.mkdtemp(prefix="patch_verification_test_")
    print(f"Created test directory: {test_dir}")
    return test_dir

def cleanup_test_dir(test_dir):
    """Clean up the temporary test directory."""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"Removed test directory: {test_dir}")

def create_test_patch(content="Test patch content"):
    """Create a test patch file."""
    return content.encode('utf-8')

def test_key_manager():
    """Test the KeyManager class."""
    print("\n=== Testing KeyManager ===")
    test_dir = setup_test_dir()
    
    try:
        # Initialize key manager
        keys_dir = os.path.join(test_dir, "keys")
        key_manager = KeyManager(keys_dir)
        print(f"Initialized key manager with keys in {keys_dir}")
        
        # Check if keys were generated
        assert os.path.exists(os.path.join(keys_dir, "private_key.pem")), "Private key not created"
        assert os.path.exists(os.path.join(keys_dir, "public_key.pem")), "Public key not created"
        
        # Sign data
        test_data = b"Hello, world!"
        signature = key_manager.sign_data(test_data)
        print(f"Signed test data, signature length: {len(signature)}")
        
        # Verify signature
        verified = key_manager.verify_signature(test_data, signature)
        print(f"Signature verification: {verified}")
        assert verified, "Signature verification failed"
        
        # Try with incorrect data
        modified_data = b"Hello, world!!"
        verified = key_manager.verify_signature(modified_data, signature)
        print(f"Modified data verification: {verified}")
        assert not verified, "Verification should fail with modified data"
        
        # Export and import public key
        public_key_data = key_manager.export_public_key()
        print(f"Exported public key, length: {len(public_key_data)}")
        
        # Create new key manager
        new_keys_dir = os.path.join(test_dir, "new_keys")
        new_key_manager = KeyManager(new_keys_dir)
        
        # Import public key
        success = new_key_manager.import_public_key(public_key_data)
        print(f"Imported public key: {success}")
        assert success, "Public key import failed"
        
        # Verify signature with imported key
        verified = new_key_manager.verify_signature(test_data, signature)
        print(f"Verification with imported key: {verified}")
        assert verified, "Verification with imported key failed"
        
        return True
    except Exception as e:
        print(f"Error in key manager test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_patch_verifier():
    """Test the PatchVerifier class."""
    print("\n=== Testing PatchVerifier ===")
    test_dir = setup_test_dir()
    
    try:
        # Create key manager and patch verifier
        keys_dir = os.path.join(test_dir, "keys")
        key_manager = KeyManager(keys_dir)
        verifier = PatchVerifier(key_manager)
        print("Created patch verifier")
        
        # Create patch signer
        signer = PatchSigner(key_manager)
        
        # Create test patch
        patch_data = create_test_patch()
        
        # Sign patch
        signature = signer.sign_patch(patch_data)
        print(f"Signed patch, signature length: {len(signature)}")
        
        # Save patch and signature
        patch_file = os.path.join(test_dir, "test_patch.patch")
        signature_file = patch_file + ".sig"
        
        with open(patch_file, "wb") as f:
            f.write(patch_data)
        
        with open(signature_file, "wb") as f:
            f.write(signature)
        
        # Verify patch
        verified = verifier.verify_patch(patch_data, signature)
        print(f"Patch verification: {verified}")
        assert verified, "Patch verification failed"
        
        # Verify patch file
        verified = verifier.verify_patch_file(patch_file)
        print(f"Patch file verification: {verified}")
        assert verified, "Patch file verification failed"
        
        # Modify patch data
        modified_patch_data = create_test_patch("Modified patch content")
        modified_patch_file = os.path.join(test_dir, "modified_patch.patch")
        
        with open(modified_patch_file, "wb") as f:
            f.write(modified_patch_data)
        
        with open(modified_patch_file + ".sig", "wb") as f:
            f.write(signature)
        
        # Verify modified patch
        verified = verifier.verify_patch(modified_patch_data, signature)
        print(f"Modified patch verification: {verified}")
        assert not verified, "Modified patch verification should fail"
        
        # Verify modified patch file
        verified = verifier.verify_patch_file(modified_patch_file)
        print(f"Modified patch file verification: {verified}")
        assert not verified, "Modified patch file verification should fail"
        
        return True
    except Exception as e:
        print(f"Error in patch verifier test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_patch_signer():
    """Test the PatchSigner class."""
    print("\n=== Testing PatchSigner ===")
    test_dir = setup_test_dir()
    
    try:
        # Create key manager and patch signer
        keys_dir = os.path.join(test_dir, "keys")
        key_manager = KeyManager(keys_dir)
        signer = PatchSigner(key_manager)
        print("Created patch signer")
        
        # Create test patch
        patch_data = create_test_patch()
        
        # Sign patch
        signature = signer.sign_patch(patch_data)
        print(f"Signed patch, signature length: {len(signature)}")
        assert len(signature) > 0, "Signature should not be empty"
        
        # Save patch
        patch_file = os.path.join(test_dir, "test_patch.patch")
        
        with open(patch_file, "wb") as f:
            f.write(patch_data)
        
        # Sign patch file
        success = signer.sign_patch_file(patch_file)
        print(f"Signed patch file: {success}")
        assert success, "Patch file signing failed"
        
        # Check if signature file was created
        signature_file = patch_file + ".sig"
        assert os.path.exists(signature_file), "Signature file not created"
        
        # Verify signature
        with open(signature_file, "rb") as f:
            signature_data = f.read()
        
        verifier = PatchVerifier(key_manager)
        verified = verifier.verify_patch(patch_data, signature_data)
        print(f"Signature verification: {verified}")
        assert verified, "Signature verification failed"
        
        return True
    except Exception as e:
        print(f"Error in patch signer test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_patch_metadata():
    """Test the PatchMetadata class."""
    print("\n=== Testing PatchMetadata ===")
    
    try:
        # Create patch metadata
        metadata = PatchMetadata(
            patch_id="test-patch-123",
            author="Test Author",
            description="Test patch description",
            target_files=["file1.py", "file2.py"]
        )
        print(f"Created patch metadata: {metadata.patch_id}")
        
        # Convert to dictionary
        metadata_dict = metadata.to_dict()
        print(f"Metadata dictionary keys: {list(metadata_dict.keys())}")
        
        # Check fields
        assert metadata_dict["patch_id"] == "test-patch-123", "Patch ID does not match"
        assert metadata_dict["author"] == "Test Author", "Author does not match"
        assert metadata_dict["description"] == "Test patch description", "Description does not match"
        assert metadata_dict["target_files"] == ["file1.py", "file2.py"], "Target files do not match"
        
        # Create from dictionary
        new_metadata = PatchMetadata.from_dict(metadata_dict)
        print(f"Created metadata from dictionary: {new_metadata.patch_id}")
        
        # Check fields
        assert new_metadata.patch_id == "test-patch-123", "Recreated patch ID does not match"
        assert new_metadata.author == "Test Author", "Recreated author does not match"
        assert new_metadata.description == "Test patch description", "Recreated description does not match"
        assert new_metadata.target_files == ["file1.py", "file2.py"], "Recreated target files do not match"
        
        # Test with signature
        signature = b"test-signature"
        metadata.signature = signature
        metadata.verification_status = True
        
        metadata_dict = metadata.to_dict()
        print(f"Metadata with signature: {metadata_dict['signature'] is not None}")
        
        # Create from dictionary with signature
        new_metadata = PatchMetadata.from_dict(metadata_dict)
        print(f"Recreated with signature: {new_metadata.signature is not None}")
        
        # Check signature
        assert new_metadata.signature == signature, "Signature does not match"
        assert new_metadata.verification_status is True, "Verification status does not match"
        
        return True
    except Exception as e:
        print(f"Error in patch metadata test: {e}")
        return False

def test_secure_patch():
    """Test the SecurePatch class."""
    print("\n=== Testing SecurePatch ===")
    test_dir = setup_test_dir()
    
    try:
        # Create key manager, signer, and verifier
        keys_dir = os.path.join(test_dir, "keys")
        key_manager = KeyManager(keys_dir)
        signer = PatchSigner(key_manager)
        verifier = PatchVerifier(key_manager)
        
        # Create test patch
        patch_data = create_test_patch()
        
        # Create metadata
        metadata = PatchMetadata(
            author="Test Author",
            description="Test patch description",
            target_files=["file1.py", "file2.py"]
        )
        
        # Create secure patch
        patch = SecurePatch(patch_data, metadata)
        print(f"Created secure patch: {patch.metadata.patch_id}")
        
        # Sign patch
        success = patch.sign(signer, "Test Author")
        print(f"Signed patch: {success}")
        assert success, "Patch signing failed"
        assert patch.metadata.signature is not None, "Signature not set"
        
        # Verify patch
        verified = patch.verify(verifier)
        print(f"Verified patch: {verified}")
        assert verified, "Patch verification failed"
        assert patch.metadata.verification_status is True, "Verification status not set"
        
        # Save patch
        patch_file = os.path.join(test_dir, "secure_patch.patch")
        success = patch.save(patch_file)
        print(f"Saved patch: {success}")
        assert success, "Patch saving failed"
        
        # Check if files were created
        assert os.path.exists(patch_file), "Patch file not created"
        assert os.path.exists(patch_file + ".meta"), "Metadata file not created"
        assert os.path.exists(patch_file + ".sig"), "Signature file not created"
        
        # Load patch
        loaded_patch = SecurePatch.load(patch_file)
        print(f"Loaded patch: {loaded_patch is not None}")
        assert loaded_patch is not None, "Patch loading failed"
        
        # Check metadata
        assert loaded_patch.metadata.patch_id == patch.metadata.patch_id, "Patch ID does not match"
        assert loaded_patch.metadata.author == "Test Author", "Author does not match"
        assert loaded_patch.metadata.description == "Test patch description", "Description does not match"
        
        # Verify loaded patch
        verified = loaded_patch.verify(verifier)
        print(f"Verified loaded patch: {verified}")
        assert verified, "Loaded patch verification failed"
        
        # Modify patch data
        modified_patch = SecurePatch(create_test_patch("Modified patch content"), metadata)
        modified_patch.metadata.signature = patch.metadata.signature  # Use original signature
        
        # Verify modified patch
        verified = modified_patch.verify(verifier)
        print(f"Verified modified patch: {verified}")
        assert not verified, "Modified patch verification should fail"
        
        return True
    except Exception as e:
        print(f"Error in secure patch test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_api_functions():
    """Test the API functions."""
    print("\n=== Testing API Functions ===")
    test_dir = setup_test_dir()
    
    try:
        # Test create_key_manager
        key_manager = create_key_manager(os.path.join(test_dir, "keys"))
        print(f"Created key manager: {key_manager is not None}")
        assert key_manager is not None, "Key manager creation failed"
        
        # Test create_patch_verifier
        verifier = create_patch_verifier(key_manager)
        print(f"Created patch verifier: {verifier is not None}")
        assert verifier is not None, "Patch verifier creation failed"
        
        # Test create_patch_signer
        signer = create_patch_signer(key_manager)
        print(f"Created patch signer: {signer is not None}")
        assert signer is not None, "Patch signer creation failed"
        
        # Test create_secure_patch
        patch_data = create_test_patch()
        patch = create_secure_patch(patch_data, "Test Author", "Test description")
        print(f"Created secure patch: {patch is not None}")
        assert patch is not None, "Secure patch creation failed"
        assert patch.metadata.author == "Test Author", "Author not set"
        assert patch.metadata.description == "Test description", "Description not set"
        
        # Test sign_patch_file
        patch_file = os.path.join(test_dir, "api_test.patch")
        with open(patch_file, "wb") as f:
            f.write(patch_data)
        
        success = sign_patch_file(patch_file, "API Test Author", "API test description")
        print(f"Signed patch file: {success}")
        assert success, "Patch file signing failed"
        
        # Check if files were created
        assert os.path.exists(patch_file + ".meta"), "Metadata file not created"
        assert os.path.exists(patch_file + ".sig"), "Signature file not created"
        
        # Test verify_patch_file
        verified = verify_patch_file(patch_file)
        print(f"Verified patch file: {verified}")
        assert verified, "Patch file verification failed"
        
        # Modify patch file
        modified_patch_file = os.path.join(test_dir, "modified_api_test.patch")
        with open(modified_patch_file, "wb") as f:
            f.write(create_test_patch("Modified API test patch"))
        
        # Copy metadata and signature
        shutil.copy(patch_file + ".meta", modified_patch_file + ".meta")
        shutil.copy(patch_file + ".sig", modified_patch_file + ".sig")
        
        # Verify modified patch file
        verified = verify_patch_file(modified_patch_file)
        print(f"Verified modified patch file: {verified}")
        assert not verified, "Modified patch file verification should fail"
        
        return True
    except Exception as e:
        print(f"Error in API functions test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def main():
    """Main function."""
    print("=== Patch Verification Test Suite ===")
    
    # Run tests
    tests = [
        ("KeyManager", test_key_manager),
        ("PatchVerifier", test_patch_verifier),
        ("PatchSigner", test_patch_signer),
        ("PatchMetadata", test_patch_metadata),
        ("SecurePatch", test_secure_patch),
        ("API Functions", test_api_functions)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\nRunning test: {name}")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error running test: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        
        print(f"{name}: {status}")
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
