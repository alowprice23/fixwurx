#!/usr/bin/env python3
"""
test_patch_integrity.py
───────────────────────
Test script for the cryptographic patch integrity verification features.
Tests hash calculation, verification, tampering detection, and multi-algorithm support.

Key features tested:
1. Hash calculation with multiple algorithms (SHA-256, SHA-384, SHA-512)
2. Integrity verification for patch bundles
3. Tamper detection through hash verification
4. Hash logging and audit trail
"""

import os
import sys
import tempfile
import time
import json
import hashlib
import subprocess
import tarfile
from pathlib import Path

# Add project root to path if needed
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# Import from both patch_bundle and rollback_manager
from patch_bundle import (
    create_bundle,
    verify_bundle,
    apply_bundle,
    calculate_hash,
    verify_content_hash,
    HashAlgorithm,
    IntegrityError,
    BundleError,
    BUNDLE_DIR,
    HASH_LOG_FILE
)

from rollback_manager import (
    register_patch, 
    rollback_patch, 
    _calculate_patch_hash, 
    _verify_patch_hash,
    RollbackError,
    ROLLBACK_DIR
)

# Test functions
def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_test_result(name, success):
    """Print test result."""
    result = "✅ PASSED" if success else "❌ FAILED"
    print(f"{result} - {name}")

def test_hash_calculation():
    """Test that hash calculation is consistent and supports multiple algorithms."""
    print_header("Testing Hash Calculation")
    
    # Test various contents
    test_cases = [
        "Simple patch content",
        "Multi-line\npatch content\nwith special chars: !@#$%^&*()",
        "@@ -1,5 +1,5 @@\n-Old line\n+New line\n Common context",
        # Add a realistic patch example
        """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -10,7 +10,7 @@
 def main():
-    print("Hello World")
+    print("Hello Universe")
     return 0
 
 if __name__ == "__main__":
"""
    ]
    
    # Test rollback_manager's hash calculation
    print("\n>> Testing rollback_manager hash calculation:")
    for i, content in enumerate(test_cases):
        # Calculate hash twice to verify consistency
        hash1 = _calculate_patch_hash(content)
        hash2 = _calculate_patch_hash(content)
        
        # Check that hash is consistent
        is_consistent = hash1 == hash2 and len(hash1) == 64  # SHA-256 is 64 hex chars
        print_test_result(f"rollback_manager hash consistency (case {i+1})", is_consistent)
        
        # Show the hash for reference
        print(f"  Content length: {len(content)} chars")
        print(f"  Hash: {hash1}")
    
    # Test different content produces different hashes (rollback_manager)
    are_different = len(set(_calculate_patch_hash(case) for case in test_cases)) == len(test_cases)
    print_test_result("rollback_manager: Different content produces different hashes", are_different)
    
    # Test patch_bundle's hash calculation with multiple algorithms
    print("\n>> Testing patch_bundle hash calculation with multiple algorithms:")
    algorithms = [HashAlgorithm.SHA256, HashAlgorithm.SHA384, HashAlgorithm.SHA512]
    hash_lengths = {HashAlgorithm.SHA256: 64, HashAlgorithm.SHA384: 96, HashAlgorithm.SHA512: 128}
    
    for algorithm in algorithms:
        print(f"\n  Algorithm: {algorithm}")
        
        for i, content in enumerate(test_cases[:2]):  # Test with first two cases
            # Calculate hash twice to verify consistency
            hash1 = calculate_hash(content, algorithm)
            hash2 = calculate_hash(content, algorithm)
            
            # Check that hash is consistent and has correct length
            expected_length = hash_lengths[algorithm]
            is_consistent = hash1 == hash2 and len(hash1) == expected_length
            
            print_test_result(f"{algorithm} consistency (case {i+1})", is_consistent)
            print(f"  Length: {len(hash1)} chars (expected {expected_length})")
            print(f"  Hash: {hash1}")
        
        # Verify output matches standard hashlib for this algorithm
        if algorithm == HashAlgorithm.SHA256:
            std_hash = hashlib.sha256(test_cases[0].encode()).hexdigest()
            our_hash = calculate_hash(test_cases[0], algorithm)
            match = std_hash == our_hash
            print_test_result(f"{algorithm} matches standard hashlib", match)
            if not match:
                print(f"  Expected: {std_hash}")
                print(f"  Got: {our_hash}")
    
    # Test that verify_content_hash works correctly
    print("\n>> Testing hash verification:")
    for algorithm in algorithms:
        content = test_cases[0]
        correct_hash = calculate_hash(content, algorithm)
        wrong_hash = "0" * len(correct_hash)
        
        # Test correct hash verification
        is_verified = verify_content_hash(content, correct_hash, algorithm)
        print_test_result(f"{algorithm} correct hash verification", is_verified)
        
        # Test incorrect hash rejection
        is_rejected = not verify_content_hash(content, wrong_hash, algorithm)
        print_test_result(f"{algorithm} incorrect hash rejection", is_rejected)

def test_rollback_manager_hash_storage():
    """Test that rollback_manager patch registration stores hashes correctly."""
    print_header("Testing Rollback Manager Hash Storage")
    
    # Create a test patch
    bug_id = f"test-hash-{int(time.time())}"
    patch_content = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,5 +1,5 @@
 def test_function():
-    return "old value"
+    return "new value"
"""
    
    # Calculate the expected hash
    expected_hash = _calculate_patch_hash(patch_content)
    print(f"Expected hash: {expected_hash}")
    
    try:
        # Register the patch
        patch_path = register_patch(bug_id, patch_content)
        print(f"Registered patch at: {patch_path}")
        
        # Read the registry file
        import json
        registry_path = ROLLBACK_DIR / "registry.json"
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        # Check that the bug ID is in the registry
        if bug_id not in registry:
            print_test_result("Patch registration in registry", False)
            print(f"  Bug ID {bug_id} not found in registry")
            return
        
        # Check registry format
        patch_info = registry[bug_id]
        
        if isinstance(patch_info, dict) and "hash" in patch_info:
            stored_hash = patch_info["hash"]
            print(f"Stored hash: {stored_hash}")
            
            # Check that the stored hash matches the expected hash
            hash_matches = stored_hash == expected_hash
            print_test_result("Stored hash matches expected hash", hash_matches)
            
            if not hash_matches:
                print(f"  Expected: {expected_hash}")
                print(f"  Actual:   {stored_hash}")
        else:
            print_test_result("Registry uses new format with hash", False)
            print(f"  Registry entry: {patch_info}")
    
    except Exception as e:
        print_test_result("Patch registration", False)
        print(f"  Error: {e}")

def test_rollback_manager_hash_verification():
    """Test hash verification during rollback."""
    print_header("Testing Hash Verification During Rollback")
    
    # Create a temporary file that we'll patch
    test_file = None
    temp_dir = None
    
    try:
        # Create a temporary directory for our test
        temp_dir = tempfile.TemporaryDirectory()
        test_file_path = Path(temp_dir.name) / "test_file.txt"
        
        # Create the original file
        original_content = "Original line 1\nOriginal line 2\nOriginal line 3\n"
        test_file_path.write_text(original_content)
        
        # Create a patch to modify the file
        patch_content = f"""diff --git a/{test_file_path.name} b/{test_file_path.name}
index abcdef1..abcdef2 100644
--- a/{test_file_path.name}
+++ b/{test_file_path.name}
@@ -1,3 +1,3 @@
 Original line 1
-Original line 2
+Modified line 2
 Original line 3
"""
        
        # Register the patch
        bug_id = f"test-verify-{int(time.time())}"
        patch_path = register_patch(bug_id, patch_content)
        
        # Manually apply the patch to simulate the effect of the patch
        # (We can't use git apply in this test as we're not in a git repo)
        modified_content = "Original line 1\nModified line 2\nOriginal line 3\n"
        test_file_path.write_text(modified_content)
        
        print(f"Registered patch for bug {bug_id} at {patch_path}")
        
        # Now let's test tampering detection by modifying the patch file
        tampered_bug_id = f"test-tamper-{int(time.time())}"
        tampered_patch_path = register_patch(tampered_bug_id, patch_content)
        
        # Tamper with the patch content
        tampered_content = patch_content.replace("Modified line 2", "Tampered line 2!!!")
        tampered_patch_path.write_text(tampered_content)
        
        print(f"Registered patch for bug {tampered_bug_id} at {tampered_patch_path}")
        print(f"Tampered with patch content at {tampered_patch_path}")
        
        # Test rollback with valid hash (should succeed)
        # We need to patch the _git_apply function in rollback_manager module, not globals()
        try:
            # Import the module to access its internals
            import rollback_manager
            
            # Save the original _git_apply function
            original_git_apply = rollback_manager._git_apply
            
            # Define a mock function
            def mock_git_apply(extra_args):
                # Just do nothing and simulate success
                print(f"Mock git apply called with args: {extra_args}")
                return
            
            # Replace the function in the module
            rollback_manager._git_apply = mock_git_apply
            
            # Try to rollback the valid patch
            print("Attempting rollback with valid hash...")
            rollback_patch(bug_id)
            print_test_result("Rollback with valid hash", True)
            
            # Try to rollback the tampered patch
            print("Attempting rollback with tampered patch...")
            tampering_detected = False
            try:
                rollback_patch(tampered_bug_id)
                print_test_result("Tampering detection", False)
                print("  Rollback succeeded but should have failed due to hash mismatch")
            except RollbackError as e:
                tampering_detected = "hash verification failed" in str(e).lower()
                print_test_result("Tampering detection", tampering_detected)
                if tampering_detected:
                    print(f"  Expected error detected: {e}")
                else:
                    print(f"  Rollback failed but with wrong error: {e}")
            
        finally:
            # Restore the original _git_apply function
            rollback_manager._git_apply = original_git_apply
    
    except Exception as e:
        print(f"Error in test: {e}")
    
    finally:
        # Clean up
        if temp_dir:
            temp_dir.cleanup()

def test_patch_bundle_integrity():
    """Test patch bundle creation, verification, and integrity checks."""
    print_header("Testing Patch Bundle Integrity")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create a test diff file
        diff_content = """diff --git a/test_file.txt b/test_file.txt
index 1234567..abcdefg 100644
--- a/test_file.txt
+++ b/test_file.txt
@@ -1,5 +1,5 @@
 Test line 1
-Test line 2 old
+Test line 2 new
 Test line 3
"""
        diff_file = test_dir / "test.diff"
        diff_file.write_text(diff_content)
        
        # Test creating bundles with different hash algorithms
        for algorithm in [HashAlgorithm.SHA256, HashAlgorithm.SHA384, HashAlgorithm.SHA512]:
            # Create unique bug ID with timestamp to avoid conflicts
            timestamp = int(time.time() * 1000)  # Millisecond precision
            bug_id = f"test-integrity-{algorithm}-{timestamp}"
            
            try:
                # Create bundle with specific algorithm
                bundle_path = create_bundle(bug_id, diff_content, algorithm)
                print_test_result(f"Create bundle with {algorithm}", bundle_path.exists())
                
                # Verify bundle
                manifest = verify_bundle(bundle_path)
                
                # Check that hash algorithm is stored
                stored_algorithm = manifest.get("hash_algorithm", "")
                algorithm_stored = stored_algorithm == algorithm
                print_test_result(f"Hash algorithm {algorithm} stored in manifest", algorithm_stored)
                
                # Check that content hash is correct
                expected_hash = calculate_hash(diff_content, algorithm)
                stored_hash = manifest.get("content_hash", "")
                hash_correct = stored_hash == expected_hash
                print_test_result(f"Content hash correct for {algorithm}", hash_correct)
                
                # Test tampering detection
                # Create a temporary copy of the bundle
                tampered_path = test_dir / f"{bug_id}_tampered.tri.tgz"
                
                # Extract, modify, and repack to simulate tampering
                with tarfile.open(bundle_path, "r:gz") as src_tf:
                    # Extract files
                    src_tf.extractall(test_dir)
                
                # Modify the patch content
                patch_path = test_dir / "patch.diff"
                tampered_content = diff_content.replace("Test line 2 new", "Test line 2 TAMPERED")
                patch_path.write_text(tampered_content)
                
                # Repack without updating the hash
                with tarfile.open(tampered_path, "w:gz") as tf:
                    # Add original manifest (with old hash)
                    manifest_path = test_dir / "manifest.json"
                    tf.add(manifest_path, arcname="manifest.json")
                    
                    # Add modified patch
                    tf.add(patch_path, arcname="patch.diff")
                
                # Verify should fail for tampered bundle
                try:
                    verify_bundle(tampered_path)
                    print_test_result(f"Tampering detection for {algorithm}", False)
                    print("  Failed: Tampered bundle passed verification")
                except IntegrityError as e:
                    print_test_result(f"Tampering detection for {algorithm}", True)
                    print(f"  Correctly detected: {e}")
                except Exception as e:
                    print_test_result(f"Tampering detection for {algorithm}", False)
                    print(f"  Wrong exception: {type(e).__name__}: {e}")
            
            except Exception as e:
                print_test_result(f"Bundle tests with {algorithm}", False)
                print(f"  Error: {type(e).__name__}: {e}")


def test_hash_logging():
    """Test that hash verification attempts are logged."""
    print_header("Testing Hash Verification Logging")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test content and hash
        test_content = "Test content for hash logging"
        correct_hash = calculate_hash(test_content)
        wrong_hash = "0" * len(correct_hash)
        
        # Create a temporary log file for testing
        test_log_file = test_dir / "hash_log.txt"
        
        # Import the module to access and modify its globals
        import patch_bundle
        original_hash_log_file = patch_bundle.HASH_LOG_FILE
        
        try:
            # Redirect log file in the module
            patch_bundle.HASH_LOG_FILE = test_log_file
            
            # Create directory for log file
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Perform verifications
            verify_content_hash(test_content, correct_hash)  # Should pass
            verify_content_hash(test_content, wrong_hash)    # Should fail
            
            # Check that log file exists and contains entries
            log_exists = test_log_file.exists()
            print_test_result("Hash verification log created", log_exists)
            
            if log_exists:
                # Read the log file with binary mode and handle encoding issues
                with open(test_log_file, 'rb') as f:
                    log_bytes = f.read()
                
                # Convert to string for checking content
                log_content = log_bytes.decode('utf-8', errors='replace')
                
                # Check for verification indicators (handling encoding issues)
                has_success = "verified" in log_content
                has_failure = "FAILED" in log_content
                
                print_test_result("Log contains successful verification", has_success)
                print_test_result("Log contains failed verification", has_failure)
                
                # Display log contents
                print("\nVerification log contents:")
                print(log_content)
        
        finally:
            # Restore original log file
            patch_bundle.HASH_LOG_FILE = original_hash_log_file


def test_cli_hash_commands():
    """Test CLI hash commands by directly calling the functions they use."""
    print_header("Testing CLI Hash Commands")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create a test file
        test_file = test_dir / "test_file.txt"
        test_content = "Test content for CLI hash command testing"
        test_file.write_text(test_content)
        
        # Test hash calculation with different algorithms
        for algorithm in [HashAlgorithm.SHA256, HashAlgorithm.SHA384, HashAlgorithm.SHA512]:
            # Calculate hash with our function
            file_hash = calculate_hash(test_content, algorithm)
            
            # Calculate hash with standard hashlib for verification
            hasher = hashlib.new(algorithm)
            hasher.update(test_content.encode())
            expected_hash = hasher.hexdigest()
            
            # Verify hashes match
            match = file_hash == expected_hash
            print_test_result(f"CLI hash command with {algorithm}", match)
            
            if not match:
                print(f"  Expected: {expected_hash}")
                print(f"  Got: {file_hash}")
            
        # Test passing invalid algorithm (should default to sha256)
        invalid_algorithm = "invalid_algo"
        file_hash = calculate_hash(test_content, invalid_algorithm)
        
        # Should match sha256
        hasher = hashlib.new("sha256")
        hasher.update(test_content.encode())
        expected_hash = hasher.hexdigest()
        
        match = file_hash == expected_hash
        print_test_result("Invalid algorithm defaults to SHA-256", match)


def main():
    """Run all tests."""
    print("=" * 80)
    print(" PATCH INTEGRITY VERIFICATION TEST SUITE ".center(80, "="))
    print("=" * 80)
    
    test_hash_calculation()
    test_rollback_manager_hash_storage()
    test_rollback_manager_hash_verification()
    test_patch_bundle_integrity()
    test_hash_logging()
    test_cli_hash_commands()
    
    print("\n" + "=" * 80)
    print(" TEST SUITE COMPLETE ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    main()
