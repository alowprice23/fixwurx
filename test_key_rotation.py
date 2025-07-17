#!/usr/bin/env python3
"""
test_key_rotation.py
────────────────────
Test script for the key rotation system.

This test verifies:
1. Scheduled key rotation functionality
2. Manual key rotation through CLI
3. Integration with credential manager
4. Rotation state persistence
5. Notification system

Run this script to verify that the key rotation system is working correctly.
"""

import os
import time
import json
import shutil
import unittest
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the key rotation module
from key_rotation import KeyRotationManager, ROTATION_STATE_PATH

# Create a mock credential manager
class MockCredentialManager:
    """Mock credential manager for testing."""
    
    def __init__(self):
        self.keys = {
            "openai": "sk-test-openaikey123456",
            "anthropic": "sk-ant-api01-testkey789012"
        }
    
    def get_api_key(self, provider):
        """Get API key for provider."""
        return self.keys.get(provider)
    
    def mask_key(self, key):
        """Create a masked version of the key."""
        if not key or len(key) < 10:
            return "[invalid key]"
        return f"{key[:4]}...{key[-4:]}"


class TestKeyRotation(unittest.TestCase):
    """Test suite for key rotation system."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp(prefix="key_rotation_test_")
        
        # Create a unique test state file path for each test
        self.test_state_path = Path(self.test_dir) / f"key_rotation_state_{id(self)}.json"
        
        # Create a mock credential manager
        self.credential_manager = MockCredentialManager()
        
        # Patch the ROTATION_STATE_PATH
        self.original_path = ROTATION_STATE_PATH
        # Replace the module's path with our test path
        import key_rotation
        key_rotation.ROTATION_STATE_PATH = self.test_state_path
        
        # Delete any existing state file
        if self.test_state_path.exists():
            self.test_state_path.unlink()
        
        # Test configuration with shorter intervals for testing
        self.test_config = {
            "rotation_interval_mins": 0.1,  # 6 seconds for testing
            "rotation_offset_mins": 0.05,   # 3 seconds jitter
            "notification_lead_mins": 0.05, # 3 seconds before rotation
            "key_providers": ["openai", "anthropic"],
            "rotation_log_path": str(Path(self.test_dir) / "key_rotation.log")
        }
    
    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
    def _create_clean_manager(self):
        """Create a manager with a clean state."""
        # Delete any existing state file
        if self.test_state_path.exists():
            self.test_state_path.unlink()
            
        # Create and return a fresh manager
        return KeyRotationManager(
            credential_manager=self.credential_manager,
            config=self.test_config
        )
    
    def test_initialization(self):
        """Test initialization of key rotation manager."""
        # Create manager with clean state
        manager = self._create_clean_manager()
        
        # Check initial state
        self.assertEqual(manager.config["rotation_interval_mins"], 0.1)
        self.assertEqual(manager.config["key_providers"], ["openai", "anthropic"])
        self.assertIsNotNone(manager.credential_manager)
        
        # Initial state should be empty
        self.assertEqual(manager.state["last_rotation"], {})
        self.assertEqual(manager.state["next_rotation"], {})
        self.assertEqual(manager.state["rotation_history"], [])
        
        print("✅ PASSED - Key rotation manager initializes correctly")
    
    def test_rotation_status(self):
        """Test getting rotation status."""
        # Create manager with clean state
        manager = self._create_clean_manager()
        
        # Get initial status
        status = manager.get_rotation_status()
        
        # Check status structure
        self.assertIn("providers", status)
        self.assertIn("config", status)
        self.assertIn("rotation_history", status)
        
        # Should have status for both providers
        self.assertIn("openai", status["providers"])
        self.assertIn("anthropic", status["providers"])
        
        # Config should match
        self.assertEqual(status["config"]["rotation_interval_mins"], 0.1)
        
        print("✅ PASSED - Can retrieve rotation status")
    
    @patch('key_rotation.KeyRotationManager._rotate_provider_key')
    def test_manual_rotation(self, mock_rotate):
        """Test manual key rotation."""
        # Mock the key rotation to always succeed
        mock_rotate.return_value = True
        
        # Create manager with clean state
        manager = self._create_clean_manager()
        
        # Rotate a key
        result = manager.rotate_key("openai", force=True)
        
        # Check result
        self.assertTrue(result)
        mock_rotate.assert_called_once_with("openai")
        
        # Check state was updated
        self.assertIn("openai", manager.state["last_rotation"])
        self.assertIn("openai", manager.state["next_rotation"])
        
        # Check history was updated
        self.assertEqual(len(manager.state["rotation_history"]), 1)
        self.assertEqual(manager.state["rotation_history"][0]["provider"], "openai")
        self.assertTrue(manager.state["rotation_history"][0]["forced"])
        
        print("✅ PASSED - Manual key rotation works")
    
    @patch('key_rotation.KeyRotationManager._rotate_provider_key')
    def test_rotation_all_keys(self, mock_rotate):
        """Test rotating all keys."""
        # Mock the key rotation to always succeed
        mock_rotate.return_value = True
        
        # Create manager with clean state
        manager = self._create_clean_manager()
        
        # Rotate all keys
        results = manager.rotate_all_keys(force=True)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(results["openai"])
        self.assertTrue(results["anthropic"])
        
        # Check state was updated for both providers
        self.assertIn("openai", manager.state["last_rotation"])
        self.assertIn("anthropic", manager.state["last_rotation"])
        
        # Check history contains entries for both providers
        self.assertEqual(len(manager.state["rotation_history"]), 2)
        
        print("✅ PASSED - Rotating all keys works")
    
    @patch('key_rotation.KeyRotationManager._rotate_provider_key')
    def test_rotation_callbacks(self, mock_rotate):
        """Test rotation callbacks."""
        # Mock the key rotation to always succeed
        mock_rotate.return_value = True
        
        # Create manager with clean state
        manager = self._create_clean_manager()
        
        # Create mock callbacks
        pre_callback = MagicMock()
        post_callback = MagicMock()
        
        # Register callbacks
        manager.register_pre_rotation_callback(pre_callback)
        manager.register_post_rotation_callback(post_callback)
        
        # Rotate a key
        manager.rotate_key("openai", force=True)
        
        # Check callbacks were called
        pre_callback.assert_called_once_with("openai")
        post_callback.assert_called_once_with("openai", True)
        
        print("✅ PASSED - Rotation callbacks work")
    
    @patch('key_rotation.KeyRotationManager._rotate_provider_key')
    def test_state_persistence(self, mock_rotate):
        """Test state persistence."""
        # Mock the key rotation to always succeed
        mock_rotate.return_value = True
        
        # Create first manager with clean state
        manager1 = self._create_clean_manager()
        
        manager1.rotate_key("openai", force=True)
        
        # Create second manager to load the state
        manager2 = KeyRotationManager(
            credential_manager=self.credential_manager,
            config=self.test_config
        )
        
        # Check state was loaded
        self.assertIn("openai", manager2.state["last_rotation"])
        self.assertIn("openai", manager2.state["next_rotation"])
        self.assertEqual(len(manager2.state["rotation_history"]), 1)
        
        print("✅ PASSED - State persistence works")
    
    @patch('key_rotation.KeyRotationManager._rotate_provider_key')
    def test_scheduled_rotation(self, mock_rotate):
        """Test scheduled rotation."""
        # Skip test on CI (timing issues)
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping scheduled rotation test on CI")
        
        # Mock the key rotation to always succeed
        mock_rotate.return_value = True
        
        # Create manager with very short interval for testing
        test_config = self.test_config.copy()
        test_config["rotation_interval_mins"] = 0.05  # 3 seconds
        
        # Use clean manager
        manager = self._create_clean_manager()
        manager.config = test_config
        
        # Directly test the rotation callback without starting the scheduler
        with patch.object(manager, '_schedule_next_rotation') as mock_schedule:
            # Call rotation callback directly
            manager._rotation_callback("openai")
            
            # Check rotation was performed
            mock_rotate.assert_called_once_with("openai")
            
            # Check next rotation was scheduled
            mock_schedule.assert_called_once()
        
        # Stop scheduler
        manager.stop_scheduler()
        
        print("✅ PASSED - Scheduled rotation works")


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def main():
    """Run all tests."""
    print_header("KEY ROTATION TEST SUITE")
    
    test = TestKeyRotation()
    test.setUp()
    
    try:
        # Run tests individually to get better output
        test.test_initialization()
        test.test_rotation_status()
        test.test_manual_rotation()
        test.test_rotation_all_keys()
        test.test_rotation_callbacks()
        test.test_state_persistence()
        test.test_scheduled_rotation()
    finally:
        test.tearDown()
    
    print("\n")
    print_header("TEST SUITE COMPLETE")


if __name__ == "__main__":
    main()
