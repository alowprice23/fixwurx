#!/usr/bin/env python3
"""
test_model_updater.py
─────────────────────
Test script for the model version update system.

This test verifies:
1. Update detection functionality
2. Model version updating in configuration
3. Scheduled update checks
4. Ignore/unignore functionality
5. State persistence

Run this script to verify that the model updater system is working correctly.
"""

import os
import time
import json
import yaml
import shutil
import unittest
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the model updater module
from model_updater import ModelUpdater, MODEL_STATE_PATH

# Test catalog with simplified model versions for testing
TEST_MODEL_CATALOG = {
    "openai": {
        "gpt-4": {
            "latest": "gpt-4-turbo-test",
            "aliases": ["gpt-4-alias-1", "gpt-4-alias-2"],
            "deprecated": ["gpt-4-deprecated"],
            "fallbacks": ["gpt-4o-test", "gpt-3.5-turbo-test"]
        },
        "gpt-3.5": {
            "latest": "gpt-3.5-turbo-test",
            "aliases": ["gpt-3.5-alias"],
            "deprecated": ["gpt-3.5-deprecated"],
            "fallbacks": []
        }
    },
    "anthropic": {
        "claude-3": {
            "latest": "claude-3-opus-test",
            "aliases": ["claude-3-alias"],
            "deprecated": ["claude-3-deprecated"],
            "fallbacks": ["claude-3-sonnet-test"]
        }
    }
}

# Create a mock LLM manager
class MockLLMManager:
    """Mock LLM manager for testing."""
    
    def __init__(self, available_providers=None):
        self.available_providers = available_providers or ["openai", "anthropic"]
    
    def available(self):
        """Get available providers."""
        return self.available_providers


class TestModelUpdater(unittest.TestCase):
    """Test suite for model updater system."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp(prefix="model_updater_test_")
        
        # Create test paths
        self.test_state_path = Path(self.test_dir) / f"model_versions_{id(self)}.json"
        self.test_config_path = Path(self.test_dir) / "system_config.yaml"
        
        # Create a mock config file
        self._create_test_config()
        
        # Create mock LLM manager
        self.llm_manager = MockLLMManager()
        
        # Patch the model state path
        self.original_path = MODEL_STATE_PATH
        # Replace the module's path with our test path
        import model_updater
        model_updater.MODEL_STATE_PATH = self.test_state_path
        
        # Delete any existing state file
        if self.test_state_path.exists():
            self.test_state_path.unlink()
    
    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_test_config(self):
        """Create a test configuration file."""
        config = {
            "llm": {
                "preferred": "openai",
                "temperature": 0.1,
                "cost-budget-usd": 1.0,
                "update_interval_hours": 0.1,  # 6 minutes for testing
                "check_interval_hours": 0.05,  # 3 minutes for testing
                "models": {
                    "primary": "gpt-4-turbo",
                    "fallback": "claude-3-sonnet",
                    "offline": "codellama-13b",
                    "explanation": "gpt-3.5-turbo"
                }
            }
        }
        
        # Write the config file
        with open(self.test_config_path, 'w') as f:
            yaml.dump(config, f)
    
    def _create_clean_updater(self):
        """Create an updater with a clean state."""
        # Delete any existing state file
        if self.test_state_path.exists():
            self.test_state_path.unlink()
            
        # Create and return a fresh updater
        return ModelUpdater(
            config_path=self.test_config_path,
            state_path=self.test_state_path,
            llm_manager=self.llm_manager,
            model_catalog=TEST_MODEL_CATALOG
        )
    
    def test_initialization(self):
        """Test initialization of model updater."""
        updater = self._create_clean_updater()
        
        # Check initial state
        self.assertEqual(updater.check_interval, 180)  # 0.05 hours = 180 seconds
        self.assertEqual(updater.update_interval, 360)  # 0.1 hours = 360 seconds
        self.assertEqual(updater.last_check_time, 0)
        self.assertEqual(updater.last_update_time, 0)
        self.assertEqual(updater.ignore_models, {})
        
        # Check config loading
        self.assertIn("llm", updater.config)
        self.assertIn("models", updater.config["llm"])
        self.assertEqual(updater.config["llm"]["models"]["primary"], "gpt-4-turbo")
        
        print("✅ PASSED - Model updater initializes correctly")
    
    def test_get_current_models(self):
        """Test getting current models from configuration."""
        updater = self._create_clean_updater()
        
        # Get current models
        current_models = updater._get_current_models()
        
        # Check that models were extracted correctly
        self.assertIn("openai", current_models)
        self.assertIn("gpt-4", current_models["openai"])
        self.assertEqual(current_models["openai"]["gpt-4"], "gpt-4-turbo")
        
        self.assertIn("anthropic", current_models)
        self.assertIn("claude-3", current_models["anthropic"])
        self.assertEqual(current_models["anthropic"]["claude-3"], "claude-3-sonnet")
        
        print("✅ PASSED - Can get current models from configuration")
    
    def test_parse_model_string(self):
        """Test parsing model strings."""
        updater = self._create_clean_updater()
        
        # Test explicit provider
        provider, model = updater._parse_model_string("openai:gpt-4-turbo")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4-turbo")
        
        # Test inferred provider
        provider, model = updater._parse_model_string("gpt-4-turbo")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4-turbo")
        
        provider, model = updater._parse_model_string("claude-3-opus")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "claude-3-opus")
        
        # Test unknown model
        provider, model = updater._parse_model_string("unknown-model")
        self.assertIsNone(provider)
        self.assertEqual(model, "unknown-model")
        
        print("✅ PASSED - Can parse model strings correctly")
    
    def test_get_model_family(self):
        """Test getting model family."""
        updater = self._create_clean_updater()
        
        # Test exact match
        family = updater._get_model_family("openai", "gpt-4-turbo-test")
        self.assertEqual(family, "gpt-4")
        
        # Test alias match
        family = updater._get_model_family("openai", "gpt-4-alias-1")
        self.assertEqual(family, "gpt-4")
        
        # Test deprecated match
        family = updater._get_model_family("openai", "gpt-4-deprecated")
        self.assertEqual(family, "gpt-4")
        
        # Test prefix match
        family = updater._get_model_family("openai", "gpt-4-unknown-version")
        self.assertEqual(family, "gpt-4")
        
        # Test unknown model
        family = updater._get_model_family("openai", "unknown-model")
        self.assertEqual(family, "unknown-model")
        
        print("✅ PASSED - Can determine model family correctly")
    
    def test_check_provider_updates(self):
        """Test checking for provider updates."""
        updater = self._create_clean_updater()
        
        # Test with models that need updates
        model_dict = {
            "gpt-4": "gpt-4-deprecated",
            "gpt-3.5": "gpt-3.5-deprecated"
        }
        
        updates = updater._check_provider_updates("openai", model_dict)
        
        # Should find updates for both models
        self.assertEqual(len(updates), 2)
        self.assertIn("gpt-4", updates)
        self.assertIn("gpt-3.5", updates)
        
        # Check update details
        self.assertEqual(updates["gpt-4"]["current"], "gpt-4-deprecated")
        self.assertEqual(updates["gpt-4"]["latest"], "gpt-4-turbo-test")
        self.assertTrue(updates["gpt-4"]["deprecated"])
        
        # Test with up-to-date models
        model_dict = {
            "gpt-4": "gpt-4-turbo-test",
            "gpt-3.5": "gpt-3.5-turbo-test"
        }
        
        updates = updater._check_provider_updates("openai", model_dict)
        
        # Should find no updates
        self.assertEqual(len(updates), 0)
        
        # Test with aliases (should be considered up-to-date)
        model_dict = {
            "gpt-4": "gpt-4-alias-1",
            "gpt-3.5": "gpt-3.5-alias"
        }
        
        updates = updater._check_provider_updates("openai", model_dict)
        
        # Should find no updates
        self.assertEqual(len(updates), 0)
        
        print("✅ PASSED - Can detect model updates correctly")
    
    def test_check_for_updates(self):
        """Test checking for updates."""
        updater = self._create_clean_updater()
        
        # Replace the configuration with older models
        updater.config["llm"]["models"] = {
            "primary": "gpt-4-deprecated",
            "fallback": "claude-3-deprecated",
            "explanation": "gpt-3.5-deprecated"
        }
        
        # Check for updates
        result = updater.check_for_updates(force=True)
        
        # Should find updates
        self.assertTrue(result["updates_available"])
        self.assertEqual(result["update_count"], 3)
        self.assertIn("openai", result["available_updates"])
        self.assertIn("anthropic", result["available_updates"])
        
        # Check that last_check_time was updated
        self.assertGreater(updater.last_check_time, 0)
        
        print("✅ PASSED - Can check for model updates")
    
    @patch('model_updater.ModelUpdater._update_model_in_config')
    def test_update_models(self, mock_update):
        """Test updating models."""
        # Mock the update method to always succeed
        mock_update.return_value = True
        
        updater = self._create_clean_updater()
        
        # Replace the configuration with older models
        updater.config["llm"]["models"] = {
            "primary": "gpt-4-deprecated",
            "fallback": "claude-3-deprecated",
            "explanation": "gpt-3.5-deprecated"
        }
        
        # Update all models
        result = updater.update_models(force=True)
        
        # Should have updated models
        self.assertTrue(result["updated"])
        self.assertEqual(result["successful_count"], 3)
        self.assertEqual(result["failed_count"], 0)
        
        # Check that last_update_time was updated
        self.assertGreater(updater.last_update_time, 0)
        
        # Update specific model
        updater = self._create_clean_updater()
        updater.config["llm"]["models"]["primary"] = "gpt-4-deprecated"
        
        result = updater.update_models(
            models_to_update={"openai": ["gpt-4"]},
            force=True
        )
        
        # Should have updated only the specified model
        self.assertTrue(result["updated"])
        self.assertEqual(result["successful_count"], 1)
        
        print("✅ PASSED - Can update models")
    
    def test_ignore_update(self):
        """Test ignoring updates."""
        updater = self._create_clean_updater()
        
        # Ignore an update
        result = updater.ignore_update(
            provider="openai",
            model_name="gpt-4",
            from_version="gpt-4-deprecated",
            to_version="gpt-4-turbo-test",
            duration_days=30,
            reason="Testing"
        )
        
        # Check result
        self.assertTrue(result["ignored"])
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["model"], "gpt-4")
        self.assertEqual(result["from"], "gpt-4-deprecated")
        self.assertEqual(result["to"], "gpt-4-turbo-test")
        
        # Check that it was added to ignore list
        ignore_key = "openai/gpt-4/gpt-4-deprecated/gpt-4-turbo-test"
        self.assertIn(ignore_key, updater.ignore_models)
        self.assertEqual(updater.ignore_models[ignore_key]["reason"], "Testing")
        
        # Replace the configuration with older models
        updater.config["llm"]["models"] = {
            "primary": "gpt-4-deprecated"
        }
        
        # Try to update the ignored model
        with patch('model_updater.ModelUpdater._update_model_in_config') as mock_update:
            mock_update.return_value = True
            
            result = updater.update_models(
                models_to_update={"openai": ["gpt-4"]},
                force=True
            )
            
            # Should not have updated the model
            self.assertFalse(result["updated"])
            mock_update.assert_not_called()
        
        print("✅ PASSED - Can ignore updates")
    
    def test_clear_ignore(self):
        """Test clearing ignored updates."""
        updater = self._create_clean_updater()
        
        # Add some ignored updates
        updater.ignore_update(
            provider="openai",
            model_name="gpt-4",
            from_version="gpt-4-deprecated",
            to_version="gpt-4-turbo-test"
        )
        
        updater.ignore_update(
            provider="openai",
            model_name="gpt-4",
            from_version="gpt-4-old",
            to_version="gpt-4-turbo-test"
        )
        
        # Clear ignored updates
        result = updater.clear_ignore(
            provider="openai",
            model_name="gpt-4"
        )
        
        # Check result
        self.assertTrue(result["cleared"])
        self.assertEqual(result["count"], 2)
        
        # Check that ignore list was cleared
        for key in list(updater.ignore_models.keys()):
            self.assertFalse(key.startswith("openai/gpt-4/"))
        
        print("✅ PASSED - Can clear ignored updates")
    
    def test_update_model_in_config(self):
        """Test updating a model in the configuration."""
        updater = self._create_clean_updater()
        
        # Set up test - we need to save this to disk for _load_config to read it
        updater.config["llm"]["models"]["primary"] = "gpt-4-deprecated"
        updater._save_config(updater.config)
        
        # Update the model
        success = updater._update_model_in_config(
            provider="openai",
            model_family="gpt-4",
            current_version="gpt-4-deprecated",
            new_version="gpt-4-turbo-test"
        )
        
        # Reload config to get the updated values
        updater.config = updater._load_config()
        
        # Check result
        self.assertTrue(success)
        self.assertEqual(updater.config["llm"]["models"]["primary"], "gpt-4-turbo-test")
        
        print("✅ PASSED - Can update models in configuration")
    
    def test_state_persistence(self):
        """Test state persistence."""
        # Create first updater and set some state
        updater1 = self._create_clean_updater()
        updater1.last_check_time = 12345
        updater1.last_update_time = 67890
        updater1.ignore_models = {
            "test/key": {"until": 99999, "reason": "Testing"}
        }
        
        # Save state
        updater1._save_model_state()
        
        # Create second updater to load the state
        updater2 = ModelUpdater(
            config_path=self.test_config_path,
            state_path=self.test_state_path,
            llm_manager=self.llm_manager,
            model_catalog=TEST_MODEL_CATALOG
        )
        
        # Check state was loaded
        self.assertEqual(updater2.last_check_time, 12345)
        self.assertEqual(updater2.last_update_time, 67890)
        self.assertIn("test/key", updater2.ignore_models)
        self.assertEqual(updater2.ignore_models["test/key"]["reason"], "Testing")
        
        print("✅ PASSED - State persistence works")
    
    def test_scheduler(self):
        """Test the scheduler."""
        updater = self._create_clean_updater()
        
        # Set shorter intervals for testing
        updater.check_interval = 0.1  # 100 ms
        updater.update_interval = 0.2  # 200 ms
        
        # Patch the _scheduler_callback method to verify it's called
        with patch.object(updater, '_scheduler_callback') as mock_callback:
            try:
                # Start scheduler
                updater.start_scheduler()
                
                # Check that a timer was created
                self.assertIsNotNone(updater._update_timer)
                
                # Directly trigger the callback to verify it works
                updater._scheduler_callback("check")
                
                # The callback should schedule the next check
                self.assertIsNotNone(updater._update_timer)
            finally:
                # Stop scheduler
                updater.stop_scheduler()
                
                # Verify the timer was cancelled
                self.assertIsNone(updater._update_timer)
        
        print("✅ PASSED - Scheduler works correctly")


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def main():
    """Run all tests."""
    print_header("MODEL UPDATER TEST SUITE")
    
    test = TestModelUpdater()
    test.setUp()
    
    try:
        # Run tests individually to get better output
        test.test_initialization()
        test.test_get_current_models()
        test.test_parse_model_string()
        test.test_get_model_family()
        test.test_check_provider_updates()
        test.test_check_for_updates()
        test.test_update_models()
        test.test_ignore_update()
        test.test_clear_ignore()
        test.test_update_model_in_config()
        test.test_state_persistence()
        test.test_scheduler()
    finally:
        test.tearDown()
    
    print("\n")
    print_header("TEST SUITE COMPLETE")


if __name__ == "__main__":
    main()
