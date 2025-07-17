#!/usr/bin/env python3
"""
Fixed manual test script for model_updater.py
Demonstrates practical usage in real-world scenarios
"""

import os
import logging
import time
import json
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Import the model updater
from model_updater import ModelUpdater

print("=== Manual Model Updater Test ===")

# Create a temporary directory for testing
temp_dir = Path(".temp_model_test")
temp_dir.mkdir(exist_ok=True)

# Create a mock config file with YAML format
config_path = temp_dir / "config.yaml"
config_data = {
    "schema_version": "1.1",
    "llm": {
        "models": {
            "primary": "gpt-4-old",
            "fallback": "claude-3-old",
            "explanation": "gpt-3.5-old"
        }
    }
}

with open(config_path, "w") as f:
    yaml.dump(config_data, f)

# Create a mock model catalog
TEST_MODEL_CATALOG = {
    "openai": {
        "gpt-4": {
            "latest": "gpt-4-turbo-preview",
            "aliases": ["gpt-4-turbo", "gpt-4"],
            "deprecated": ["gpt-4-old"],
            "fallbacks": ["gpt-4-0613", "gpt-3.5-turbo"]
        },
        "gpt-3.5": {
            "latest": "gpt-3.5-turbo-0701",
            "aliases": ["gpt-3.5-turbo"],
            "deprecated": ["gpt-3.5-old"],
            "fallbacks": ["gpt-3.5-turbo-0613"]
        }
    },
    "anthropic": {
        "claude-3": {
            "latest": "claude-3-sonnet-20250701",
            "aliases": ["claude-3-sonnet"],
            "deprecated": ["claude-3-old"],
            "fallbacks": ["claude-3-haiku"]
        }
    }
}

# Create a mock LLM manager
class MockLLMManager:
    def __init__(self):
        self.available_providers = ["openai", "anthropic"]
    
    def get_available_providers(self):
        return self.available_providers
    
    def get_model_provider(self, model_name):
        if model_name.startswith("gpt"):
            return "openai"
        elif model_name.startswith("claude"):
            return "anthropic"
        return None
    
    def available(self):
        return self.available_providers

try:
    # Create the model updater
    print("\nInitializing Model Updater...")
    updater = ModelUpdater(
        config_path=config_path,
        state_path=temp_dir / "model_state.json",
        llm_manager=MockLLMManager(),
        model_catalog=TEST_MODEL_CATALOG
    )

    # Check current models from config
    print("\nCurrent Model Configuration:")
    try:
        if hasattr(updater, 'config') and 'llm' in updater.config and 'models' in updater.config['llm']:
            models = updater.config['llm']['models']
            for model_type, model_name in models.items():
                print(f"  {model_type}: {model_name}")
        else:
            print("  No models found in configuration")
    except Exception as e:
        print(f"  Error accessing models: {e}")

    # Check for updates
    print("\nChecking for model updates...")
    try:
        update_info = updater.check_for_updates()
        print(f"Updates available: {update_info['updates_available']}")
        print(f"Update count: {update_info['update_count']}")
        
        if update_info['update_count'] > 0:
            print("\nAvailable updates:")
            for provider, families in update_info['available_updates'].items():
                for family, details in families.items():
                    print(f"  {provider}/{family}: {details['current']} → {details['latest']}")
                    if details.get('deprecated', False):
                        print(f"    ⚠️ Current model is deprecated")
        
        # Extract list of models that need updating for next step
        models_to_update = {}
        for provider, families in update_info['available_updates'].items():
            if families:
                models_to_update[provider] = list(families.keys())
    except Exception as e:
        print(f"  Error checking for updates: {e}")
        models_to_update = {}

    # Update models
    if models_to_update:
        print("\nUpdating models...")
        try:
            # Use update_models method with the correct structure
            result = updater.update_models(models_to_update)
            print(f"Update result: {result}")
        except Exception as e:
            print(f"  Error updating models: {e}")

        # Check configuration after update
        print("\nUpdated Model Configuration:")
        try:
            if hasattr(updater, 'config') and 'llm' in updater.config and 'models' in updater.config['llm']:
                models = updater.config['llm']['models']
                for model_type, model_name in models.items():
                    print(f"  {model_type}: {model_name}")
            else:
                print("  No models found in configuration")
        except Exception as e:
            print(f"  Error accessing updated models: {e}")

    # Ignore updates
    print("\nIgnoring update for claude-3...")
    try:
        # Try different method name variations
        if hasattr(updater, 'ignore_update'):
            updater.ignore_update("anthropic", "claude-3", "Testing model update process")
            print("  Successfully ignored claude-3 updates")
        else:
            print("  No ignore_update method found")
    except Exception as e:
        print(f"  Error ignoring update: {e}")

    # Check ignored models
    print("\nChecking ignored models:")
    try:
        if hasattr(updater, 'ignore_models'):
            print(f"  Ignored models: {updater.ignore_models}")
        else:
            print("  No ignore_models attribute found")
    except Exception as e:
        print(f"  Error checking ignored models: {e}")

except Exception as e:
    print(f"\n❌ Error: {e}")

finally:
    # Clean up
    print("\nCleaning up...")
    try:
        if config_path.exists():
            os.remove(config_path)
        state_path = temp_dir / "model_state.json"
        if state_path.exists():
            os.remove(state_path)
        os.rmdir(temp_dir)
        print("  ✅ Temporary files removed")
    except Exception as e:
        print(f"  ❌ Failed to clean up: {e}")

print("\n=== Test Complete ===")
