#!/usr/bin/env python3
"""
Manual test script for model_updater.py
Demonstrates practical usage in real-world scenarios
"""

import os
import logging
import time
import json
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

# Create a mock config file
config_path = temp_dir / "config.json"
with open(config_path, "w") as f:
    json.dump({
        "llm": {
            "models": {
                "primary": "gpt-4-old",
                "fallback": "claude-3-old",
                "explanation": "gpt-3.5-old"
            }
        }
    }, f)

# Create a mock update feed
update_feed_path = temp_dir / "model_updates.json"
with open(update_feed_path, "w") as f:
    json.dump({
        "updates": [
            {
                "provider": "openai",
                "model_family": "gpt-4",
                "old_version": "gpt-4-old",
                "new_version": "gpt-4-turbo-preview",
                "release_date": "2025-07-10",
                "breaking_changes": False,
                "performance_improvement": 25,
                "token_price_change": -5
            },
            {
                "provider": "anthropic",
                "model_family": "claude-3",
                "old_version": "claude-3-old",
                "new_version": "claude-3-sonnet-20250701",
                "release_date": "2025-07-01",
                "breaking_changes": False,
                "performance_improvement": 15,
                "token_price_change": 0
            },
            {
                "provider": "openai",
                "model_family": "gpt-3.5",
                "old_version": "gpt-3.5-old",
                "new_version": "gpt-3.5-turbo-0701",
                "release_date": "2025-07-01",
                "breaking_changes": False,
                "performance_improvement": 10,
                "token_price_change": -10
            }
        ]
    }, f)

# Create the model updater
updater = ModelUpdater(
    config_path=config_path,
    update_feed_path=update_feed_path,
    state_file=temp_dir / "model_state.json",
    check_interval_hours=24,
    auto_update=False,
    test_mode=True  # Explicitly set test mode to avoid LLM initialization
)

# Check current models
print("\nCurrent Model Configuration:")
models = updater.get_current_models()
for model_type, model_name in models.items():
    print(f"  {model_type}: {model_name}")

# Check for updates
print("\nChecking for model updates...")
updates = updater.check_updates()
print(f"Found {len(updates)} available updates:")
for update in updates:
    print(f"  - {update['provider']}/{update['model_family']}: {update['old_version']} → {update['new_version']}")
    print(f"    Released: {update['release_date']}, Performance gain: {update['performance_improvement']}%")
    if update['token_price_change'] < 0:
        print(f"    Price reduction: {abs(update['token_price_change'])}%")
    elif update['token_price_change'] > 0:
        print(f"    Price increase: {update['token_price_change']}%")
    else:
        print(f"    No price change")

# Update a specific model
print("\nUpdating primary model (gpt-4)...")
update_result = updater.update_model("primary", updates[0])
print(f"  Update {'successful' if update_result else 'failed'}")

# Check current models after update
print("\nUpdated Model Configuration:")
models = updater.get_current_models()
for model_type, model_name in models.items():
    print(f"  {model_type}: {model_name}")

# Ignore an update
print("\nIgnoring update for fallback model...")
ignore_result = updater.ignore_update(
    provider="anthropic",
    model_family="claude-3",
    reason="Testing model update on primary first"
)
print(f"  Ignore {'successful' if ignore_result else 'failed'}")

# Check updates again (should show one less)
print("\nChecking for model updates after ignore...")
updates = updater.check_updates()
print(f"Found {len(updates)} available updates:")
for update in updates:
    print(f"  - {update['provider']}/{update['model_family']}: {update['old_version']} → {update['new_version']}")

# Clear ignored updates
print("\nClearing ignored updates...")
cleared = updater.clear_ignored_updates("anthropic", "claude-3")
print(f"  Cleared {cleared} ignored updates")

# Update all models
print("\nUpdating all models...")
updated = updater.update_all_models()
print(f"  Updated {updated} models")

# Final model configuration
print("\nFinal Model Configuration:")
models = updater.get_current_models()
for model_type, model_name in models.items():
    print(f"  {model_type}: {model_name}")

# Clean up
print("\nCleaning up...")
try:
    # Safe cleanup - only remove files if they exist
    for file_path in [config_path, update_feed_path, temp_dir / "model_state.json"]:
        if os.path.exists(file_path):
            os.remove(file_path)
            
    # Check if directory is empty before removing
    if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
        # Only remove if directory is empty
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
        else:
            print(f"  ⚠️ Directory not empty, skipping removal: {temp_dir}")
    
    print("  ✅ Temporary files removed")
except Exception as e:
    print(f"  ❌ Failed to clean up: {e}")

print("\n=== Test Complete ===")
