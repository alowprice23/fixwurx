#!/usr/bin/env python3
"""
Updated manual test script for key_rotation.py
Demonstrates practical usage in real-world scenarios
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Import the key rotation manager
from key_rotation import KeyRotationManager

print("=== Manual Key Rotation Test ===")

# Create a temporary directory for testing
temp_dir = Path(".temp_key_test")
temp_dir.mkdir(exist_ok=True)

# Create a mock credential manager
class MockCredentialManager:
    """Simple mock credential manager for testing."""
    
    def __init__(self):
        self.keys = {
            "openai": "sk-test12345",
            "anthropic": "sk-ant12345"
        }
        self.key_counter = 0
    
    def get_api_key(self, provider):
        """Get API key for provider."""
        return self.keys.get(provider)
    
    def set_api_key(self, provider, key):
        """Set API key for provider."""
        self.keys[provider] = key
        return True
    
    def generate_key(self, provider):
        """Generate a new key for provider."""
        self.key_counter += 1
        if provider == "openai":
            return f"sk-t{int(time.time())%10000:04d}"
        elif provider == "anthropic":
            return f"sk-a{int(time.time())%10000:04d}"
        return f"sk-{provider}-{self.key_counter}"
    
    def mask_key(self, key):
        """Create a masked version of the key."""
        if not key or len(key) < 10:
            return "[invalid key]"
        return f"{key[:5]}...{key[-4:]}"

# Create credential manager
cred_manager = MockCredentialManager()

# Create config
config = {
    "rotation_interval_mins": 0.1,  # 6 seconds for testing
    "rotation_offset_mins": 0.05,   # 3 seconds jitter
    "notification_lead_mins": 0.05, # 3 seconds before rotation
    "key_providers": ["openai", "anthropic"],
    "state_path": str(temp_dir / "rotation_state.json")
}

# Create a key rotation manager
manager = KeyRotationManager(
    credential_manager=cred_manager,
    config=config
)

# Register providers for testing
manager.register_provider(
    "openai", 
    cred_manager.get_api_key("openai"),
    lambda: cred_manager.generate_key("openai")
)
manager.register_provider(
    "anthropic", 
    cred_manager.get_api_key("anthropic"),
    lambda: cred_manager.generate_key("anthropic")
)

# Display initial status
print("\nInitial Key Status:")
status = manager.get_rotation_status()

# Display status details
print(f"Status structure: {list(status.keys())}")
for provider_name, provider_data in status.get("providers", {}).items():
    print(f"  {provider_name}: {provider_data}")

# Manual rotation
print("\nPerforming manual rotation for OpenAI key...")
old_key = cred_manager.get_api_key("openai")
success = manager._test_rotate_key("openai", force=True)
new_key = cred_manager.get_api_key("openai")
print(f"  OpenAI key rotated: {cred_manager.mask_key(old_key)} → {cred_manager.mask_key(new_key)} (Success: {success})")

# Add a callback
print("\nTesting rotation callback:")
def rotation_callback(provider, success):
    if success:
        print(f"  ✅ Callback triggered: {provider} key rotated successfully")
    else:
        print(f"  ❌ Callback triggered: {provider} key rotation failed")

manager.register_post_rotation_callback(rotation_callback)
manager._test_rotate_key("anthropic", force=True)

# Try to get rotation status
print("\nChecking rotation status:")
status = manager.get_rotation_status()
print(f"  Providers configured: {', '.join(status.get('providers', {}).keys())}")
print(f"  Rotation interval: {status.get('config', {}).get('rotation_interval_mins', 'N/A')} minutes")

if status.get('rotation_history'):
    latest = status['rotation_history'][-1]
    print(f"  Last rotation: {latest.get('provider')} at {latest.get('timestamp')}")

# Rotate all keys
print("\nRotating all keys:")
results = {}
for provider in ["openai", "anthropic"]:
    try:
        results[provider] = manager._test_rotate_key(provider, force=True)
    except Exception as e:
        print(f"Error rotating {provider} key: {e}")
        results[provider] = False
for provider, success in results.items():
    print(f"  {provider}: {'✅ Success' if success else '❌ Failed'}")

# Final status
print("\nFinal Key Status:")
status = manager.get_rotation_status()
for provider_name, provider_data in status.get("providers", {}).items():
    key = cred_manager.get_api_key(provider_name)
    masked_key = cred_manager.mask_key(key) if key else "N/A"
    next_rotation = provider_data.get("next_rotation", "N/A")
    print(f"  {provider_name}: {masked_key} (next rotation: {next_rotation})")

# Clean up
try:
    state_path = Path(config["state_path"])
    if state_path.exists():
        os.remove(state_path)
    os.rmdir(temp_dir)
    print("\nTemporary files cleaned up")
except Exception as e:
    print(f"\nFailed to clean up: {e}")

print("\n=== Test Complete ===")
