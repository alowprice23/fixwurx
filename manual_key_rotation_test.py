#!/usr/bin/env python3
"""
Manual test script for key_rotation.py
Demonstrates practical usage in real-world scenarios
"""

import os
import logging
import time
from pathlib import Path

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

# Create a key rotation manager
manager = KeyRotationManager(
    state_file=temp_dir / "key_state.json", 
    rotation_interval_days=0.0001  # Very short interval for testing (about 8.6 seconds)
)

# Register API keys
manager.register_provider(
    "openai", 
    current_key="sk-test12345", 
    key_generator=lambda: f"sk-t{int(time.time())%10000:04d}",
    next_rotation=time.time() + 5  # Schedule rotation in 5 seconds
)

manager.register_provider(
    "anthropic", 
    current_key="sk-ant12345", 
    key_generator=lambda: f"sk-a{int(time.time())%10000:04d}",
    next_rotation=time.time() + 10  # Schedule rotation in 10 seconds
)

# Display initial status
print("\nInitial Key Status:")
status = manager.get_status()
for provider, details in status.items():
    next_rotation_time = time.strftime('%H:%M:%S', time.localtime(details['next_rotation']))
    print(f"  {provider}: {details['key'][:5]}... (next rotation at {next_rotation_time})")

# Manual rotation
print("\nPerforming manual rotation for OpenAI key...")
old_key = manager.get_key("openai")
manager.rotate_key("openai")
new_key = manager.get_key("openai")
print(f"  OpenAI key rotated: {old_key[:5]}... → {new_key[:5]}...")

# Wait for scheduled rotation
print("\nWaiting for scheduled rotation (5 seconds)...")
time.sleep(6)

# Check if key was rotated
latest_key = manager.get_key("openai")
if latest_key != new_key:
    print(f"  ✅ Scheduled rotation successful: {new_key[:5]}... → {latest_key[:5]}...")
else:
    print("  ❌ Scheduled rotation didn't occur")

# Try to get an unknown provider
print("\nTrying to get key for unknown provider:")
try:
    manager.get_key("unknown")
    print("  ❌ Expected error for unknown provider")
except KeyError:
    print("  ✅ Properly raised KeyError for unknown provider")

# Add a callback
print("\nTesting rotation callback:")
def rotation_callback(provider, old_key, new_key):
    print(f"  ✅ Callback triggered: {provider} key rotated from {old_key[:5]}... to {new_key[:5]}...")

manager.add_rotation_callback(rotation_callback)
manager.rotate_key("anthropic")

# Final status
print("\nFinal Key Status:")
status = manager.get_status()
for provider, details in status.items():
    next_rotation_time = time.strftime('%H:%M:%S', time.localtime(details['next_rotation']))
    print(f"  {provider}: {details['key'][:5]}... (next rotation at {next_rotation_time})")

# Clean up
print("\nCleaning up...")
try:
    os.remove(temp_dir / "key_state.json")
    os.rmdir(temp_dir)
    print("  ✅ Temporary files removed")
except Exception as e:
    print(f"  ❌ Failed to clean up: {e}")

print("\n=== Test Complete ===")
