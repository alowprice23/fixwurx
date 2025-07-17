#!/usr/bin/env python3
"""
Manual test script for config_manager.py
Demonstrates practical usage in real-world scenarios
"""

import os
import logging
import time
import yaml
import json
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Import the configuration manager
from config_manager import ConfigManager

print("=== Manual Configuration Manager Test ===")

# Create a temporary directory for testing
temp_dir = Path(".temp_config_test")
temp_dir.mkdir(exist_ok=True)

# Create a test config file
config_path = temp_dir / "system_config.yaml"

# Create a ConfigManager instance
manager = ConfigManager(config_path=config_path)

# Display initial configuration
print("\nInitial Configuration:")
print(f"Environment: {manager.environment}")
print(f"Schema Version: {manager.schema_version}")

# Display specific sections
print("\nLLM Configuration:")
llm_config = manager.get_section("llm")
for key, value in llm_config.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for subkey, subvalue in value.items():
            print(f"    {subkey}: {subvalue}")
    else:
        print(f"  {key}: {value}")

# Modify configuration
print("\nModifying configuration...")
manager.set("llm.temperature", 0.8)
manager.set("llm.cost-budget-usd", 2.5)
manager.set("logging.level", "DEBUG")

# Add a new section
storage_config = manager.get_section("storage")
storage_config["backup_location"] = "/custom/backup/path"
storage_config["auto_backup"] = True
manager.set_section("storage", storage_config)

# Display updated configuration
print("\nUpdated Configuration:")
print(f"LLM Temperature: {manager.get('llm.temperature')}")
print(f"LLM Cost Budget: ${manager.get('llm.cost-budget-usd')}")
print(f"Logging Level: {manager.get('logging.level')}")
print(f"Storage Backup Location: {manager.get('storage.backup_location')}")
print(f"Storage Auto Backup: {manager.get('storage.auto_backup')}")

# Create a backup
print("\nCreating configuration backup...")
backup_path = manager.backup("test_backup")
print(f"Backup created at: {backup_path}")

# Make additional changes
print("\nMaking additional changes...")
manager.set("llm.temperature", 0.2)
manager.delete("storage.auto_backup")
print(f"New Temperature: {manager.get('llm.temperature')}")
print(f"Auto Backup Setting: {manager.get('storage.auto_backup', 'Deleted')}")

# List backups
print("\nListing available backups:")
backups = manager.list_backups()
for backup in backups:
    print(f"  {backup['name']} - {backup['created']} - {backup['environment']}")

# Restore from backup
print("\nRestoring from backup...")
manager.restore(backup_path)
print(f"Restored Temperature: {manager.get('llm.temperature')}")
print(f"Restored Auto Backup Setting: {manager.get('storage.auto_backup')}")

# Test environment switching
print("\nSwitching to staging environment...")
manager.switch_environment("staging")
print(f"Current Environment: {manager.environment}")

# Configure staging environment differently
print("Configuring staging environment...")
manager.set("llm.temperature", 0.5)
manager.set("security.audit_logging_enabled", True)
print(f"Staging Temperature: {manager.get('llm.temperature')}")
print(f"Staging Audit Logging: {manager.get('security.audit_logging_enabled')}")

# Switch to production
print("\nSwitching to production environment...")
manager.switch_environment("production")
print(f"Current Environment: {manager.environment}")

# Configure production environment
print("Configuring production environment...")
manager.set("llm.temperature", 0.3)
manager.set("logging.level", "ERROR")
manager.set("security.access_control_enabled", True)
print(f"Production Temperature: {manager.get('llm.temperature')}")
print(f"Production Logging Level: {manager.get('logging.level')}")
print(f"Production Access Control: {manager.get('security.access_control_enabled')}")

# Compare environments
print("\nComparing production with staging environment:")
comparison = manager.compare_environments("staging")
print(f"Total Differences: {comparison['total_differences']}")
print(f"Added: {comparison['added_count']}")
print(f"Removed: {comparison['removed_count']}")
print(f"Modified: {comparison['modified_count']}")

if comparison['modified']:
    print("\nModified Keys:")
    for key in comparison['modified'][:5]:  # Show at most 5 differences
        print(f"  {key}")

# Return to development environment
print("\nSwitching back to development environment...")
manager.switch_environment("development")
print(f"Current Environment: {manager.environment}")
print(f"Development Temperature: {manager.get('llm.temperature')}")

# View change history
print("\nConfiguration Change History:")
history = manager.list_history()
for i, entry in enumerate(history[-5:]):  # Show the last 5 entries
    print(f"  {entry['datetime']} - {entry['environment']} - v{entry['schema_version']}")

# Clean up
print("\nCleaning up...")
try:
    # Remove temporary files
    shutil.rmtree(temp_dir)
    # Also remove .triangulum directory created by ConfigManager
    triangulum_dir = Path(".triangulum")
    if triangulum_dir.exists():
        shutil.rmtree(triangulum_dir)
    print("  ✅ Temporary files removed")
except Exception as e:
    print(f"  ❌ Failed to clean up: {e}")

print("\n=== Test Complete ===")
