#!/usr/bin/env python3
"""
Fixed manual test script for log_retention.py
Demonstrates practical usage in real-world scenarios
"""

import os
import logging
import time
import shutil
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Import the log retention system
from log_retention import LogRetentionManager, LogRetentionPolicy

print("=== Manual Log Retention Test ===")

# Create a temporary directory for testing
temp_dir = Path(".temp_log_test")
temp_dir.mkdir(exist_ok=True)

log_dir = temp_dir / "logs"
archive_dir = log_dir / "archive"

# Create directories
log_dir.mkdir(exist_ok=True)
archive_dir.mkdir(exist_ok=True)

# Custom policy for testing
test_policy = {
    "max_size_mb": 1,  # Small size for testing
    "max_age_days": 30,
    "rotation_size_mb": 0.1,  # 100KB for testing
    "archive_older_than_days": 7,
    "delete_archives_older_than_days": 30,
    "severity_retention": {
        "CRITICAL": 90,
        "ERROR": 60,
        "WARNING": 30,
        "INFO": 15,
        "DEBUG": 7
    },
    "excluded_logs": ["system.log"],
    "protected_logs": ["critical.log"],
    "compression_enabled": True
}

# Function to create test log files
def create_test_log(name, size_kb, age_days=0, content_type="INFO"):
    """Create a test log file with specified properties"""
    file_path = log_dir / name
    
    # Generate appropriate content
    if content_type == "INFO":
        line_template = "[INFO] This is a sample log entry for testing purposes.\n"
    elif content_type == "ERROR":
        line_template = "[ERROR] This is a sample error entry for testing purposes.\n"
    elif content_type == "MIXED":
        templates = [
            "[INFO] This is a sample info entry.\n",
            "[WARNING] This is a sample warning entry.\n",
            "[ERROR] This is a sample error entry.\n"
        ]
        # Create the content by cycling through templates
        content = ""
        for i in range(size_kb * 20):  # ~50 bytes per line, so ~20 lines per KB
            content += templates[i % len(templates)]
        
        with open(file_path, 'w') as f:
            f.write(content)
            
        # Set file modification time if needed
        if age_days > 0:
            mod_time = time.time() - (age_days * 86400)
            os.utime(file_path, (mod_time, mod_time))
            
        return file_path
    
    # For INFO and ERROR logs, use the line template
    # Each line is ~70 bytes, so we need about 15 lines per KB
    lines_needed = size_kb * 15
    content = line_template * int(lines_needed)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    # Set file modification time if needed
    if age_days > 0:
        mod_time = time.time() - (age_days * 86400)
        os.utime(file_path, (mod_time, mod_time))
    
    return file_path

try:
    # Create some test logs
    print("\nCreating test log files...")
    test_logs = [
        create_test_log("app.log", 20),  # 20KB current log
        create_test_log("system.log", 5),  # 5KB excluded log
        create_test_log("critical.log", 15, content_type="ERROR"),  # 15KB protected log
        create_test_log("error.log", 25, content_type="ERROR"),  # 25KB error log
        create_test_log("debug.log", 10, 10, content_type="INFO"),  # 10KB, 10 days old
        create_test_log("old_app.log.20250601", 15, 30)  # 15KB rotated log, 30 days old
    ]

    # Create a log retention manager
    manager = LogRetentionManager(
        log_dir=log_dir, 
        archive_dir=archive_dir,
        policy=test_policy
    )

    # Display initial status
    print("\nInitial Log Status:")
    status = manager.get_status()
    print(f"  Log directory: {status['log_directory']}")
    print(f"  Archive directory: {status['archive_directory']}")
    print(f"  Total size: {status['total_size_mb']:.2f} MB")
    print(f"  Log files: {status['log_file_count']}")
    print(f"  Rotated files: {status['rotated_file_count']}")
    print(f"  Archived files: {status['archived_file_count']}")
    print(f"  Total files: {status['total_file_count']}")

    # Test log rotation
    print("\nRotating logs that exceed rotation size...")
    rotation_results = manager.rotate_logs()
    print(f"  Rotated {rotation_results['rotated']} logs, {rotation_results['errors']} errors")

    # Display files after rotation
    print("\nFiles after rotation:")
    for f in log_dir.glob("*"):
        file_size_kb = f.stat().st_size / 1024
        print(f"  {f.name} - {file_size_kb:.2f} KB")

    # Test log archiving
    print("\nArchiving old rotated logs...")
    archive_results = manager.archive_logs()
    print(f"  Archived {archive_results['archived']} logs, freed {archive_results['space_freed_bytes']/1024:.2f} KB")

    # Display files after archiving
    print("\nFiles after archiving:")
    for f in log_dir.glob("*"):
        file_size_kb = f.stat().st_size / 1024
        print(f"  {f.name} - {file_size_kb:.2f} KB")
    print("\nArchive directory:")
    for f in archive_dir.glob("*"):
        file_size_kb = f.stat().st_size / 1024
        print(f"  {f.name} - {file_size_kb:.2f} KB")

    # Test log deletion
    print("\nDeleting old logs...")
    deletion_results = manager.delete_logs()
    print(f"  Deleted {deletion_results['deleted']} logs, freed {deletion_results['space_freed_bytes']/1024:.2f} KB")

    # Test enforcing size limits
    print("\nEnforcing size limits...")
    size_results = manager.enforce_size_limits()
    print(f"  Deleted {size_results['deleted']} logs to enforce size limits")
    print(f"  Freed {size_results['space_freed_bytes']/1024:.2f} KB")

    # Test policy enforcement
    print("\nEnforcing retention policy...")
    policy_results = manager.enforce_policies()
    print(f"  Policy enforcement completed:")
    print(f"  - Rotated: {policy_results['rotated']}")
    print(f"  - Archived: {policy_results['archived']}")
    print(f"  - Deleted: {policy_results['deleted']}")
    
    # Check for space_freed key with different names
    if 'space_freed_mb' in policy_results:
        print(f"  - Freed space: {policy_results['space_freed_mb']:.2f} MB")
    elif 'space_freed_bytes' in policy_results:
        print(f"  - Freed space: {policy_results['space_freed_bytes']/1024/1024:.2f} MB")
    elif 'space_freed' in policy_results:
        print(f"  - Freed space: {policy_results['space_freed']:.2f} MB")
    
    # Check for duration with different names
    if 'duration_seconds' in policy_results:
        print(f"  - Duration: {policy_results['duration_seconds']:.2f} seconds")
    elif 'duration' in policy_results:
        print(f"  - Duration: {policy_results['duration']:.2f} seconds")
    elif 'took' in policy_results:
        print(f"  - Duration: {policy_results['took']:.2f} seconds")

    # Display final status
    print("\nFinal Log Status:")
    status = manager.get_status()
    print(f"  Total size: {status['total_size_mb']:.2f} MB")
    print(f"  Log files: {status['log_file_count']}")
    print(f"  Rotated files: {status['rotated_file_count']}")
    print(f"  Archived files: {status['archived_file_count']}")
    print(f"  Total files: {status['total_file_count']}")

except Exception as e:
    print(f"\n❌ Error: {e}")

finally:
    # Clean up
    print("\nCleaning up...")
    try:
        shutil.rmtree(temp_dir)
        print("  ✅ Temporary files removed")
    except Exception as e:
        print(f"  ❌ Failed to clean up: {e}")

print("\n=== Test Complete ===")
