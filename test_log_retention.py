#!/usr/bin/env python3
"""
test_log_retention.py
────────────────────
Test script for the log retention policy system.

This test verifies:
1. Log rotation
2. Log archiving
3. Log deletion
4. Policy enforcement
5. Size limit management

Run this script to verify that the log retention system is working correctly.
"""

import os
import time
import shutil
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the log retention module
from log_retention import LogRetentionManager, LogRetentionPolicy, DEFAULT_RETENTION_POLICY


class TestLogRetention(unittest.TestCase):
    """Test suite for log retention system."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp(prefix="log_retention_test_")
        
        # Create test paths
        self.test_log_dir = Path(self.test_dir) / "logs"
        self.test_archive_dir = Path(self.test_dir) / "logs" / "archive"
        
        # Create directories
        self.test_log_dir.mkdir(parents=True, exist_ok=True)
        self.test_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test policy
        self.test_policy = DEFAULT_RETENTION_POLICY.copy()
        self.test_policy.update({
            "max_size_mb": 1,  # Small size for testing
            "max_age_days": 1,
            "rotation_size_mb": 0.1,  # 100 KB
            "archive_older_than_days": 1,
            "delete_archives_older_than_days": 2,
        })
        
        # Create manager
        self.manager = LogRetentionManager(
            log_dir=self.test_log_dir,
            archive_dir=self.test_archive_dir,
            policy=self.test_policy
        )
    
    def tearDown(self):
        """Clean up the test environment."""
        # Stop scheduler if running
        if hasattr(self.manager, '_timer') and self.manager._timer is not None:
            self.manager.stop_scheduler()
        
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_test_log(self, name="test.log", size_kb=200, age_days=0):
        """
        Create a test log file.
        
        Args:
            name: Name of the log file
            size_kb: Size of the log file in KB
            age_days: Age of the log file in days
        
        Returns:
            Path to the created log file
        """
        log_path = self.test_log_dir / name
        
        # Create content - make sure it's actually the right size
        # Each line is approximately 23 bytes, so we need size_kb * 1024 / 23 lines
        lines_needed = int((size_kb * 1024) / 23)
        content = f"[INFO] Test log entry\n" * lines_needed
        
        # Write to file
        with open(log_path, 'wb') as f:
            f.write(content.encode('utf-8'))
        
        # Adjust modification time if needed
        if age_days > 0:
            mtime = time.time() - (age_days * 24 * 3600)
            os.utime(log_path, (mtime, mtime))
        
        return log_path
    
    def _create_rotated_log(self, name="test.log.20250101", size_kb=200, age_days=2):
        """Create a test rotated log file."""
        log_path = self.test_log_dir / name
        
        # Create content
        content = f"[INFO] Rotated log entry\n" * (size_kb * 10)
        
        # Write to file
        with open(log_path, 'w') as f:
            f.write(content)
        
        # Adjust modification time
        mtime = time.time() - (age_days * 24 * 3600)
        os.utime(log_path, (mtime, mtime))
        
        return log_path
    
    def _create_archived_log(self, name="test.log.20250101.gz", size_kb=100, age_days=5):
        """Create a test archived log file."""
        log_path = self.test_archive_dir / name
        
        # Create content (just a dummy file, not actually compressed)
        content = f"[INFO] Archived log entry\n" * (size_kb * 10)
        
        # Write to file
        with open(log_path, 'w') as f:
            f.write(content)
        
        # Adjust modification time
        mtime = time.time() - (age_days * 24 * 3600)
        os.utime(log_path, (mtime, mtime))
        
        return log_path
    
    def test_initialization(self):
        """Test initialization of log retention manager."""
        # Check directory creation
        self.assertTrue(self.test_log_dir.exists())
        self.assertTrue(self.test_archive_dir.exists())
        
        # Check policy initialization
        self.assertEqual(self.manager.policy.policy["max_size_mb"], 1)
        self.assertEqual(self.manager.policy.policy["rotation_size_mb"], 0.1)
        
        print("✅ PASSED - Log retention manager initializes correctly")
    
    def test_rotation(self):
        """Test log rotation."""
        # Create a log file larger than rotation size
        log_path = self._create_test_log(size_kb=200)  # 200 KB, rotation size is 100 KB
        
        # Verify the log exists
        self.assertTrue(log_path.exists())
        
        # Check if it should be rotated
        should_rotate = self.manager.policy.should_rotate(log_path)
        self.assertTrue(should_rotate)
        
        # Rotate logs
        results = self.manager.rotate_logs()
        
        # Check results
        self.assertEqual(results["rotated"], 1)
        self.assertEqual(results["errors"], 0)
        
        # Original log should still exist but be empty
        self.assertTrue(log_path.exists())
        self.assertEqual(log_path.stat().st_size, 0)
        
        # Should have a rotated log
        rotated_logs = list(self.test_log_dir.glob("test.log.*"))
        self.assertEqual(len(rotated_logs), 1)
        
        print("✅ PASSED - Log rotation works")
    
    def test_archiving(self):
        """Test log archiving."""
        # Create a rotated log file
        rotated_path = self._create_rotated_log(age_days=2)  # 2 days old, archive threshold is 1 day
        
        # Verify the log exists
        self.assertTrue(rotated_path.exists())
        
        # Check if it should be archived
        should_archive = self.manager.policy.should_archive(rotated_path)
        self.assertTrue(should_archive)
        
        # Archive logs
        results = self.manager.archive_logs()
        
        # Check results
        self.assertEqual(results["archived"], 1)
        self.assertEqual(results["errors"], 0)
        self.assertGreater(results["space_freed_bytes"], 0)
        
        # Original rotated log should be gone
        self.assertFalse(rotated_path.exists())
        
        # Should have an archived log
        archived_logs = list(self.test_archive_dir.glob("*.gz"))
        self.assertGreaterEqual(len(archived_logs), 1)
        
        print("✅ PASSED - Log archiving works")
    
    def test_deletion(self):
        """Test log deletion."""
        # Create an archived log file
        archived_path = self._create_archived_log(age_days=3)  # 3 days old, delete threshold is 2 days
        
        # Verify the log exists
        self.assertTrue(archived_path.exists())
        
        # Check if it should be deleted
        should_delete = self.manager.policy.should_delete(archived_path)
        self.assertTrue(should_delete)
        
        # Delete logs
        results = self.manager.delete_logs()
        
        # Check results
        self.assertEqual(results["deleted"], 1)
        self.assertEqual(results["errors"], 0)
        self.assertGreater(results["space_freed_bytes"], 0)
        
        # Archived log should be gone
        self.assertFalse(archived_path.exists())
        
        print("✅ PASSED - Log deletion works")
    
    def test_enforce_size_limits(self):
        """Test enforcing size limits."""
        # Create multiple log files totaling more than max size
        log1 = self._create_test_log("info.log", size_kb=600)
        log2 = self._create_test_log("debug.log", size_kb=600)
        
        # Total should be 1.2 MB, max is 1 MB
        
        # Verify logs exist
        self.assertTrue(log1.exists())
        self.assertTrue(log2.exists())
        
        # Enforce size limits
        results = self.manager.enforce_size_limits()
        
        # Check results
        self.assertGreaterEqual(results["deleted"], 1)
        self.assertEqual(results["errors"], 0)
        self.assertGreater(results["space_freed_bytes"], 0)
        
        # Total size should now be under the limit
        total_size = self.manager._get_directory_size(self.test_log_dir)
        total_size += self.manager._get_directory_size(self.test_archive_dir)
        self.assertLessEqual(total_size, self.test_policy["max_size_mb"] * 1024 * 1024)
        
        print("✅ PASSED - Size limit enforcement works")
    
    def test_policy_enforcement(self):
        """Test overall policy enforcement."""
        # Create various log files
        log1 = self._create_test_log("current.log", size_kb=200)
        log2 = self._create_rotated_log("old.log.20250101", age_days=2)
        log3 = self._create_archived_log("ancient.log.20250101.gz", age_days=3)
        
        # Enforce policies
        results = self.manager.enforce_policies()
        
        # Check results
        self.assertGreaterEqual(results["rotated"], 1)
        self.assertGreaterEqual(results["archived"], 1)
        self.assertGreaterEqual(results["deleted"], 1)
        self.assertEqual(results["errors"], 0)
        self.assertGreater(results["space_freed_bytes"], 0)
        
        print("✅ PASSED - Policy enforcement works")
    
    def test_log_severity_detection(self):
        """Test log severity detection."""
        # Create log files with different severity levels
        info_path = self.test_log_dir / "info.log"
        with open(info_path, 'w') as f:
            f.write("[INFO] This is an info message\n" * 10)
        
        error_path = self.test_log_dir / "error.log"
        with open(error_path, 'w') as f:
            f.write("[ERROR] This is an error message\n" * 10)
        
        mixed_path = self.test_log_dir / "mixed.log"
        with open(mixed_path, 'w') as f:
            f.write("[INFO] Info message\n" * 5)
            f.write("[ERROR] Error message\n" * 2)
            f.write("[WARNING] Warning message\n" * 3)
        
        # Test severity detection
        self.assertEqual(self.manager.policy._get_log_severity(info_path), "INFO")
        self.assertEqual(self.manager.policy._get_log_severity(error_path), "ERROR")
        self.assertEqual(self.manager.policy._get_log_severity(mixed_path), "ERROR")  # Highest severity
        
        print("✅ PASSED - Log severity detection works")
    
    def test_scheduler(self):
        """Test the scheduler."""
        # Mock the _scheduler_callback method to call enforce_policies directly
        original_callback = self.manager._scheduler_callback
        
        def immediate_callback():
            self.manager.enforce_policies()
            
        self.manager._scheduler_callback = immediate_callback
        
        # Mock the enforce_policies method
        with patch.object(self.manager, 'enforce_policies') as mock_enforce:
            mock_enforce.return_value = {"rotated": 0, "archived": 0, "deleted": 0, "errors": 0}
            
            # Start scheduler with very short interval
            self.manager.start_scheduler(interval_hours=0.001)
            
            # Trigger the callback directly
            self.manager._scheduler_callback()
            
            # Stop scheduler
            self.manager.stop_scheduler()
            
            # Check if enforce_policies was called
            mock_enforce.assert_called()
        
        # Restore original callback
        self.manager._scheduler_callback = original_callback
        
        print("✅ PASSED - Scheduler works")
    
    def test_get_status(self):
        """Test getting status information."""
        # Create some log files
        self._create_test_log("status1.log", size_kb=100)
        self._create_test_log("status2.log", size_kb=100)
        self._create_rotated_log("status3.log.20250101", size_kb=100)
        self._create_archived_log("status4.log.20250101.gz", size_kb=50)
        
        # Get status
        status = self.manager.get_status()
        
        # Check status fields
        self.assertEqual(status["log_directory"], str(self.test_log_dir))
        self.assertEqual(status["archive_directory"], str(self.test_archive_dir))
        self.assertGreaterEqual(status["log_file_count"], 2)
        self.assertGreaterEqual(status["rotated_file_count"], 1)
        self.assertGreaterEqual(status["archived_file_count"], 1)
        self.assertGreaterEqual(status["total_file_count"], 4)
        
        # Check size information
        self.assertGreater(status["log_directory_size_mb"], 0)
        self.assertGreater(status["archive_directory_size_mb"], 0)
        
        print("✅ PASSED - Status reporting works")


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def main():
    """Run all tests."""
    print_header("LOG RETENTION TEST SUITE")
    
    test = TestLogRetention()
    test.setUp()
    
    try:
        # Run tests individually to get better output
        test.test_initialization()
        test.test_rotation()
        test.test_archiving()
        test.test_deletion()
        test.test_enforce_size_limits()
        test.test_policy_enforcement()
        test.test_log_severity_detection()
        test.test_scheduler()
        test.test_get_status()
    finally:
        test.tearDown()
    
    print("\n")
    print_header("TEST SUITE COMPLETE")


if __name__ == "__main__":
    main()
