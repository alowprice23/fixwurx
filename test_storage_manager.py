#!/usr/bin/env python3
"""
Test Script for Storage Manager Module

This script tests the functionality of the storage manager module by verifying
compressed storage, rotating buffers, version control, neural pattern storage,
and cross-session knowledge persistence.
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Ensure the storage_manager module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from storage_manager import (
        CompressedStorage, RotatingBuffer, VersionControl,
        NeuralPatternStorage, PersistenceManager, StorageManager,
        create_storage_manager
    )
except ImportError:
    print("Error: Could not import storage_manager module")
    sys.exit(1)

def setup_test_dir():
    """Create a temporary test directory."""
    test_dir = tempfile.mkdtemp(prefix="storage_test_")
    print(f"Created test directory: {test_dir}")
    return test_dir

def cleanup_test_dir(test_dir):
    """Clean up the temporary test directory."""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"Removed test directory: {test_dir}")

def test_compressed_storage():
    """Test the CompressedStorage class."""
    print("\n=== Testing CompressedStorage ===")
    test_dir = setup_test_dir()
    
    try:
        # Initialize compressed storage
        storage = CompressedStorage(os.path.join(test_dir, "compressed"))
        print("Initialized compressed storage")
        
        # Store string data
        string_data = "Hello, this is a test string"
        string_id = storage.store(string_data, "text", {"description": "Test string"})
        print(f"Stored string data with ID: {string_id}")
        
        # Store dictionary data
        dict_data = {
            "name": "Test Object",
            "values": [1, 2, 3, 4, 5],
            "metadata": {
                "version": "1.0",
                "author": "Test Script"
            }
        }
        dict_id = storage.store(dict_data, "config", {"description": "Test config"})
        print(f"Stored dictionary data with ID: {dict_id}")
        
        # Retrieve string data
        retrieved_string, string_metadata = storage.retrieve(string_id)
        print(f"Retrieved string data: {retrieved_string[:20]}...")
        print(f"String metadata: {list(string_metadata.keys())}")
        
        # Verify string data
        if isinstance(retrieved_string, bytes):
            # Convert bytes to string for comparison
            retrieved_string_text = retrieved_string.decode('utf-8')
            assert retrieved_string_text == string_data, "String data does not match"
        else:
            assert retrieved_string == string_data, "String data does not match"
        
        # Retrieve dictionary data
        retrieved_dict, dict_metadata = storage.retrieve(dict_id)
        print(f"Retrieved dictionary data: {list(retrieved_dict.keys())}")
        print(f"Dictionary metadata: {list(dict_metadata.keys())}")
        
        # Verify dictionary data
        assert retrieved_dict["name"] == dict_data["name"], "Dictionary data does not match"
        assert retrieved_dict["values"] == dict_data["values"], "Dictionary data does not match"
        
        # List items
        items = storage.list_items()
        print(f"Listed {len(items)} items")
        
        # Filter by type
        text_items = storage.list_items("text")
        print(f"Listed {len(text_items)} text items")
        
        # Delete an item
        deleted = storage.delete(string_id)
        print(f"Deleted string item: {deleted}")
        
        # Verify deletion
        items_after_delete = storage.list_items()
        print(f"Items after deletion: {len(items_after_delete)}")
        assert len(items_after_delete) == len(items) - 1, "Item not deleted"
        
        return True
    except Exception as e:
        print(f"Error in compressed storage test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_rotating_buffer():
    """Test the RotatingBuffer class."""
    print("\n=== Testing RotatingBuffer ===")
    test_dir = setup_test_dir()
    
    try:
        # Initialize rotating buffer
        buffer_file = os.path.join(test_dir, "buffer.json.gz")
        buffer = RotatingBuffer(buffer_file, max_entries=5, compression=True)
        print("Initialized rotating buffer")
        
        # Add entries
        for i in range(10):
            buffer.add_entry({
                "id": i,
                "message": f"Test message {i}",
                "level": "INFO" if i % 2 == 0 else "ERROR"
            })
        
        print("Added 10 entries to buffer with max_entries=5")
        
        # Get entries
        entries = buffer.get_entries()
        print(f"Retrieved {len(entries)} entries")
        
        # Verify buffer size limit
        assert len(entries) == 5, f"Buffer should contain 5 entries, but contains {len(entries)}"
        
        # Verify newest entries are kept
        entry_ids = [e["id"] for e in entries]
        print(f"Entry IDs in buffer: {entry_ids}")
        assert min(entry_ids) == 5, "Buffer should contain newest entries"
        
        # Filter entries
        info_entries = buffer.get_entries(filter_func=lambda e: e["level"] == "INFO")
        print(f"Retrieved {len(info_entries)} INFO entries")
        
        # Get limited entries
        limited_entries = buffer.get_entries(count=2)
        print(f"Retrieved {len(limited_entries)} limited entries")
        assert len(limited_entries) == 2, "Should retrieve exactly 2 entries"
        
        # Clear buffer
        buffer.clear()
        print("Cleared buffer")
        
        # Verify clear
        empty_entries = buffer.get_entries()
        print(f"Entries after clear: {len(empty_entries)}")
        assert len(empty_entries) == 0, "Buffer should be empty after clear"
        
        return True
    except Exception as e:
        print(f"Error in rotating buffer test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_version_control():
    """Test the VersionControl class."""
    print("\n=== Testing VersionControl ===")
    test_dir = setup_test_dir()
    
    try:
        # Initialize version control
        version_control = VersionControl(os.path.join(test_dir, "versions"))
        print("Initialized version control")
        
        # Create a file and commit multiple versions
        file_path = "test_file.txt"
        version_ids = []
        
        # Version 1
        content1 = "This is version 1 of the test file."
        version_id1 = version_control.commit(file_path, content1, "Initial version")
        version_ids.append(version_id1)
        print(f"Committed version 1 with ID: {version_id1}")
        
        # Version 2
        content2 = "This is version 2 of the test file with more content."
        version_id2 = version_control.commit(file_path, content2, "Added more content")
        version_ids.append(version_id2)
        print(f"Committed version 2 with ID: {version_id2}")
        
        # Version 3
        content3 = "This is version 3 with completely different content."
        version_id3 = version_control.commit(file_path, content3, "Changed content", ["important"])
        version_ids.append(version_id3)
        print(f"Committed version 3 with ID: {version_id3}")
        
        # List versions
        versions = version_control.list_versions(file_path)
        print(f"Listed {len(versions)} versions")
        
        # Get specific version by ID
        retrieved_content2, version_info2 = version_control.get_version(file_path, version_id2)
        print(f"Retrieved version 2: {retrieved_content2[:20]}...")
        print(f"Version 2 info: {version_info2['version']}")
        assert retrieved_content2 == content2, "Version 2 content does not match"
        
        # Get specific version by number
        retrieved_content1, version_info1 = version_control.get_version(file_path, version_num=1)
        print(f"Retrieved version 1: {retrieved_content1[:20]}...")
        print(f"Version 1 info: {version_info1['version']}")
        assert retrieved_content1 == content1, "Version 1 content does not match"
        
        # Get latest version
        latest_content, latest_info = version_control.get_version(file_path)
        print(f"Retrieved latest version: {latest_content[:20]}...")
        print(f"Latest version info: {latest_info['version']}")
        assert latest_content == content3, "Latest content does not match version 3"
        
        # Rollback to version 1
        success, rollback_content = version_control.rollback(file_path, version_num=1)
        print(f"Rolled back to version 1: {success}")
        assert success, "Rollback should succeed"
        assert rollback_content == content1, "Rollback content does not match version 1"
        
        return True
    except Exception as e:
        print(f"Error in version control test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_neural_pattern_storage():
    """Test the NeuralPatternStorage class."""
    print("\n=== Testing NeuralPatternStorage ===")
    test_dir = setup_test_dir()
    
    try:
        # Initialize neural pattern storage
        pattern_storage = NeuralPatternStorage(os.path.join(test_dir, "neural_patterns"))
        print("Initialized neural pattern storage")
        
        # Create and store a pattern
        pattern_data = {
            "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
            "biases": [-0.1, -0.2, -0.3],
            "activation": "relu",
            "layers": [
                {"size": 5, "type": "input"},
                {"size": 10, "type": "hidden"},
                {"size": 2, "type": "output"}
            ]
        }
        
        pattern_id = pattern_storage.store_pattern(
            "network",
            pattern_data,
            {"description": "Test neural network"}
        )
        print(f"Stored neural pattern with ID: {pattern_id}")
        
        # Retrieve the pattern
        retrieved_pattern, pattern_metadata = pattern_storage.retrieve_pattern(pattern_id)
        print(f"Retrieved pattern: {list(retrieved_pattern.keys())}")
        print(f"Pattern metadata: {pattern_metadata}")
        
        # Verify pattern data
        assert retrieved_pattern["weights"] == pattern_data["weights"], "Pattern weights do not match"
        assert retrieved_pattern["activation"] == pattern_data["activation"], "Pattern activation does not match"
        
        # Update the pattern
        updated_pattern = pattern_data.copy()
        updated_pattern["weights"] = [0.2, 0.3, 0.4, 0.5, 0.6]
        updated_pattern["activation"] = "sigmoid"
        
        success = pattern_storage.update_pattern(
            pattern_id,
            updated_pattern,
            {"update_reason": "Changed activation function"}
        )
        print(f"Updated pattern: {success}")
        
        # Retrieve updated pattern
        retrieved_updated, updated_metadata = pattern_storage.retrieve_pattern(pattern_id)
        print(f"Retrieved updated pattern: {retrieved_updated['activation']}")
        print(f"Updated metadata: {updated_metadata.get('update_reason')}")
        
        # Verify updated pattern
        assert retrieved_updated["activation"] == "sigmoid", "Updated activation does not match"
        assert updated_metadata.get("update_reason") == "Changed activation function", "Update reason not found"
        
        # List patterns
        patterns = pattern_storage.list_patterns()
        print(f"Listed {len(patterns)} patterns")
        
        # Delete pattern
        deleted = pattern_storage.delete_pattern(pattern_id)
        print(f"Deleted pattern: {deleted}")
        
        # Verify deletion
        patterns_after_delete = pattern_storage.list_patterns()
        print(f"Patterns after deletion: {len(patterns_after_delete)}")
        assert len(patterns_after_delete) == 0, "Pattern not deleted"
        
        return True
    except Exception as e:
        print(f"Error in neural pattern storage test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_persistence_manager():
    """Test the PersistenceManager class."""
    print("\n=== Testing PersistenceManager ===")
    test_dir = setup_test_dir()
    
    try:
        # Initialize persistence manager
        db_file = os.path.join(test_dir, "persistence.json")
        persistence = PersistenceManager(db_file)
        print("Initialized persistence manager")
        
        # Store items in different collections
        persistence.store_item("settings", "theme", "dark")
        persistence.store_item("settings", "font_size", 14)
        persistence.store_item("history", "last_run", time.time())
        persistence.store_item("counters", "errors", 0)
        print("Stored items in different collections")
        
        # Retrieve items
        theme = persistence.retrieve_item("settings", "theme")
        font_size = persistence.retrieve_item("settings", "font_size")
        last_run = persistence.retrieve_item("history", "last_run")
        error_count = persistence.retrieve_item("counters", "errors")
        
        print(f"Retrieved items: theme={theme}, font_size={font_size}, error_count={error_count}")
        
        # Verify retrievals
        assert theme == "dark", "Theme does not match"
        assert font_size == 14, "Font size does not match"
        assert error_count == 0, "Error count does not match"
        
        # Update an item
        persistence.store_item("counters", "errors", 5)
        print("Updated error count to 5")
        
        # Verify update
        updated_count = persistence.retrieve_item("counters", "errors")
        print(f"Updated error count: {updated_count}")
        assert updated_count == 5, "Updated error count does not match"
        
        # Get a collection
        settings = persistence.get_collection("settings")
        print(f"Retrieved settings collection: {settings}")
        assert "theme" in settings, "Theme not found in settings collection"
        assert "font_size" in settings, "Font size not found in settings collection"
        
        # Delete an item
        deleted = persistence.delete_item("settings", "font_size")
        print(f"Deleted font_size: {deleted}")
        
        # Verify deletion
        deleted_item = persistence.retrieve_item("settings", "font_size")
        print(f"Deleted item value: {deleted_item}")
        assert deleted_item is None, "Deleted item should return None"
        
        # List collections
        collections = persistence.list_collections()
        print(f"Listed collections: {collections}")
        assert "settings" in collections, "Settings collection not found"
        assert "history" in collections, "History collection not found"
        assert "counters" in collections, "Counters collection not found"
        
        # Clear a collection
        cleared = persistence.clear_collection("counters")
        print(f"Cleared counters collection: {cleared}")
        
        # Verify clear
        counters = persistence.get_collection("counters")
        print(f"Counters after clear: {counters}")
        assert len(counters) == 0, "Counters collection should be empty after clear"
        
        return True
    except Exception as e:
        print(f"Error in persistence manager test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_storage_manager():
    """Test the StorageManager class."""
    print("\n=== Testing StorageManager ===")
    test_dir = setup_test_dir()
    
    try:
        # Initialize storage manager
        manager = create_storage_manager(test_dir)
        print("Initialized storage manager")
        
        # Store a fix
        fix_data = {
            "file": "example.py",
            "line": 42,
            "issue": "Missing return statement",
            "solution": "Add return result",
            "patch": "@@ -42,1 +42,2 @@\n    process_data()\n+    return result"
        }
        
        fix_id = manager.store_fix(fix_data)
        print(f"Stored fix with ID: {fix_id}")
        
        # Store a plan
        plan_data = {
            "steps": [
                {"action": "analyze", "file": "example.py"},
                {"action": "identify", "issue": "Missing return"},
                {"action": "fix", "solution": "Add return statement"}
            ],
            "estimated_time": 5,
            "difficulty": "easy"
        }
        
        plan_id = manager.store_plan(plan_data)
        print(f"Stored plan with ID: {plan_id}")
        
        # Log an error
        error_data = {
            "timestamp": time.time(),
            "level": "ERROR",
            "message": "Failed to parse file",
            "details": {
                "file": "broken.py",
                "reason": "Syntax error",
                "line": 123
            }
        }
        
        manager.log_error(error_data)
        print("Logged an error")
        
        # Get recent errors
        errors = manager.get_recent_errors(5)
        print(f"Retrieved {len(errors)} recent errors")
        assert len(errors) > 0, "Should have at least one error"
        
        # Commit a file version
        file_content = "def example():\n    return True"
        version_id = manager.commit_file_version("example.py", file_content, "Initial version")
        print(f"Committed file version with ID: {version_id}")
        
        # Store a neural pattern
        pattern_data = {
            "type": "bug_pattern",
            "signature": [1, 0, 1, 1, 0],
            "confidence": 0.85
        }
        
        pattern_id = manager.store_neural_pattern("bug", pattern_data)
        print(f"Stored neural pattern with ID: {pattern_id}")
        
        # Persist data
        persisted = manager.persist_data("app_state", "last_session", {
            "time": time.time(),
            "files_processed": 10,
            "bugs_fixed": 3
        })
        print(f"Persisted app state: {persisted}")
        
        # Retrieve persisted data
        app_state = manager.retrieve_persisted_data("app_state", "last_session")
        print(f"Retrieved app state: {app_state is not None}")
        assert app_state is not None, "App state should not be None"
        assert "files_processed" in app_state, "App state missing files_processed"
        
        return True
    except Exception as e:
        print(f"Error in storage manager test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def test_api_functions():
    """Test the API functions."""
    print("\n=== Testing API Functions ===")
    test_dir = setup_test_dir()
    
    try:
        # Test create_storage_manager
        from storage_manager import (
            create_storage_manager, store_fix, store_plan,
            commit_file_version, rollback_file, log_error,
            get_recent_errors, store_neural_pattern,
            persist_data, retrieve_persisted_data
        )
        
        # Create a manager
        manager = create_storage_manager(test_dir)
        print(f"Created manager through API: {manager is not None}")
        assert manager is not None, "Manager should not be None"
        
        # Test individual API functions
        fix_data = {"file": "test.py", "fix": "Add missing import"}
        fix_id = store_fix(fix_data, storage_dir=os.path.join(test_dir, "compressed"))
        print(f"Stored fix through API: {fix_id is not None}")
        
        plan_data = {"steps": ["Analyze", "Fix", "Verify"]}
        plan_id = store_plan(plan_data, storage_dir=os.path.join(test_dir, "compressed"))
        print(f"Stored plan through API: {plan_id is not None}")
        
        file_content = "Test content"
        version_id = commit_file_version(
            "test.txt", file_content, "API test",
            repo_dir=os.path.join(test_dir, "versions")
        )
        print(f"Committed file through API: {version_id is not None}")
        
        success, content = rollback_file(
            "test.txt", version_id=version_id,
            repo_dir=os.path.join(test_dir, "versions")
        )
        print(f"Rolled back file through API: {success}")
        assert content == file_content, "Rollback content does not match"
        
        log_error(
            {"message": "API test error"},
            buffer_file=os.path.join(test_dir, "logs", "error_log.json.gz")
        )
        print("Logged error through API")
        
        errors = get_recent_errors(
            count=5,
            buffer_file=os.path.join(test_dir, "logs", "error_log.json.gz")
        )
        print(f"Retrieved {len(errors)} errors through API")
        
        pattern_id = store_neural_pattern(
            "test", {"data": [1, 2, 3]},
            storage_dir=os.path.join(test_dir, "neural_patterns")
        )
        print(f"Stored neural pattern through API: {pattern_id is not None}")
        
        success = persist_data(
            "api_test", "key1", "value1",
            db_file=os.path.join(test_dir, "persistence", "db.json")
        )
        print(f"Persisted data through API: {success}")
        
        value = retrieve_persisted_data(
            "api_test", "key1",
            db_file=os.path.join(test_dir, "persistence", "db.json")
        )
        print(f"Retrieved persisted data through API: {value}")
        assert value == "value1", "Retrieved value does not match"
        
        return True
    except Exception as e:
        print(f"Error in API functions test: {e}")
        return False
    finally:
        cleanup_test_dir(test_dir)

def main():
    """Main function."""
    print("=== Storage Manager Test Suite ===")
    
    # Run tests
    tests = [
        ("CompressedStorage", test_compressed_storage),
        ("RotatingBuffer", test_rotating_buffer),
        ("VersionControl", test_version_control),
        ("NeuralPatternStorage", test_neural_pattern_storage),
        ("PersistenceManager", test_persistence_manager),
        ("StorageManager", test_storage_manager),
        ("API Functions", test_api_functions)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\nRunning test: {name}")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error running test: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        
        print(f"{name}: {status}")
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
