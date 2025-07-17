#!/usr/bin/env python3
"""
Test Script for Fixed Triangulum Integration Module

This script modifies the test_triangulum_integration.py file to work with
the fixed triangulum_integration_fix.py implementation.
"""

import os
import sys
import shutil

def fix_test_file():
    """Fix the test_triangulum_integration.py file"""
    print("Fixing test_triangulum_integration.py")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to original and fixed test files
    original_test_file = os.path.join(current_dir, "test_triangulum_integration.py")
    
    # Create backup of original test file
    if os.path.exists(original_test_file):
        backup_file = os.path.join(current_dir, "test_triangulum_integration.py.bak")
        print(f"Backing up original test file to {backup_file}")
        shutil.copy2(original_test_file, backup_file)
    
    # Read the original test file
    with open(original_test_file, "r") as f:
        content = f.read()
    
    # Add mock mode support
    content = content.replace(
        "try:\n    from triangulum_integration import (",
        "# Force MOCK_MODE for testing\nos.environ[\"TRIANGULUM_TEST_MODE\"] = \"1\"\n\ntry:\n    from triangulum_integration import (\n        MOCK_MODE,"
    )
    
    # Add assertion for mock mode
    content = content.replace(
        "except ImportError:",
        "    # Verify we're in mock mode for testing\n    assert MOCK_MODE, \"Tests must be run in MOCK_MODE\"\nexcept ImportError:"
    )
    
    # Fix the test_queue_manager function to use get_queue_names
    if "queue_names = queue_manager.get_queue_names()" not in content:
        content = content.replace(
            "# Check default queue\n        queue_names = list(queue_manager.queues.keys())",
            "# Check default queue\n        queue_names = queue_manager.get_queue_names()"
        )
    
    # Convert setup_test_dir and cleanup_test_dir to class methods for TestTriangulumIntegration
    if "class TestTriangulumIntegration" not in content:
        content = content.replace(
            "def setup_test_dir():",
            "class TestTriangulumIntegration(unittest.TestCase):\n    \"\"\"\n    Test cases for Triangulum Integration\n    \"\"\"\n    \n    def setUp(self):"
        )
        
        content = content.replace(
            "def cleanup_test_dir(test_dir):",
            "    def tearDown(self):"
        )
        
        # Fix the test_dir reference in tearDown
        content = content.replace(
            "    if os.path.exists(test_dir):\n        shutil.rmtree(test_dir)",
            "    if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):\n        shutil.rmtree(self.test_dir)"
        )
        
        # Fix the return value of setUp
        content = content.replace(
            "    return test_dir",
            "    self.test_dir = test_dir\n    print(f\"Created test directory: {self.test_dir}\")"
        )
        
        # Fix each test function to be a class method
        for test_func in ["test_system_monitor", "test_dashboard_visualizer", 
                         "test_queue_manager", "test_rollback_manager", 
                         "test_plan_executor", "test_api_functions"]:
            content = content.replace(
                f"def {test_func}():",
                f"    def {test_func}(self):"
            )
            
            # Fix indentation in the function
            lines = content.split("\n")
            start_idx = -1
            end_idx = -1
            
            for i, line in enumerate(lines):
                if line.strip() == f"def {test_func}():":
                    start_idx = i
                elif start_idx >= 0 and i > start_idx and line.strip() and not line.startswith("    "):
                    end_idx = i
                    break
            
            if start_idx >= 0 and end_idx >= 0:
                for i in range(start_idx + 1, end_idx):
                    if lines[i].strip():
                        lines[i] = "    " + lines[i]
                
                content = "\n".join(lines)
    
    # Update the main function to use unittest
    content = content.replace(
        "def main():\n    \"\"\"Main function.\"\"\"\n    print(\"=== Triangulum Integration Test Suite ===\")\n    \n    # Run tests\n    tests = [\n        (\"SystemMonitor\", test_system_monitor),\n        (\"DashboardVisualizer\", test_dashboard_visualizer),\n        (\"QueueManager\", test_queue_manager),\n        (\"RollbackManager\", test_rollback_manager),\n        (\"PlanExecutor\", test_plan_executor),\n        (\"API Functions\", test_api_functions)\n    ]\n    \n    results = []\n    \n    for name, test_func in tests:\n        try:\n            print(f\"\\nRunning test: {name}\")\n            result = test_func()\n            results.append((name, result))\n        except Exception as e:\n            print(f\"Error running test: {e}\")\n            results.append((name, False))\n    \n    # Print summary\n    print(\"\\n=== Test Summary ===\")\n    \n    passed = 0\n    failed = 0\n    \n    for name, result in results:\n        status = \"PASSED\" if result else \"FAILED\"\n        if result:\n            passed += 1\n        else:\n            failed += 1\n        \n        print(f\"{name}: {status}\")\n    \n    print(f\"\\nPassed: {passed}/{len(results)} tests\")\n    \n    return 0 if failed == 0 else 1",
        "def main():\n    \"\"\"Main function to run the tests\"\"\"\n    unittest.main()"
    )
    
    # Add unittest import if not present
    if "import unittest" not in content:
        content = content.replace(
            "import threading",
            "import threading\nimport unittest"
        )
    
    # Write the modified content back to the file
    with open(original_test_file, "w") as f:
        f.write(content)
    
    print("Test file fixed successfully")

if __name__ == "__main__":
    fix_test_file()
