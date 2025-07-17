#!/usr/bin/env python3
"""
Simple test runner for the triangulum_integration module

This script runs basic tests on the triangulum_components to verify
that our implementation is working correctly.
"""

import os
import sys
import unittest
import time

# Force MOCK_MODE for testing
os.environ["TRIANGULUM_TEST_MODE"] = "1"

try:
    # Import directly from components
    from triangulum_components.system_monitor import SystemMonitor
    from triangulum_components.dashboard import DashboardVisualizer
    from triangulum_components.queue_manager import QueueManager
    
    # Define mock mode flag
    MOCK_MODE = True
    print("Successfully imported triangulum components")
    print(f"MOCK_MODE = {MOCK_MODE}")
except ImportError as e:
    print(f"Error importing triangulum components: {e}")
    sys.exit(1)

class TestTriangulumIntegration(unittest.TestCase):
    """Test cases for Triangulum Integration"""
    
    def test_system_monitor(self):
        """Test the SystemMonitor class"""
        monitor = SystemMonitor()
        self.assertIsNotNone(monitor)
        
        # Check metrics initialization
        metrics = monitor.get_metrics()
        self.assertIn("system", metrics)
        self.assertIn("application", metrics)
        self.assertIn("resources", metrics)
        self.assertIn("triangulum", metrics)
        
        # Update application metrics
        app_metrics = {
            "tasks_pending": 5,
            "tasks_running": 2,
            "tasks_completed": 10,
            "tasks_failed": 1,
            "error_rate": 9.1,
            "average_task_duration": 3.5
        }
        monitor.update_application_metrics(app_metrics)
        
        # Check updated metrics
        updated_metrics = monitor.get_metrics()
        self.assertEqual(updated_metrics["application"]["tasks_pending"], 5)
    
    def test_queue_manager(self):
        """Test the QueueManager class"""
        queue_manager = QueueManager()
        self.assertIsNotNone(queue_manager)
        
        # Check default queue exists
        self.assertIn("default", queue_manager.get_queue_names())
        
        # Test queue operations
        success = queue_manager.create_queue("test_queue")
        self.assertTrue(success)
        
        success = queue_manager.enqueue("test_item", "test_queue")
        self.assertTrue(success)
        
        # Check the queue has an item
        self.assertEqual(queue_manager.get_queue_size("test_queue"), 1)
        
        # Dequeue and check the item
        item = queue_manager.dequeue("test_queue")
        self.assertEqual(item, "test_item")
    
    def test_dashboard_visualizer(self):
        """Test the DashboardVisualizer class"""
        config = {"dashboard_port": 8888}
        visualizer = DashboardVisualizer(config)
        self.assertIsNotNone(visualizer)
        
        # Just test initialization since actual HTTP server might be difficult in test env
        self.assertEqual(visualizer.dashboard_port, 8888)
        self.assertEqual(visualizer.dashboard_host, "localhost")
        self.assertEqual(visualizer.dashboard_url, "http://localhost:8888")

if __name__ == "__main__":
    unittest.main()
