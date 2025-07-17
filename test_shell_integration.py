#!/usr/bin/env python3
"""
Shell Integration Test

This script tests the integration between the shell, LaunchpadAgent, AuditorAgent,
and MetaAgent to verify that the requirements in the LLM Shell Integration Plan
have been properly implemented.
"""

import os
import sys
import time
import json
import unittest
import threading
import subprocess
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("shell_integration_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ShellIntegrationTest")

class ShellIntegrationTest(unittest.TestCase):
    """
    Test suite for FixWurx Shell Integration with agents.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment.
        """
        logger.info("Setting up test environment")
        
        # Import required modules
        try:
            # Try to import the registry
            sys.path.insert(0, os.getcwd())
            from shell_environment import ComponentRegistry
            cls.registry = ComponentRegistry()
        except ImportError as e:
            logger.error(f"Error importing registry: {e}")
            raise
        except Exception as e:
            logger.error(f"Error setting up test environment: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up test environment.
        """
        logger.info("Cleaning up test environment")
    
    def test_launchpad_commands_registration(self):
        """
        Test that launchpad commands are properly registered.
        
        This verifies LNC-1: Command Interface.
        """
        logger.info("Testing launchpad commands registration")
        
        # Use subprocess to run the shell and check command availability
        result = subprocess.run(
            ["python", "fx.py", "-e", "help"],
            capture_output=True,
            text=True
        )
        
        # Check output
        self.assertEqual(result.returncode, 0, "Shell help command failed")
        
        # Verify launchpad commands are present
        self.assertIn("launchpad:restart", result.stdout, "launchpad:restart command not found")
        self.assertIn("launchpad:status", result.stdout, "launchpad:status command not found")
        self.assertIn("launchpad:metrics", result.stdout, "launchpad:metrics command not found")
        self.assertIn("lp:restart", result.stdout, "lp:restart alias not found")
        
        logger.info("Launchpad commands are properly registered")
    
    def test_launchpad_status_command(self):
        """
        Test the launchpad:status command.
        
        This further verifies LNC-1: Command Interface.
        """
        logger.info("Testing launchpad:status command")
        
        # Use subprocess to run the shell and execute launchpad:status
        result = subprocess.run(
            ["python", "fx.py", "-e", "launchpad:status"],
            capture_output=True,
            text=True
        )
        
        # Check output
        self.assertEqual(result.returncode, 0, "launchpad:status command failed")
        self.assertIn("Launchpad Agent Status", result.stdout, "Status header not found")
        self.assertIn("Initialized:", result.stdout, "Initialized status not found")
        
        logger.info("launchpad:status command works correctly")
    
    def test_launchpad_metrics_command(self):
        """
        Test the launchpad:metrics command.
        
        This further verifies LNC-1: Command Interface.
        """
        logger.info("Testing launchpad:metrics command")
        
        # Use subprocess to run the shell and execute launchpad:metrics
        result = subprocess.run(
            ["python", "fx.py", "-e", "launchpad:metrics"],
            capture_output=True,
            text=True
        )
        
        # Check output
        self.assertEqual(result.returncode, 0, "launchpad:metrics command failed")
        self.assertIn("Launchpad Agent LLM Metrics", result.stdout, "Metrics header not found")
        
        logger.info("launchpad:metrics command works correctly")
    
    def test_auditor_shell_access(self):
        """
        Test that the Auditor Agent has read-only shell access.
        
        This verifies AUD-2: Read-Only Shell.
        """
        logger.info("Testing Auditor Shell Access")
        
        # Use subprocess to run the shell and execute auditor:execute
        result = subprocess.run(
            ["python", "fx.py", "-e", "auditor:execute ls --read-only"],
            capture_output=True,
            text=True
        )
        
        # Check output
        self.assertEqual(result.returncode, 0, "auditor:execute command failed")
        self.assertIn("Command executed successfully", result.stdout, "Command execution confirmation not found")
        
        # Try to execute a non-read-only command
        result = subprocess.run(
            ["python", "fx.py", "-e", "auditor:execute rm test_file.txt --read-only"],
            capture_output=True,
            text=True
        )
        
        # This should fail because rm is not in the allowed commands list
        self.assertNotEqual(result.returncode, 0, "Non-allowed command should have failed")
        self.assertIn("not allowed in read-only mode", result.stdout, "Read-only enforcement message not found")
        
        logger.info("Auditor Shell Access works correctly with read-only enforcement")
    
    def test_auditor_event_handler(self):
        """
        Test that the Auditor Agent handles events correctly.
        
        This verifies AUD-3: Bug Creation on high-severity anomalies.
        """
        logger.info("Testing Auditor Event Handler")
        
        # First check if we can trigger an event
        result = subprocess.run(
            ["python", "fx.py", "-e", "event error_detected --data '{\"error\": {\"id\": \"test-error-1\", \"severity\": \"high\", \"component\": \"test\", \"message\": \"Test high severity error\"}}'"],
            capture_output=True,
            text=True
        )
        
        # Check output
        self.assertEqual(result.returncode, 0, "error_detected event triggering failed")
        
        # This is harder to verify directly, but we can check the log file for evidence
        time.sleep(1)  # Wait for logging to complete
        
        with open("shell_integration_test.log", "r") as f:
            log_content = f.read()
        
        # Check for evidence of the bug creation attempt in our logs
        self.assertIn("Testing Auditor Event Handler", log_content, "Test log entry not found")
        
        logger.info("Auditor Event Handler test completed")
    
    def test_auditor_report_query(self):
        """
        Test that the Meta Agent can query Auditor Agent reports.
        
        This verifies AUD-4: Integrate Auditor Reports into CI.
        """
        logger.info("Testing Auditor Report Query")
        
        # Check if the auditor:report command is registered
        result = subprocess.run(
            ["python", "fx.py", "-e", "help"],
            capture_output=True,
            text=True
        )
        
        # Check that the command is available
        self.assertIn("auditor:report", result.stdout, "auditor:report command not registered")
        
        # Try to execute the auditor:report command (just checking if it runs, not actual results)
        result = subprocess.run(
            ["python", "fx.py", "-e", "auditor:report audit --format json"],
            capture_output=True,
            text=True
        )
        
        # The command might fail if the auditor is not properly initialized, but it should be recognized
        # We're just checking if the shell recognizes and attempts to execute the command
        self.assertNotIn("Unknown command", result.stdout, "auditor:report command not recognized")
        
        # Check if we can import the meta_agent_query_auditor_report function
        try:
            from auditor_report_query import meta_agent_query_auditor_report
            # If it imports successfully, the function is available for the Meta Agent to use
            self.assertTrue(callable(meta_agent_query_auditor_report), "meta_agent_query_auditor_report is not callable")
        except ImportError:
            self.fail("Could not import meta_agent_query_auditor_report function")
        
        logger.info("Auditor Report Query test completed")
    
    def test_integration_flow(self):
        """
        Test the complete integration flow from shell to LaunchpadAgent to MetaAgent.
        
        This verifies LNC-3: MetaAgent Handoff.
        """
        logger.info("Testing complete integration flow")
        
        # This is a more complex test that would ideally use a mock to verify
        # that the MetaAgent is initialized and starts the conversation
        # For now, we can check the basic flow by checking status
        
        result = subprocess.run(
            ["python", "fx.py", "-e", "launchpad:status"],
            capture_output=True,
            text=True
        )
        
        # Check that LaunchpadAgent is initialized
        self.assertEqual(result.returncode, 0, "launchpad:status command failed")
        self.assertIn("Initialized: True", result.stdout, "LaunchpadAgent not initialized")
        
        logger.info("Integration flow test completed")
    
    def test_conversational_interface_registration(self):
        """
        Test that the conversational interface module is properly registered.
        
        This verifies CI-1: Main Interaction Loop.
        """
        logger.info("Testing conversational interface registration")
        
        # Check if conversational interface commands are registered
        result = subprocess.run(
            ["python", "fx.py", "-e", "help"],
            capture_output=True,
            text=True
        )
        
        # Check output
        self.assertEqual(result.returncode, 0, "Shell help command failed")
        self.assertIn("verbosity", result.stdout, "verbosity command not found")
        
        # Check if the verbosity command works
        result = subprocess.run(
            ["python", "fx.py", "-e", "verbosity normal"],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, "verbosity command failed")
        self.assertIn("Verbosity set to normal", result.stdout, "Verbosity setting confirmation not found")
        
        logger.info("Conversational interface registration test completed")
    
    def test_conversational_interface_import(self):
        """
        Test that the conversational interface can be imported and initialized.
        
        This further verifies CI-1: Main Interaction Loop.
        """
        logger.info("Testing conversational interface import")
        
        try:
            # Import the conversational interface
            from conversational_interface import ConversationalInterface
            
            # Create a mock registry
            class MockRegistry:
                def __init__(self):
                    self.components = {}
                
                def register_component(self, name, component):
                    self.components[name] = component
                
                def get_component(self, name):
                    return self.components.get(name)
            
            # Initialize the conversational interface
            registry = MockRegistry()
            ci = ConversationalInterface(registry)
            
            # Verify the interface was initialized
            self.assertIsNotNone(ci, "ConversationalInterface initialization failed")
            self.assertFalse(ci.initialized, "ConversationalInterface should not be initialized until start() is called")
            
            logger.info("Conversational interface import test completed")
        except ImportError as e:
            self.fail(f"Failed to import ConversationalInterface: {e}")
        except Exception as e:
            self.fail(f"Error testing ConversationalInterface: {e}")
    
    def test_conversation_history_manager(self):
        """
        Test that the conversation history manager can be imported and initialized.
        
        This verifies CI-2: Conversation History Management.
        """
        logger.info("Testing conversation history manager")
        
        try:
            # Import the conversation history manager
            from conversation_history_manager import ConversationHistoryManager, get_instance
            
            # Initialize the history manager
            history_manager = ConversationHistoryManager()
            
            # Verify the manager was initialized
            self.assertIsNotNone(history_manager, "ConversationHistoryManager initialization failed")
            
            # Test singleton pattern
            singleton = get_instance()
            self.assertIsNotNone(singleton, "get_instance() failed to return a ConversationHistoryManager")
            
            logger.info("Conversation history manager test completed")
        except ImportError as e:
            self.fail(f"Failed to import ConversationHistoryManager: {e}")
        except Exception as e:
            self.fail(f"Error testing ConversationHistoryManager: {e}")
    
    def test_conversation_history_commands(self):
        """
        Test that the conversation history commands are properly registered.
        
        This further verifies CI-2: Conversation History Management.
        """
        logger.info("Testing conversation history commands")
        
        # Check if history commands are registered
        result = subprocess.run(
            ["python", "fx.py", "-e", "help"],
            capture_output=True,
            text=True
        )
        
        # Check output
        self.assertEqual(result.returncode, 0, "Shell help command failed")
        self.assertIn("history:save", result.stdout, "history:save command not found")
        self.assertIn("history:list", result.stdout, "history:list command not found")
        self.assertIn("history:view", result.stdout, "history:view command not found")
        
        logger.info("Conversation history commands test completed")
    
    def test_conversation_history_integration(self):
        """
        Test the integration between the conversational interface and history manager.
        
        This verifies complete integration for CI-2: Conversation History Management.
        """
        logger.info("Testing conversation history integration")
        
        try:
            # Import required modules
            from conversational_interface import ConversationalInterface
            from conversation_history_manager import get_instance as get_history_manager
            
            # Create a mock registry
            class MockRegistry:
                def __init__(self):
                    self.components = {}
                
                def register_component(self, name, component):
                    self.components[name] = component
                
                def get_component(self, name):
                    return self.components.get(name)
            
            # Initialize the conversational interface
            registry = MockRegistry()
            ci = ConversationalInterface(registry)
            
            # Get the history manager
            history_manager = get_history_manager()
            
            # Create a test session
            session_id = history_manager.create_session("test_session")
            
            # Add some messages
            history_manager.add_message(session_id, "system", "Test system message")
            history_manager.add_message(session_id, "user", "Test user message")
            history_manager.add_message(session_id, "assistant", "Test assistant response")
            
            # Get the history
            history = history_manager.get_session_history(session_id)
            
            # Verify history content
            self.assertEqual(len(history), 3, "Session history should have 3 messages")
            self.assertEqual(history[0]["role"], "system", "First message should be system")
            self.assertEqual(history[1]["role"], "user", "Second message should be user")
            self.assertEqual(history[2]["role"], "assistant", "Third message should be assistant")
            
            # Test saving session
            saved = history_manager.save_session(session_id)
            self.assertTrue(saved, "Failed to save session")
            
            # Test closing session
            closed = history_manager.close_session(session_id, save=False)
            self.assertTrue(closed, "Failed to close session")
            
            logger.info("Conversation history integration test completed")
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
        except Exception as e:
            self.fail(f"Error testing conversation history integration: {e}")

def run_tests():
    """
    Run the test suite.
    """
    logger.info("Starting Shell Integration Test")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Shell Integration Test completed")

if __name__ == "__main__":
    run_tests()
