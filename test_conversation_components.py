#!/usr/bin/env python3
"""
Test Conversation Components

A focused test script that tests the conversational interface and history management
components that we've implemented, without requiring the full agent system.
"""

import os
import sys
import time
import json
import unittest
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("conversation_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ConversationComponentsTest")

class ConversationComponentsTest(unittest.TestCase):
    """
    Test suite for Conversational Interface and History Management components.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment.
        """
        logger.info("Setting up test environment")
        
        # Create a temp directory for conversation history
        cls.test_history_dir = "test_conversation_history"
        os.makedirs(cls.test_history_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up test environment.
        """
        logger.info("Cleaning up test environment")
        
        # Remove the test conversation history directory
        if os.path.exists(cls.test_history_dir):
            shutil.rmtree(cls.test_history_dir)
    
    def test_conversational_interface_import(self):
        """
        Test that the conversational interface can be imported and initialized.
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
        """
        logger.info("Testing conversation history manager")
        
        try:
            # Import the conversation history manager
            from conversation_history_manager import ConversationHistoryManager, get_instance
            
            # Initialize the history manager with test directory
            config = {
                "storage_dir": self.test_history_dir
            }
            history_manager = ConversationHistoryManager(config)
            
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
    
    def test_conversation_history_operations(self):
        """
        Test the basic operations of the conversation history manager.
        """
        logger.info("Testing conversation history operations")
        
        try:
            # Import the conversation history manager
            from conversation_history_manager import ConversationHistoryManager
            
            # Initialize the history manager with test directory
            config = {
                "storage_dir": self.test_history_dir
            }
            history_manager = ConversationHistoryManager(config)
            
            # Create a test session
            session_id = history_manager.create_session("test_session")
            self.assertIsNotNone(session_id, "Failed to create session")
            
            # Add messages
            system_msg = history_manager.add_message(session_id, "system", "Test system message")
            user_msg = history_manager.add_message(session_id, "user", "Test user message")
            assistant_msg = history_manager.add_message(session_id, "assistant", "Test assistant response")
            
            # Verify messages were added
            self.assertEqual(system_msg["role"], "system", "System message role incorrect")
            self.assertEqual(user_msg["role"], "user", "User message role incorrect")
            self.assertEqual(assistant_msg["role"], "assistant", "Assistant message role incorrect")
            
            # Get session history
            history = history_manager.get_session_history(session_id)
            self.assertEqual(len(history), 3, "Session history should have 3 messages")
            
            # Get formatted history
            formatted_history = history_manager.get_session_history(session_id, formatted=True)
            self.assertEqual(len(formatted_history), 3, "Formatted history should have 3 messages")
            self.assertEqual(formatted_history[0]["role"], "system", "First message should be system")
            self.assertEqual(formatted_history[1]["role"], "user", "Second message should be user")
            self.assertEqual(formatted_history[2]["role"], "assistant", "Third message should be assistant")
            
            # Get latest messages
            latest = history_manager.get_latest_messages(session_id, count=2)
            self.assertEqual(len(latest), 2, "Latest messages should have 2 items")
            self.assertEqual(latest[0]["role"], "user", "First latest message should be user")
            self.assertEqual(latest[1]["role"], "assistant", "Second latest message should be assistant")
            
            # Save session
            saved = history_manager.save_session(session_id)
            self.assertTrue(saved, "Failed to save session")
            
            # Verify file was created
            files = os.listdir(self.test_history_dir)
            self.assertTrue(any(session_id in f for f in files), "Session file not found")
            
            # Close session
            closed = history_manager.close_session(session_id, save=False)
            self.assertTrue(closed, "Failed to close session")
            
            # Try to get history for closed session
            closed_history = history_manager.get_session_history(session_id)
            self.assertEqual(len(closed_history), 0, "Closed session should return empty history")
            
            # Load session
            loaded = history_manager.load_session(session_id)
            self.assertTrue(loaded, "Failed to load session")
            
            # Verify loaded session
            loaded_history = history_manager.get_session_history(session_id)
            self.assertEqual(len(loaded_history), 3, "Loaded session should have 3 messages")
            
            logger.info("Conversation history operations test completed")
        except Exception as e:
            self.fail(f"Error testing conversation history operations: {e}")
    
    def test_conversation_summarization(self):
        """
        Test the conversation summarization functionality.
        """
        logger.info("Testing conversation summarization")
        
        try:
            # Import the conversation history manager
            from conversation_history_manager import ConversationHistoryManager
            
            # Initialize the history manager with test directory and small limits
            config = {
                "storage_dir": self.test_history_dir,
                "max_history_items": 10,
                "summarize_after": 5
            }
            history_manager = ConversationHistoryManager(config)
            
            # Create a test session
            session_id = history_manager.create_session("summarize_test")
            
            # Add system message
            history_manager.add_message(session_id, "system", "Initial system message")
            
            # Add several messages to trigger summarization
            for i in range(7):
                history_manager.add_message(session_id, "user", f"User message {i}")
                history_manager.add_message(session_id, "assistant", f"Assistant response {i}")
            
            # Create a summary
            test_summary = "This is a test summary of the conversation."
            summarized = history_manager.summarize_session(session_id, summary=test_summary, keep_recent=3)
            
            # Verify summarization
            self.assertTrue(summarized, "Summarization failed")
            
            # Get history after summarization
            history = history_manager.get_session_history(session_id)
            
            # Verify structure after summarization
            self.assertLess(len(history), 15, "History should be reduced after summarization")
            
            # Check for summary message
            has_summary = False
            for msg in history:
                if msg["role"] == "system" and "summary" in msg["content"]:
                    has_summary = True
                    break
            
            self.assertTrue(has_summary, "Summary message not found in history")
            
            logger.info("Conversation summarization test completed")
        except Exception as e:
            self.fail(f"Error testing conversation summarization: {e}")
    
    def test_integration_with_conversational_interface(self):
        """
        Test the integration between the conversational interface and history manager.
        """
        logger.info("Testing integration with conversational interface")
        
        try:
            # Import required modules
            from conversational_interface import ConversationalInterface
            from conversation_history_manager import get_instance as get_history_manager, ConversationHistoryManager
            
            # Reset the singleton and configure our test instance
            import conversation_history_manager
            
            # Configure history manager
            config = {
                "storage_dir": self.test_history_dir
            }
            
            # First reset the singleton
            conversation_history_manager._instance = None
            
            # Then set our instance as the singleton
            history_manager = ConversationHistoryManager(config)
            conversation_history_manager._instance = history_manager
            
            # Create mock objects
            class MockMetaAgent:
                def __init__(self):
                    self.initialized = True
                
                def process_query(self, query, context):
                    return f"Mock response to: {query}"
                
                def start_oversight(self):
                    pass
            
            class MockConversationLogger:
                def __init__(self):
                    pass
                
                def create_session(self, session_id):
                    return True
                
                def close_session(self, session_id):
                    return True
                
                def log_user_query(self, session_id, query):
                    return True
                
                def log_agent_response(self, session_id, agent_id, response, success, llm_used):
                    return True
            
            class MockACS:
                def __init__(self):
                    pass
                
                def register_agent(self, agent_id, agent_type, capabilities):
                    return True
                
                def deactivate_agent(self, agent_id):
                    return True
            
            class MockRegistry:
                def __init__(self):
                    self.components = {}
                
                def register_component(self, name, component):
                    self.components[name] = component
                
                def get_component(self, name):
                    if name == "meta_agent":
                        return MockMetaAgent()
                    return self.components.get(name)
            
            # Initialize the conversational interface with mocks
            registry = MockRegistry()
            ci = ConversationalInterface(registry)
            
            # Inject our mocks
            ci.conversation_logger = MockConversationLogger()
            ci.acs = MockACS()
            
            # Start the interface
            ci.start()
            self.assertTrue(ci.initialized, "Conversational interface not initialized")
            self.assertIsNotNone(ci.current_session_id, "No session ID created")
            
            # Get the history manager and verify it's the same instance
            test_manager = get_history_manager()
            self.assertEqual(test_manager, history_manager, "History manager singleton not working")
            
            # Manually add greeting to history (this would normally happen in display_greeting)
            test_manager.add_message(ci.current_session_id, "system", "Hi, how can I help you today?")
            
            # Manually add user message to history (this would normally happen in run_interaction_loop)
            test_user_input = "Hello, this is a test"
            test_manager.add_message(ci.current_session_id, "user", test_user_input)
            
            # Process a user input
            response = ci.process_user_input(test_user_input)
            self.assertIsNotNone(response, "No response returned")
            
            # Manually add assistant response to history (this would normally happen in run_interaction_loop)
            test_manager.add_message(ci.current_session_id, "assistant", response)
            
            # Verify history was updated
            history = test_manager.get_session_history(ci.current_session_id)
            self.assertGreaterEqual(len(history), 3, "Not enough history entries created")
            
            # Check for specific message types
            system_messages = [msg for msg in history if msg["role"] == "system"]
            user_messages = [msg for msg in history if msg["role"] == "user"]
            assistant_messages = [msg for msg in history if msg["role"] == "assistant"]
            
            self.assertGreaterEqual(len(system_messages), 1, "No system messages in history")
            self.assertGreaterEqual(len(user_messages), 1, "No user messages in history")
            self.assertGreaterEqual(len(assistant_messages), 1, "No assistant messages in history")
            
            # Check content of messages
            self.assertEqual(user_messages[0]["content"], test_user_input, "User message content doesn't match")
            self.assertTrue(assistant_messages[0]["content"].startswith("Mock response to:"), "Assistant message content doesn't match")
            
            # Shutdown the interface
            ci.shutdown()
            
            logger.info("Integration test completed")
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
        except Exception as e:
            self.fail(f"Error testing integration: {e}")


def run_tests():
    """
    Run the test suite.
    """
    logger.info("Starting Conversation Components Test")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Conversation Components Test completed")

if __name__ == "__main__":
    run_tests()
