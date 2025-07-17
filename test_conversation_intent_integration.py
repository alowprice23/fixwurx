#!/usr/bin/env python3
"""
Test script to verify that the conversational interface correctly integrates with
the intent classification system.
"""

import sys
import json
from unittest.mock import MagicMock

# Import the necessary components
from components.conversational_interface import ConversationalInterface
from components.intent_classification_system import IntentClassificationSystem, Intent
from components.state_manager import StateManager
from components.intent_caching_system import IntentOptimizationSystem

class MockRegistry:
    """Mock component registry for testing."""
    
    def __init__(self):
        self.components = {}
    
    def register_component(self, name, component):
        self.components[name] = component
        
    def get_component(self, name):
        return self.components.get(name)

def create_test_environment():
    """Create a test environment with mocked components."""
    registry = MockRegistry()
    
    # Create and register necessary components
    state_manager = StateManager(registry)
    registry.register_component("state_manager", state_manager)
    
    command_executor = MagicMock()
    command_executor.execute.return_value = {"success": True, "output": "Command executed successfully"}
    registry.register_component("command_executor", command_executor)
    
    file_access = MagicMock()
    file_access.read_file.return_value = {"success": True, "content": "Test file content", "size": 18}
    registry.register_component("file_access_utility", file_access)
    
    # Create an intent classification system
    intent_system = IntentClassificationSystem(registry)
    registry.register_component("intent_classification_system", intent_system)
    
    # Create a caching system
    caching_system = IntentOptimizationSystem(cache_capacity=10, history_size=20, window_size=10)
    registry.register_component("intent_optimization_system", caching_system)
    
    # Create a conversation logger
    conversation_logger = MagicMock()
    conversation_logger.start_conversation.return_value = {"success": True, "conversation_id": "test-conversation"}
    registry.register_component("conversation_logger", conversation_logger)
    
    # Create a mock LLM client
    llm_client = MagicMock()
    llm_client.chat.return_value = {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the LLM."
                }
            }
        ]
    }
    registry.register_component("llm_client", llm_client)
    
    # Create a mock planning engine
    planning_engine = MagicMock()
    planning_engine.generate_plan.return_value = {
        "success": True,
        "goal": "Test goal",
        "steps": [
            {"description": "Step 1: Test step"}
        ]
    }
    registry.register_component("planning_engine", planning_engine)
    
    # Create the conversational interface
    conversational_interface = ConversationalInterface(registry)
    
    return registry, conversational_interface

def test_direct_intent_classification():
    """Test direct intent classification and execution."""
    registry, ci = create_test_environment()
    
    # Test a file access intent
    response = ci.process_input("read file test.txt")
    print("File Access Response:", response)
    
    # Test a command execution intent
    response = ci.process_input("!ls -la")
    print("Command Execution Response:", response)
    
    # Test a complex intent that requires planning
    response = ci.process_input("I need to debug this Python script")
    print("Planning Intent Response:", response)
    
    # Add an entry to the conversation history to test context awareness
    ci.history.append({"role": "user", "content": "read file test.txt"})
    ci.history.append({"role": "system", "content": "File: test.txt (18 bytes)\n\nTest file content"})
    
    # Test a follow-up intent with context reference
    response = ci.process_input("modify it to say Hello World")
    print("Context-aware Follow-up Response:", response)
    
    # Test agent introspection
    response = ci.process_input("how are you performing?")
    print("Agent Introspection Response:", response)

def test_fallback_mechanism():
    """Test that the fallback mechanism works when agents fail."""
    registry, ci = create_test_environment()
    
    # Mock the agent collaboration hub to simulate a failed agent
    agent_hub = registry.get_component("conversational_interface").agent_collaboration_hub
    original_orchestrate = agent_hub.orchestrate
    
    def mock_orchestrate(intent):
        if intent.type == "system_debugging":
            # Simulate a failed agent
            failed_agents = ["analyst"]
            updated_agents, fallbacks = registry.get_component("intent_classification_system").handle_agent_failure(intent, failed_agents)
            return f"Agent failure handled. Updated agents: {updated_agents}, Fallbacks: {fallbacks}"
        return original_orchestrate(intent)
    
    agent_hub.orchestrate = mock_orchestrate
    
    # Test a request that will trigger agent collaboration with a failure
    response = ci.process_input("debug the system")
    print("Agent Fallback Response:", response)

def test_caching_mechanism():
    """Test that the caching mechanism works for repeated queries."""
    registry, ci = create_test_environment()
    
    # First query to cache
    first_response = ci.process_input("read file test.txt")
    print("First Query Response:", first_response)
    
    # Monitor what happens with the intent classification system
    intent_system = registry.get_component("intent_classification_system")
    original_classify = intent_system.classify_intent
    
    # Create a mock to track classification calls
    classification_count = 0
    def mock_classify(query, context):
        nonlocal classification_count
        classification_count += 1
        return original_classify(query, context)
    
    intent_system.classify_intent = mock_classify
    
    # Repeat the same query - should use cache
    second_response = ci.process_input("read file test.txt")
    print("Second Query Response:", second_response)
    
    # Check if classification was called
    print(f"Classification called {classification_count} times (should be 1 if caching works)")

if __name__ == "__main__":
    print("===== Testing Direct Intent Classification =====")
    test_direct_intent_classification()
    
    print("\n===== Testing Fallback Mechanism =====")
    test_fallback_mechanism()
    
    print("\n===== Testing Caching Mechanism =====")
    test_caching_mechanism()
    
    print("\nAll tests completed.")
