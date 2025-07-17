#!/usr/bin/env python3
"""
Intent Classification Demo (Standalone)

This script demonstrates the core functionality of the intent classification system
without external dependencies like Neural Matrix and Triangulum.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Set

# Import the core components we need
from components.intent_classification_system import IntentClassificationSystem, Intent
from components.intent_caching_system import IntentOptimizationSystem
from components.state_manager import StateManager
from components.conversational_interface import ConversationalInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intent_demo.log"), logging.StreamHandler()]
)
logger = logging.getLogger("IntentClassificationDemo")

class ComponentRegistry:
    """Simple component registry for managing system components."""
    
    def __init__(self):
        """Initialize the component registry."""
        self.components = {}
        
    def register_component(self, name: str, component: Any) -> None:
        """Register a component with the registry."""
        self.components[name] = component
        logger.info(f"Registered component: {name}")
        
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component from the registry."""
        return self.components.get(name)

class SimpleFileAccessUtility:
    """Simple file access utility that can read and write files."""
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file and return its content."""
        try:
            if not os.path.exists(path):
                return {"success": False, "error": f"File not found: {path}"}
            
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            return {
                "success": True,
                "content": content,
                "size": len(content),
                "is_binary": False
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(path, 'w', encoding='utf-8') as file:
                file.write(content)
                
            return {
                "success": True,
                "message": f"File written successfully: {path}",
                "size": len(content)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_directory(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        """List files in a directory."""
        try:
            if not os.path.exists(path):
                return {"success": False, "error": f"Directory not found: {path}"}
            
            if not os.path.isdir(path):
                return {"success": False, "error": f"Not a directory: {path}"}
            
            files = []
            
            if recursive:
                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
            else:
                # Just list files in the directory
                files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                
            return {
                "success": True,
                "files": files,
                "count": len(files)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class SimpleCommandExecutor:
    """Simple command executor that can run commands."""
    
    def execute(self, command: str, user_id: str) -> Dict[str, Any]:
        """Execute a command and return its output."""
        try:
            # For the demo, just echo the command back
            return {
                "success": True,
                "output": f"Simulated execution of: {command}",
                "command": command,
                "user_id": user_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

def initialize_intent_system() -> ComponentRegistry:
    """
    Initialize the intent system with the basic components.
    
    Returns:
        Initialized component registry
    """
    # Create registry
    registry = ComponentRegistry()
    
    # Initialize state manager
    state_manager = StateManager(registry)
    registry.register_component("state_manager", state_manager)
    
    # Initialize file access utility
    file_access = SimpleFileAccessUtility()
    registry.register_component("file_access_utility", file_access)
    
    # Initialize command executor
    command_executor = SimpleCommandExecutor()
    registry.register_component("command_executor", command_executor)
    
    # Initialize intent classification system
    intent_system = IntentClassificationSystem(registry)
    registry.register_component("intent_classification_system", intent_system)
    
    # Initialize intent caching system
    intent_cache = IntentOptimizationSystem(
        cache_capacity=100,
        history_size=50,
        window_size=20
    )
    registry.register_component("intent_optimization_system", intent_cache)
    
    # Create a simple conversation logger
    class SimpleConversationLogger:
        def start_conversation(self, user_id):
            return {"success": True, "conversation_id": "demo-conversation"}
            
        def add_message(self, conversation_id, role, content):
            pass
            
        def end_conversation(self, conversation_id):
            return {"success": True}
    
    registry.register_component("conversation_logger", SimpleConversationLogger())
    
    # Create a simple planning engine for handling generic intents
    class SimplePlanningEngine:
        def generate_plan(self, query, context=None):
            return {
                "success": True,
                "goal": "Process user query",
                "steps": [
                    {"description": "Interpret user query", "status": "completed"},
                    {"description": "Generate response", "status": "completed"}
                ],
                "response": f"This is a simulated response to: '{query}'"
            }
    
    registry.register_component("planning_engine", SimplePlanningEngine())
    
    # Initialize conversational interface
    conversational_interface = ConversationalInterface(registry)
    registry.register_component("conversational_interface", conversational_interface)
    
    # Initialize conversational interface
    conversational_interface.initialize()
    
    logger.info("Intent system initialized")
    
    return registry

def run_demo():
    """Run a command-line demo of the intent classification system."""
    print("\n=== FixWurx Intent Classification Demo ===")
    print("This demo showcases the intent classification system with caching and agent fallback.")
    print("Type 'exit', 'quit', or 'q' to end the demo.\n")
    
    # Initialize the system
    registry = initialize_intent_system()
    
    # Get key components
    conversational_interface = registry.get_component("conversational_interface")
    intent_system = registry.get_component("intent_classification_system")
    state_manager = registry.get_component("state_manager")
    
    # Create a test file for the demo
    demo_file_path = "demo_file.txt"
    file_access = registry.get_component("file_access_utility")
    file_access.write_file(demo_file_path, "This is a test file for the demo.\nIt has multiple lines of content.\nYou can modify this file through the conversational interface.")
    
    # Create a simple console interface
    while True:
        try:
            # Get user input
            user_input = input("\n> ")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting demo...")
                break
                
            # Process through conversation interface
            if user_input.startswith("!intent "):
                # Direct intent processing demo
                query = user_input[8:]  # Remove "!intent " prefix
                
                # Get context
                context = {
                    "state": state_manager.get_state(),
                    "history": conversational_interface.history,
                    "current_context": state_manager.get_context()
                }
                
                # Classify intent
                intent = intent_system.classify_intent(query, context)
                
                # Determine required agents
                required_agents = intent_system.determine_required_agents(intent)
                
                # Display results
                print("\nIntent Classification Results:")
                print(f"Type: {intent.type}")
                print(f"Execution Path: {intent.execution_path}")
                print(f"Parameters: {intent.parameters}")
                print(f"Required Agents: {', '.join(required_agents) if required_agents else 'None'}")
                
                # Predict next intents
                predicted_intents = registry.get_component("intent_optimization_system").predict_next_intents(intent.type)
                if predicted_intents:
                    print(f"\nPredicted next intents: {', '.join(predicted_intents)}")
                
            elif user_input.startswith("!cache "):
                # Cache management commands
                action = user_input[7:]  # Remove "!cache " prefix
                
                optimization_system = registry.get_component("intent_optimization_system")
                
                if action == "stats":
                    # Get cache stats
                    stats = optimization_system.get_stats()
                    
                    print("\nIntent Classification Cache Statistics:")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                        
                elif action == "clear":
                    # Clear the cache
                    optimization_system.clear_cache()
                    print("\nIntent classification cache cleared")
                    
                else:
                    print(f"\nUnknown cache action: {action}")
                    
            elif user_input.startswith("!fail "):
                # Test agent failure handling
                agent = user_input[6:]  # Remove "!fail " prefix
                
                # Get context
                context = {
                    "state": state_manager.get_state(),
                    "history": conversational_interface.history,
                    "current_context": state_manager.get_context()
                }
                
                # Create a generic intent
                intent = Intent(
                    query="Test query",
                    type="test_intent",
                    execution_path="agent_collaboration",
                    parameters={},
                    required_agents=["planner", agent, "executor"]
                )
                
                # Handle agent failure
                updated_agents, fallbacks = intent_system.handle_agent_failure(intent, [agent])
                
                # Display results
                print(f"\nAgent Failure Handling Results:")
                print(f"Failed Agent: {agent}")
                print(f"Updated Agents: {', '.join(updated_agents)}")
                print(f"Fallback Mapping: {fallbacks}")
                
            else:
                # Regular conversation processing
                print("\nProcessing...")
                response = conversational_interface.process_input(user_input)
                
                # Display response
                print("\nResponse:")
                print(response)
                
        except KeyboardInterrupt:
            print("\nExiting demo...")
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nDemo completed.")

if __name__ == "__main__":
    run_demo()
