#!/usr/bin/env python3
"""
Demonstration script for the Conversational Interface with Intent Classification.
This script provides a simple command-line interface to interact with the 
FixWurx conversational system, showing how intent classification, agent fallback,
and caching work together.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Import the components we need
from components.conversational_interface import ConversationalInterface
from components.state_manager import StateManager
from components.intent_classification_system import IntentClassificationSystem
from components.intent_caching_system import IntentOptimizationSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intent_demo.log"), logging.StreamHandler()]
)
logger = logging.getLogger("IntentDemo")

class Registry:
    """Component registry for the demo."""
    
    def __init__(self):
        self.components = {}
    
    def register_component(self, name, component):
        self.components[name] = component
        
    def get_component(self, name):
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

def setup_environment():
    """Set up the demo environment."""
    registry = Registry()
    
    # Create and register components
    state_manager = StateManager(registry)
    registry.register_component("state_manager", state_manager)
    
    # File access utility
    file_access = SimpleFileAccessUtility()
    registry.register_component("file_access_utility", file_access)
    
    # Command executor
    command_executor = SimpleCommandExecutor()
    registry.register_component("command_executor", command_executor)
    
    # Create a simple conversation logger
    class SimpleConversationLogger:
        def start_conversation(self, user_id):
            return {"success": True, "conversation_id": "demo-conversation"}
            
        def add_message(self, conversation_id, role, content):
            pass
            
        def end_conversation(self, conversation_id):
            return {"success": True}
    
    registry.register_component("conversation_logger", SimpleConversationLogger())
    
    # Set up a planning engine placeholder
    class SimplePlanningEngine:
        def generate_plan(self, query):
            return {
                "success": True,
                "goal": f"Process query: {query}",
                "steps": [
                    {"description": "Step 1: Analyze query"},
                    {"description": "Step 2: Generate response"}
                ]
            }
    
    registry.register_component("planning_engine", SimplePlanningEngine())
    
    # Create the intent classification system
    intent_system = IntentClassificationSystem(registry)
    registry.register_component("intent_classification_system", intent_system)
    
    # Create the intent optimization system (caching)
    intent_opt_system = IntentOptimizationSystem(cache_capacity=10, history_size=20, window_size=10)
    registry.register_component("intent_optimization_system", intent_opt_system)
    
    # Create the conversational interface
    conversational_interface = ConversationalInterface(registry)
    
    return registry, conversational_interface

def run_demo():
    """Run the conversation demo."""
    registry, ci = setup_environment()
    
    print("\n=== FixWurx Conversational Interface Demo ===")
    print("This demo showcases the intent classification, agent fallback,")
    print("and caching systems working together.")
    print("Type 'exit' or 'quit' to end the demo.\n")
    
    # Create a test file for the demo
    demo_file_path = "demo_file.txt"
    file_access = registry.get_component("file_access_utility")
    file_access.write_file(demo_file_path, "This is a test file for the demo.\nIt has multiple lines of content.\nYou can modify this file through the conversational interface.")
    
    # Initialize the conversational interface
    ci.initialize()
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting demo...")
                break
                
            # Process the input through the conversational interface
            print("\nProcessing...")
            response = ci.process_input(user_input)
            
            # Display the response
            print("\nResponse:")
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting demo...")
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nDemo completed. Thank you for trying the FixWurx Conversational Interface!")

if __name__ == "__main__":
    run_demo()
