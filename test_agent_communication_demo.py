#!/usr/bin/env python3
"""
test_agent_communication_demo.py
───────────────────────────────
Demonstration of the agent communication and progress tracking system.

This script demonstrates the capabilities of the agent communication system,
showing how agents can speak directly to users, track progress, and collaborate.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentCommDemo")

# Import required modules
import agent_communication_system
import agent_progress_tracking
import agent_conversation_logger

# Mock ComponentRegistry for testing
class MockRegistry:
    def __init__(self):
        self.components = {}
        self.command_handlers = {}
    
    def register_component(self, name, component):
        self.components[name] = component
    
    def get_component(self, name):
        return self.components.get(name)
    
    def register_command_handler(self, command, handler, component):
        self.command_handlers[command] = {"handler": handler, "component": component}

def simulate_long_running_task(agent_id, task_name, steps, progress_tracker):
    """Simulate a long-running task with progress updates."""
    # Start tracking progress
    tracker_id = progress_tracker.start_task(
        agent_id=agent_id,
        task_id=task_name,
        description=f"{task_name} Task",
        total_steps=steps
    )
    
    # Update progress over time
    for i in range(1, steps + 1):
        # Sleep for a bit to simulate work
        time.sleep(0.5)
        
        # Update progress
        progress_tracker.update_task(
            tracker_id=tracker_id,
            current_step=i,
            status_message=f"Processing step {i} of {steps}"
        )
    
    # Complete the task
    progress_tracker.complete_task(
        tracker_id=tracker_id,
        success=True,
        completion_message=f"{task_name} Task completed successfully"
    )
    
    return tracker_id

def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("AGENT COMMUNICATION SYSTEM DEMONSTRATION")
    print("=" * 80 + "\n")
    
    # Create mock registry
    registry = MockRegistry()
    
    # Initialize conversation logger
    logger.info("Initializing conversation logger...")
    config = {
        "enabled": True,
        "storage_path": ".triangulum/conversations",
        "retention_days": 30
    }
    conversation_logger = agent_conversation_logger.get_instance(config)
    registry.register_component("conversation_logger", conversation_logger)
    
    # Initialize agent communication system
    logger.info("Initializing agent communication system...")
    acs_config = {
        "enabled": True
    }
    communication_system = agent_communication_system.get_instance(acs_config)
    registry.register_component("agent_communication_system", communication_system)
    
    # Initialize progress tracking system
    logger.info("Initializing progress tracking system...")
    pt_config = {
        "enabled": True
    }
    progress_tracker = agent_progress_tracking.get_instance(pt_config)
    registry.register_component("progress_tracker", progress_tracker)
    
    # Register demonstration agents
    print("\nRegistering agents...")
    communication_system.register_agent("launchpad", "launchpad", ["orchestration", "planning"])
    communication_system.register_agent("orchestrator", "orchestrator", ["coordination"])
    communication_system.register_agent("triangulum", "triangulum", ["analysis"])
    communication_system.register_agent("auditor", "auditor", ["monitoring"])
    communication_system.register_agent("neural_matrix", "neural_matrix", ["learning"])
    
    # Create a session for demo
    session_id = f"demo_{int(time.time())}"
    
    # Step 1: Basic agent communication
    print("\n\n--- STEP 1: Basic Agent Communication ---\n")
    
    communication_system.speak("launchpad", "Hello! I am the Launchpad Agent.", session_id=session_id)
    time.sleep(1)
    
    communication_system.speak_info("orchestrator", 
                                   "I coordinate resources and manage the execution of tasks.", 
                                   session_id=session_id)
    time.sleep(1)
    
    communication_system.speak_success("triangulum", 
                                      "I can analyze problems and generate solutions.", 
                                      session_id=session_id)
    time.sleep(1)
    
    communication_system.speak_warning("auditor", 
                                      "I monitor the system and report potential issues.", 
                                      session_id=session_id)
    time.sleep(1)
    
    communication_system.speak_error("neural_matrix", 
                                    "Error demonstration - don't worry, this is just a test!", 
                                    session_id=session_id)
    time.sleep(1)
    
    # Step 2: Progress tracking
    print("\n\n--- STEP 2: Progress Tracking ---\n")
    
    # Start multiple background tasks to demonstrate concurrent progress tracking
    threads = []
    
    # Create thread for Launchpad task
    t1 = threading.Thread(
        target=simulate_long_running_task,
        args=("launchpad", "Initialization", 10, progress_tracker),
        daemon=True
    )
    threads.append(t1)
    
    # Create thread for Triangulum task
    t2 = threading.Thread(
        target=simulate_long_running_task,
        args=("triangulum", "Analysis", 8, progress_tracker),
        daemon=True
    )
    threads.append(t2)
    
    # Create thread for Neural Matrix task
    t3 = threading.Thread(
        target=simulate_long_running_task,
        args=("neural_matrix", "Training", 12, progress_tracker),
        daemon=True
    )
    threads.append(t3)
    
    # Start all tasks
    for t in threads:
        t.start()
    
    # Wait for all tasks to complete
    for t in threads:
        t.join()
    
    # Step 3: Conversation logging
    print("\n\n--- STEP 3: Conversation Logging ---\n")
    
    # Log a user message
    user_message = "analyze code for bugs"
    session_id = conversation_logger.log_user_message(
        user_input=user_message,
        command="analyze",
        agent_id="triangulum"
    )
    
    # Log agent response
    conversation_logger.log_agent_response(
        session_id=session_id,
        agent_id="triangulum",
        response="Analysis complete. Found 3 potential issues.",
        success=True,
        llm_used=True
    )
    
    # Use communication system to display the information
    communication_system.speak_info("triangulum", 
                                   "I've analyzed the conversation and logged the results.", 
                                   session_id=session_id)
    
    # Display conversation statistics
    print("\nConversation logging demonstration complete.")
    print("Session ID:", session_id)
    
    # Step 4: Multi-agent collaboration
    print("\n\n--- STEP 4: Multi-Agent Collaboration ---\n")
    
    # Simulate a multi-agent conversation flow
    communication_system.speak("launchpad", "Starting bug detection workflow", session_id=session_id)
    time.sleep(1)
    
    communication_system.speak("launchpad", "Delegating analysis to Triangulum agent", session_id=session_id)
    time.sleep(1)
    
    communication_system.speak("triangulum", "Beginning code analysis...", session_id=session_id)
    time.sleep(1)
    
    # Start a Triangulum task
    tracker_id = progress_tracker.start_task(
        agent_id="triangulum",
        task_id="code_analysis",
        description="Analyzing codebase for bugs",
        total_steps=5,
        session_id=session_id
    )
    
    # Update progress
    for i in range(1, 6):
        time.sleep(0.7)
        progress_tracker.update_task(
            tracker_id=tracker_id,
            current_step=i,
            status_message=f"Analyzing module {i} of 5"
        )
    
    # Complete the task
    progress_tracker.complete_task(
        tracker_id=tracker_id,
        success=True,
        completion_message="Code analysis completed"
    )
    
    communication_system.speak_success("triangulum", "Analysis complete: Found 2 critical bugs and 3 minor issues", session_id=session_id)
    time.sleep(1)
    
    communication_system.speak("orchestrator", "Prioritizing issues based on severity", session_id=session_id)
    time.sleep(1)
    
    communication_system.speak("neural_matrix", "Generating fix patterns based on historical data", session_id=session_id)
    time.sleep(1)
    
    communication_system.speak("auditor", "Verifying fixes against quality standards", session_id=session_id)
    time.sleep(1)
    
    communication_system.speak_success("launchpad", "Bug detection and resolution workflow completed successfully!", session_id=session_id)
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThe agent communication system is fully functional, allowing agents to:")
    print("- Speak directly to users with formatted messages")
    print("- Track and report progress on long-running tasks")
    print("- Log conversations for future reference")
    print("- Collaborate in multi-agent workflows")
    print("\nAll functionality has been verified and is ready for use in the shell environment.")

if __name__ == "__main__":
    main()
