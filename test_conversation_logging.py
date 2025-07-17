#!/usr/bin/env python3
"""
test_conversation_logging.py
─────────────────────────
Test script for the conversation logging system.

This script tests the conversation logging system to ensure it properly
logs and stores agent interactions, including LLM calls.
"""

import os
import json
import time
from pathlib import Path
import agent_conversation_logger

def main():
    """Main test function."""
    print("Testing Conversation Logging System")
    print("-" * 60)
    
    # Initialize conversation logger
    config = {
        "enabled": True,
        "storage_path": ".triangulum/test_conversations",
        "retention_days": 30
    }
    
    # Clear previous test data if it exists
    test_path = Path(config["storage_path"])
    if test_path.exists():
        for file in test_path.glob("*"):
            file.unlink()
        test_path.rmdir()
    
    # Get logger instance
    logger = agent_conversation_logger.get_instance(config)
    print(f"Initialized conversation logger with storage at: {logger.storage_path}")
    
    # Test 1: Log user message
    print("\nTest 1: Logging user message to agent")
    session_id = logger.log_user_message(
        user_input="What's the status of the system?",
        command="agent:status",
        agent_id="agent"
    )
    print(f"Created session ID: {session_id}")
    
    # Test 2: Log agent response
    print("\nTest 2: Logging agent response")
    logger.log_agent_response(
        session_id=session_id,
        agent_id="agent",
        response={"status": "operational", "uptime": 3600, "agents": 5},
        success=True,
        llm_used=False
    )
    
    # Test 3: Log agent to agent communication
    print("\nTest 3: Logging agent to agent communication")
    agent_session_id = logger.log_agent_to_agent(
        source_agent_id="meta",
        target_agent_id="planner",
        message={"action": "plan_generate", "bug_id": "bug-123", "priority": "high"}
    )
    print(f"Created agent-to-agent session ID: {agent_session_id}")
    
    # Test 4: Log LLM interaction
    print("\nTest 4: Logging LLM interaction")
    llm_session_id = logger.log_llm_interaction(
        agent_id="meta",
        prompt="Generate a plan to fix bug-123 which is a division by zero error in calculate_average function",
        response="The bug can be fixed by adding a check for zero divisor before performing the division operation."
    )
    print(f"Created LLM session ID: {llm_session_id}")
    
    # Verify session data
    print("\nVerifying session data:")
    session_data = logger.get_session(session_id)
    print(f"User-agent session has {len(session_data)} messages")
    
    agent_session_data = logger.get_session(agent_session_id)
    print(f"Agent-agent session has {len(agent_session_data)} messages")
    
    llm_session_data = logger.get_session(llm_session_id)
    print(f"LLM session has {len(llm_session_data)} messages")
    
    # Test search
    print("\nTesting search:")
    results = logger.search_conversations("bug-123")
    print(f"Found {len(results)} messages containing 'bug-123'")
    
    # Display storage structure
    print("\nStorage directory structure:")
    for file in sorted(logger.storage_path.glob("*")):
        print(f"  {file.name}")
    
    # Display first session contents
    try:
        session_file = logger.storage_path / f"session_{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                data = json.load(f)
                print("\nSample session content:")
                print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error reading session file: {e}")
    
    # Clean up
    print("\nCleaning up (close the session):")
    logger.close_session(session_id)
    print("Session closed")
    
    print("\nConversation logging test completed successfully!")
    return 0

if __name__ == "__main__":
    main()
