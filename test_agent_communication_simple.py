#!/usr/bin/env python3
"""
test_agent_communication_simple.py
──────────────────────────────────
Simplified test for agent-to-agent communication with conversation logging.

This script tests the agent-to-agent communication capabilities using the
conversation logger without relying on the full agent system implementation.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Import conversation logger
import agent_conversation_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentCommunicationTest")

def simulate_agent_communication(conversation_logger):
    """
    Simulate communication between agents using the conversation logger.
    
    Args:
        conversation_logger: The conversation logger instance
        
    Returns:
        List of session IDs
    """
    sessions = []
    
    # Simulate Meta and Planner communication
    print("\nSimulating Meta → Planner communication")
    session_id = conversation_logger.log_agent_to_agent(
        source_agent_id="meta",
        target_agent_id="planner",
        message={
            "action": "plan_generate",
            "bug_id": "bug-123",
            "priority": "high",
            "timeout": 300
        }
    )
    
    # Simulate response
    conversation_logger.log_agent_to_agent(
        source_agent_id="planner",
        target_agent_id="meta",
        message={
            "action": "plan_generated",
            "plan_id": "plan-456",
            "steps": [
                {"id": 1, "action": "analyze_bug", "assigned_to": "observer"},
                {"id": 2, "action": "generate_fix", "assigned_to": "analyst"},
                {"id": 3, "action": "verify_fix", "assigned_to": "verifier"}
            ],
            "estimated_time": 120
        },
        session_id=session_id
    )
    
    print(f"Created Meta-Planner session: {session_id}")
    sessions.append(session_id)
    
    # Simulate Observer and Analyst communication
    print("\nSimulating Observer → Analyst communication")
    session_id = conversation_logger.log_agent_to_agent(
        source_agent_id="observer",
        target_agent_id="analyst",
        message={
            "action": "bug_analysis",
            "bug_id": "bug-123",
            "analysis": {
                "file": "calculator.py",
                "line": 42,
                "type": "division_by_zero",
                "severity": "high"
            }
        }
    )
    
    # Simulate response
    conversation_logger.log_agent_to_agent(
        source_agent_id="analyst",
        target_agent_id="observer",
        message={
            "action": "analysis_received",
            "status": "processing",
            "estimated_time": 30
        },
        session_id=session_id
    )
    
    # Add another message to the same session
    conversation_logger.log_agent_to_agent(
        source_agent_id="analyst",
        target_agent_id="observer",
        message={
            "action": "analysis_complete",
            "patch_id": "patch-789",
            "changes": [
                {"file": "calculator.py", "line": 42, "change": "Add zero check"},
                {"file": "calculator.py", "line": 43, "change": "Add error handling"}
            ]
        },
        session_id=session_id
    )
    
    print(f"Created Observer-Analyst session: {session_id}")
    sessions.append(session_id)
    
    # Simulate Analyst and Verifier communication
    print("\nSimulating Analyst → Verifier communication")
    session_id = conversation_logger.log_agent_to_agent(
        source_agent_id="analyst",
        target_agent_id="verifier",
        message={
            "action": "verify_patch",
            "patch_id": "patch-789",
            "test_cases": [
                {"input": "10/0", "expected": "Error"},
                {"input": "10/2", "expected": "5"}
            ]
        }
    )
    
    # Simulate response with LLM-generated content
    conversation_logger.log_agent_to_agent(
        source_agent_id="verifier",
        target_agent_id="analyst",
        message={
            "action": "verification_results",
            "patch_id": "patch-789",
            "passed": True,
            "test_results": [
                {"test": "10/0", "result": "Error", "passed": True},
                {"test": "10/2", "result": "5", "passed": True}
            ],
            "llm_analysis": "The patch successfully handles division by zero by adding a check and properly returns the expected result for valid inputs. The error handling is robust and follows best practices."
        },
        session_id=session_id
    )
    
    print(f"Created Analyst-Verifier session: {session_id}")
    sessions.append(session_id)
    
    # Simulate LLM interaction
    print("\nSimulating LLM interaction from Meta agent")
    llm_session_id = conversation_logger.log_llm_interaction(
        agent_id="meta",
        prompt="Analyze the system performance metrics and provide insights",
        response="The system is operating at normal capacity. CPU usage is at 45%, memory at 60%. No anomalies detected in the agent interaction patterns. Recommend continuing with standard operations."
    )
    
    print(f"Created LLM interaction session: {llm_session_id}")
    sessions.append(llm_session_id)
    
    return sessions

def display_session_data(conversation_logger, session_id):
    """
    Display the data for a specific session.
    
    Args:
        conversation_logger: The conversation logger instance
        session_id: The session ID to display
    """
    session_data = conversation_logger.get_session(session_id)
    
    if not session_data:
        print(f"No data found for session {session_id}")
        return
    
    print(f"\nSession: {session_id}")
    print(f"Message count: {len(session_data)}")
    print("-" * 60)
    
    for i, message in enumerate(session_data, 1):
        direction = message.get("direction", "")
        timestamp = time.strftime("%H:%M:%S", time.localtime(message.get("timestamp", 0)))
        
        if direction == "agent_to_agent":
            source = message.get("source_agent", "")
            target = message.get("target_agent", "")
            msg = message.get("message", {})
            action = msg.get("action", "")
            
            print(f"[{i}] {timestamp} - {source} → {target}: {action}")
            
            # Print important message content
            for key, value in msg.items():
                if key != "action" and not isinstance(value, (dict, list)):
                    print(f"    {key}: {value}")
                elif key == "steps" or key == "changes" or key == "test_results":
                    print(f"    {key}: {json.dumps(value)[:60]}...")
        
        elif direction == "agent_to_llm":
            agent = message.get("agent_id", "")
            prompt = message.get("prompt", "")[:50] + "..." if len(message.get("prompt", "")) > 50 else message.get("prompt", "")
            
            print(f"[{i}] {timestamp} - {agent} → LLM: {prompt}")
    
    print("-" * 60)

def analyze_conversations(conversation_logger, sessions):
    """
    Analyze the recorded conversations.
    
    Args:
        conversation_logger: The conversation logger instance
        sessions: List of session IDs to analyze
        
    Returns:
        Analysis results
    """
    print("\nAnalyzing conversations...")
    
    try:
        # Try to analyze the first session as an example
        if sessions:
            analysis = conversation_logger.analyze_conversation(sessions[0])
            print("\nSample Analysis Results:")
            print("-" * 60)
            
            for key, value in analysis.items():
                if key not in ["error", "session_id", "timestamp"]:
                    if isinstance(value, (dict, list)):
                        print(f"{key}: {json.dumps(value)[:100]}...")
                    else:
                        print(f"{key}: {value}")
    except Exception as e:
        print(f"Analysis error: {e}")
    
    return True

def main():
    """Main test function."""
    print("Testing Agent-to-Agent Communication")
    print("-" * 60)
    
    # Initialize conversation logger
    config = {
        "enabled": True,
        "storage_path": ".triangulum/test_agent_communications",
        "retention_days": 1  # Short retention for testing
    }
    
    # Clear previous test data if it exists
    test_path = Path(config["storage_path"])
    if test_path.exists():
        for file in test_path.glob("*"):
            try:
                file.unlink()
            except:
                pass
    
    # Get logger instance
    conversation_logger = agent_conversation_logger.get_instance(config)
    print(f"Initialized conversation logger with storage at: {conversation_logger.storage_path}")
    
    # Simulate agent communication
    sessions = simulate_agent_communication(conversation_logger)
    
    # Display detailed session data for a sample session
    if sessions:
        display_session_data(conversation_logger, sessions[0])
    
    # Analyze conversations
    analyze_conversations(conversation_logger, sessions)
    
    # Display storage structure
    print("\nStorage directory structure:")
    for file in sorted(conversation_logger.storage_path.glob("*")):
        print(f"  {file.name}")
    
    # Close sessions
    print("\nClosing sessions:")
    for session_id in sessions:
        conversation_logger.close_session(session_id)
        print(f"  Closed session: {session_id}")
    
    print("\nAgent-to-agent communication test completed successfully!")
    return 0

if __name__ == "__main__":
    main()
