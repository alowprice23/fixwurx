#!/usr/bin/env python3
"""
conversation_commands.py
───────────────────────
Shell commands for the agent conversation logging system.

This module provides commands to interact with the agent conversation logging system,
including viewing, searching, and analyzing agent interactions.
"""

import os
import json
import time
import logging
import argparse
import shlex
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the conversation logger
import agent_conversation_logger

# Configure logging
logger = logging.getLogger("ConversationCommands")

def register_conversation_commands(registry) -> None:
    """
    Register conversation commands with the shell environment.
    
    Args:
        registry: Component registry
    """
    logger.info("Registering conversation commands")
    
    # Initialize conversation logger
    config = {
        "enabled": True,
        "storage_path": ".triangulum/conversations",
        "retention_days": 30
    }
    
    conversation_logger = agent_conversation_logger.get_instance(config)
    registry.register_component("conversation_logger", conversation_logger)
    
    # Register conversation commands
    registry.register_command_handler("conversation:list", conversation_list_command, "conversation")
    registry.register_command_handler("conversation:show", conversation_show_command, "conversation")
    registry.register_command_handler("conversation:search", conversation_search_command, "conversation")
    registry.register_command_handler("conversation:analyze", conversation_analyze_command, "conversation")
    registry.register_command_handler("conversation:clean", conversation_clean_command, "conversation")
    
    # Register aliases
    registry.register_alias("convs", "conversation:list")
    registry.register_alias("conv", "conversation:show")
    
    logger.info("Conversation commands registered successfully")

def conversation_list_command(args: str) -> int:
    """
    List conversation sessions.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="List conversation sessions")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Number of sessions to show")
    parser.add_argument("--all", "-a", action="store_true", help="Show all sessions")
    parser.add_argument("--agent", "-g", help="Filter by agent ID")
    parser.add_argument("--type", "-t", choices=["user", "agent", "llm"], help="Filter by conversation type")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get conversation logger
    conversation_logger = agent_conversation_logger.get_instance()
    
    # Get session files
    sessions = []
    try:
        storage_path = conversation_logger.storage_path
        
        # Find session files
        for file_path in storage_path.glob("session_*.json"):
            try:
                # Read session file
                with open(file_path, 'r') as f:
                    session_data = json.load(f)
                
                # Skip empty sessions
                if not session_data:
                    continue
                
                # Get session info
                session_id = file_path.stem.replace("session_", "")
                first_message = session_data[0]
                last_message = session_data[-1]
                start_time = first_message.get("timestamp", 0)
                end_time = last_message.get("timestamp", 0)
                
                # Determine session type
                direction = first_message.get("direction", "")
                if "user_to_agent" in direction:
                    session_type = "user"
                elif "agent_to_agent" in direction:
                    session_type = "agent"
                elif "agent_to_llm" in direction:
                    session_type = "llm"
                else:
                    session_type = "unknown"
                
                # Get agent info
                agent_id = None
                if "target_agent" in first_message:
                    agent_id = first_message["target_agent"]
                elif "agent_id" in first_message:
                    agent_id = first_message["agent_id"]
                elif "source_agent" in first_message:
                    agent_id = first_message["source_agent"]
                
                # Add session to list
                sessions.append({
                    "session_id": session_id,
                    "type": session_type,
                    "agent_id": agent_id,
                    "message_count": len(session_data),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time if end_time and start_time else 0
                })
            except Exception as e:
                logger.error(f"Error processing session file {file_path}: {e}")
        
        # Apply filters
        if cmd_args.agent:
            sessions = [s for s in sessions if s.get("agent_id") == cmd_args.agent]
        
        if cmd_args.type:
            sessions = [s for s in sessions if s.get("type") == cmd_args.type]
        
        # Sort sessions by start time (newest first)
        sessions.sort(key=lambda x: x.get("start_time", 0), reverse=True)
        
        # Apply limit
        if not cmd_args.all:
            sessions = sessions[:cmd_args.limit]
        
        # Print sessions
        print("\nConversation Sessions:")
        print("-" * 80)
        print(f"{'Session ID':<20} {'Type':<10} {'Agent':<15} {'Messages':<10} {'Start Time':<20} {'Duration':<10}")
        print("-" * 80)
        
        for session in sessions:
            session_id = session.get("session_id", "")
            session_type = session.get("type", "")
            agent_id = session.get("agent_id", "")
            message_count = session.get("message_count", 0)
            start_time = datetime.fromtimestamp(session.get("start_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
            duration = f"{session.get('duration', 0):.1f}s"
            
            print(f"{session_id:<20} {session_type:<10} {agent_id or 'N/A':<15} {message_count:<10} {start_time:<20} {duration:<10}")
        
        if not sessions:
            print("No conversation sessions found")
        
        return 0
    except Exception as e:
        print(f"Error listing conversations: {e}")
        return 1

def conversation_show_command(args: str) -> int:
    """
    Show a conversation session.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Show a conversation session")
    parser.add_argument("session_id", help="Session ID to show")
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")
    parser.add_argument("--full", "-f", action="store_true", help="Show full messages (including prompt and response)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get conversation logger
    conversation_logger = agent_conversation_logger.get_instance()
    
    try:
        # Get session
        session_id = cmd_args.session_id
        session_data = conversation_logger.get_session(session_id)
        
        if not session_data:
            print(f"Session {session_id} not found or empty")
            return 1
        
        # Output as JSON if requested
        if cmd_args.json:
            print(json.dumps(session_data, indent=2))
            return 0
        
        # Print session
        print(f"\nConversation Session: {session_id}")
        print(f"Messages: {len(session_data)}")
        print("-" * 80)
        
        for i, message in enumerate(session_data, 1):
            timestamp = datetime.fromtimestamp(message.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
            direction = message.get("direction", "")
            message_type = message.get("message_type", "")
            
            print(f"\n[{i}] {timestamp} - {direction} ({message_type})")
            
            if direction == "user_to_agent":
                print(f"User Input: {message.get('user_input', '')}")
                print(f"Command: {message.get('command', '')}")
                print(f"Target Agent: {message.get('target_agent', '')}")
            
            elif direction == "agent_to_user":
                print(f"Agent: {message.get('agent_id', '')}")
                print(f"Success: {message.get('success', False)}")
                print(f"LLM Used: {message.get('llm_used', False)}")
                
                # Print response (truncate if not full)
                response = message.get("response", "")
                if not cmd_args.full and len(response) > 200:
                    print(f"Response: {response[:200]}... (truncated)")
                else:
                    print(f"Response: {response}")
            
            elif direction == "agent_to_agent":
                print(f"Source Agent: {message.get('source_agent', '')}")
                print(f"Target Agent: {message.get('target_agent', '')}")
                
                # Print message (truncate if not full)
                msg = json.dumps(message.get("message", {}))
                if not cmd_args.full and len(msg) > 200:
                    print(f"Message: {msg[:200]}... (truncated)")
                else:
                    print(f"Message: {msg}")
            
            elif direction == "agent_to_llm":
                print(f"Agent: {message.get('agent_id', '')}")
                
                # Print prompt and response (truncate if not full)
                prompt = message.get("prompt", "")
                response = message.get("response", "")
                
                if cmd_args.full:
                    print(f"Prompt: {prompt}")
                    print(f"Response: {response}")
                else:
                    if len(prompt) > 200:
                        print(f"Prompt: {prompt[:200]}... (truncated)")
                    else:
                        print(f"Prompt: {prompt}")
                    
                    if len(response) > 200:
                        print(f"Response: {response[:200]}... (truncated)")
                    else:
                        print(f"Response: {response}")
        
        return 0
    except Exception as e:
        print(f"Error showing conversation: {e}")
        return 1

def conversation_search_command(args: str) -> int:
    """
    Search for conversations.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Search for conversations")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Number of results to show")
    parser.add_argument("--start-time", "-s", help="Start time (format: YYYY-MM-DD [HH:MM:SS])")
    parser.add_argument("--end-time", "-e", help="End time (format: YYYY-MM-DD [HH:MM:SS])")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get conversation logger
    conversation_logger = agent_conversation_logger.get_instance()
    
    try:
        # Parse time ranges
        start_time = None
        end_time = None
        
        if cmd_args.start_time:
            try:
                # Try full datetime format
                start_time = datetime.strptime(cmd_args.start_time, "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                # Try date-only format
                start_time = datetime.strptime(cmd_args.start_time, "%Y-%m-%d").timestamp()
        
        if cmd_args.end_time:
            try:
                # Try full datetime format
                end_time = datetime.strptime(cmd_args.end_time, "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                # Try date-only format
                end_time = datetime.strptime(cmd_args.end_time, "%Y-%m-%d").timestamp() + 86400  # Add one day
        
        # Perform search
        results = conversation_logger.search_conversations(
            cmd_args.query,
            start_time=start_time,
            end_time=end_time,
            limit=cmd_args.limit
        )
        
        # Print results
        print(f"\nSearch Results for '{cmd_args.query}':")
        print(f"Found {len(results)} matching messages")
        print("-" * 80)
        
        for i, message in enumerate(results, 1):
            timestamp = datetime.fromtimestamp(message.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
            session_id = message.get("session_id", "")
            direction = message.get("direction", "")
            message_type = message.get("message_type", "")
            
            print(f"\n[{i}] {timestamp} - Session: {session_id}")
            print(f"Direction: {direction}, Type: {message_type}")
            
            if direction == "user_to_agent":
                print(f"User Input: {message.get('user_input', '')}")
                print(f"Command: {message.get('command', '')}")
            
            elif direction == "agent_to_user":
                print(f"Agent: {message.get('agent_id', '')}")
                response = message.get("response", "")
                if len(response) > 200:
                    print(f"Response: {response[:200]}... (truncated)")
                else:
                    print(f"Response: {response}")
            
            elif direction == "agent_to_agent":
                print(f"Source: {message.get('source_agent', '')}, Target: {message.get('target_agent', '')}")
            
            elif direction == "agent_to_llm":
                print(f"Agent: {message.get('agent_id', '')}")
                prompt = message.get("prompt", "")
                if len(prompt) > 200:
                    print(f"Prompt: {prompt[:200]}... (truncated)")
                else:
                    print(f"Prompt: {prompt}")
        
        if not results:
            print("No matching conversations found")
        
        return 0
    except Exception as e:
        print(f"Error searching conversations: {e}")
        return 1

def conversation_analyze_command(args: str) -> int:
    """
    Analyze a conversation session with LLM.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze a conversation session")
    parser.add_argument("session_id", help="Session ID to analyze")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get conversation logger
    conversation_logger = agent_conversation_logger.get_instance()
    
    try:
        # Analyze session
        session_id = cmd_args.session_id
        result = conversation_logger.analyze_conversation(session_id)
        
        if "error" in result:
            print(f"Error analyzing conversation: {result['error']}")
            return 1
        
        # Print analysis results
        print(f"\nAnalysis Results for Session: {session_id}")
        print("-" * 80)
        
        for key, value in result.items():
            if key == "session_id" or key == "timestamp":
                continue
                
            if isinstance(value, dict):
                print(f"\n{key.replace('_', ' ').title()}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey.replace('_', ' ').title()}: {subvalue}")
            elif isinstance(value, list):
                print(f"\n{key.replace('_', ' ').title()}:")
                for item in value:
                    if isinstance(item, dict):
                        for subkey, subvalue in item.items():
                            print(f"  {subkey.replace('_', ' ').title()}: {subvalue}")
                        print()
                    else:
                        print(f"  - {item}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        return 0
    except Exception as e:
        print(f"Error analyzing conversation: {e}")
        return 1

def conversation_clean_command(args: str) -> int:
    """
    Clean old conversation sessions.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Clean old conversation sessions")
    parser.add_argument("--days", "-d", type=int, help="Override retention days (default: from configuration)")
    parser.add_argument("--force", "-f", action="store_true", help="Force cleaning without confirmation")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get conversation logger
    conversation_logger = agent_conversation_logger.get_instance()
    
    try:
        # Override retention days if specified
        if cmd_args.days is not None:
            conversation_logger.retention_days = cmd_args.days
        
        # Confirm cleaning
        if not cmd_args.force:
            print(f"This will remove conversation sessions older than {conversation_logger.retention_days} days.")
            confirm = input("Are you sure you want to continue? (y/N): ")
            if confirm.lower() != 'y':
                print("Cleaning cancelled")
                return 0
        
        # Clean old sessions
        cleaned = conversation_logger.clean_old_sessions()
        print(f"Cleaned {cleaned} old session files")
        
        return 0
    except Exception as e:
        print(f"Error cleaning conversations: {e}")
        return 1
