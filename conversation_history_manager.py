#!/usr/bin/env python3
"""
Conversation History Manager

This module provides functionality for managing conversation history, including
storing, retrieving, summarizing, and formatting conversation sessions between
users and agents.
"""

import os
import sys
import json
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("conversation_history.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ConversationHistoryManager")

class ConversationHistoryManager:
    """
    Manages conversation history for the FixWurx system.
    
    This class handles:
    1. Storing conversation messages (user inputs, agent responses)
    2. Managing session contexts for Meta Agent
    3. Summarizing long conversations to prevent context window overflow
    4. Persisting conversation history to disk
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the conversation history manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # History storage parameters
        self.max_history_items = self.config.get("max_history_items", 100)
        self.summarize_after = self.config.get("summarize_after", 50)
        self.storage_dir = self.config.get("storage_dir", "conversation_history")
        
        # Active sessions
        self.active_sessions = {}
        
        # Initialize storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        logger.info("Conversation History Manager initialized")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Initialize session
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
            "metadata": {
                "summary": None,
                "last_summarized_at": None,
                "message_count": 0
            }
        }
        
        logger.info(f"Created conversation session: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a message to a conversation session.
        
        Args:
            session_id: Session ID
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            The added message
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            session_id = self.create_session(session_id)
        
        # Create message
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "message_id": f"{session_id}_{len(self.active_sessions[session_id]['messages'])}",
            "metadata": metadata or {}
        }
        
        # Add to session
        self.active_sessions[session_id]["messages"].append(message)
        self.active_sessions[session_id]["updated_at"] = time.time()
        self.active_sessions[session_id]["metadata"]["message_count"] += 1
        
        # Check if summarization is needed
        if (len(self.active_sessions[session_id]["messages"]) > self.summarize_after and
            role in ["user", "assistant"]):
            self._mark_for_summarization(session_id)
        
        # Trim history if needed
        if len(self.active_sessions[session_id]["messages"]) > self.max_history_items:
            self._trim_history(session_id)
        
        return message
    
    def get_session_history(self, session_id: str, formatted: bool = False, 
                          include_system: bool = True) -> List[Dict[str, Any]]:
        """
        Get the history for a conversation session.
        
        Args:
            session_id: Session ID
            formatted: Whether to return formatted history (for LLM consumption)
            include_system: Whether to include system messages
            
        Returns:
            List of messages
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return []
        
        # Get messages
        messages = self.active_sessions[session_id]["messages"]
        
        # Filter system messages if requested
        if not include_system:
            messages = [msg for msg in messages if msg["role"] != "system"]
        
        # Format for LLM if requested
        if formatted:
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                formatted_messages.append(formatted_msg)
            return formatted_messages
        
        return messages
    
    def get_latest_messages(self, session_id: str, count: int = 5, 
                          formatted: bool = False) -> List[Dict[str, Any]]:
        """
        Get the latest messages from a conversation session.
        
        Args:
            session_id: Session ID
            count: Number of messages to return
            formatted: Whether to return formatted messages
            
        Returns:
            List of latest messages
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return []
        
        # Get latest messages
        messages = self.active_sessions[session_id]["messages"][-count:]
        
        # Format for LLM if requested
        if formatted:
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                formatted_messages.append(formatted_msg)
            return formatted_messages
        
        return messages
    
    def summarize_session(self, session_id: str, summary: Optional[str] = None,
                        keep_recent: int = 5) -> bool:
        """
        Summarize a conversation session to reduce context size.
        
        Args:
            session_id: Session ID
            summary: Optional pre-generated summary (if None, one should be generated)
            keep_recent: Number of recent messages to keep unchanged
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        # Get current messages
        messages = self.active_sessions[session_id]["messages"]
        
        # Nothing to summarize if not enough messages
        if len(messages) <= keep_recent + 1:
            return False
        
        # Get messages to summarize (all except recent ones and the first system message)
        first_system_message = None
        for msg in messages:
            if msg["role"] == "system":
                first_system_message = msg
                break
        
        # Get recent messages to keep
        recent_messages = messages[-keep_recent:] if keep_recent > 0 else []
        
        # Create summary message
        if not summary:
            # If no summary is provided, use a placeholder
            # In a real implementation, this would call the LLM to generate a summary
            summary = f"This conversation has {len(messages)} messages. The user and assistant discussed various topics."
        
        summary_message = {
            "role": "system",
            "content": f"Previous conversation summary: {summary}",
            "timestamp": time.time(),
            "message_id": f"{session_id}_summary_{int(time.time())}",
            "metadata": {
                "is_summary": True,
                "summarized_messages": len(messages) - keep_recent - (1 if first_system_message else 0)
            }
        }
        
        # Create new message list
        new_messages = []
        if first_system_message:
            new_messages.append(first_system_message)
        new_messages.append(summary_message)
        new_messages.extend(recent_messages)
        
        # Update session
        self.active_sessions[session_id]["messages"] = new_messages
        self.active_sessions[session_id]["metadata"]["last_summarized_at"] = time.time()
        
        logger.info(f"Summarized session {session_id}: Reduced from {len(messages)} to {len(new_messages)} messages")
        return True
    
    def save_session(self, session_id: str) -> bool:
        """
        Save a conversation session to disk.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        # Create filename
        timestamp = datetime.fromtimestamp(self.active_sessions[session_id]["created_at"]).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{session_id}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(self.active_sessions[session_id], f, indent=2)
            
            logger.info(f"Saved conversation session to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation session: {e}")
            return False
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a conversation session from disk.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check for files matching session ID
        for filename in os.listdir(self.storage_dir):
            if session_id in filename and filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        session_data = json.load(f)
                    
                    self.active_sessions[session_id] = session_data
                    logger.info(f"Loaded conversation session from {filepath}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading conversation session: {e}")
                    return False
        
        logger.warning(f"No saved session found for {session_id}")
        return False
    
    def close_session(self, session_id: str, save: bool = True) -> bool:
        """
        Close a conversation session.
        
        Args:
            session_id: Session ID
            save: Whether to save the session before closing
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        # Save session if requested
        if save:
            self.save_session(session_id)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Closed conversation session: {session_id}")
        return True
    
    def _mark_for_summarization(self, session_id: str) -> None:
        """
        Mark a session for summarization.
        
        Args:
            session_id: Session ID
        """
        # In a real implementation, this would add the session to a queue for summarization
        # For now, we just log it
        logger.info(f"Session {session_id} marked for summarization")
    
    def _trim_history(self, session_id: str) -> None:
        """
        Trim the history of a session to the maximum allowed items.
        
        Args:
            session_id: Session ID
        """
        messages = self.active_sessions[session_id]["messages"]
        
        # Keep track of important system messages
        first_system_message = None
        summary_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                if first_system_message is None:
                    first_system_message = msg
                elif msg.get("metadata", {}).get("is_summary", False):
                    summary_messages.append(msg)
        
        # Calculate how many messages to keep
        to_keep = self.max_history_items
        
        if len(messages) <= to_keep:
            return
        
        # Prioritize keeping:
        # 1. First system message
        # 2. Most recent summary message
        # 3. Most recent regular messages
        
        new_messages = []
        
        # Add first system message
        if first_system_message:
            new_messages.append(first_system_message)
            to_keep -= 1
        
        # Add most recent summary message
        if summary_messages:
            latest_summary = max(summary_messages, key=lambda x: x["timestamp"])
            new_messages.append(latest_summary)
            to_keep -= 1
        
        # Add most recent regular messages
        if to_keep > 0:
            new_messages.extend(messages[-to_keep:])
        
        # Update session
        self.active_sessions[session_id]["messages"] = new_messages
        
        logger.info(f"Trimmed session {session_id} history from {len(messages)} to {len(new_messages)} messages")


# Global instance
_instance = None

def get_instance(config: Optional[Dict[str, Any]] = None) -> ConversationHistoryManager:
    """
    Get the singleton instance of the Conversation History Manager.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ConversationHistoryManager instance
    """
    global _instance
    if _instance is None:
        _instance = ConversationHistoryManager(config)
    return _instance


# Register commands with shell environment
def register_conversation_commands(registry):
    """
    Register conversation history commands with the shell environment.
    
    Args:
        registry: Component registry
    """
    logger.info("Registering conversation history commands")
    
    # Initialize conversation history manager
    config = {
        "max_history_items": 100,
        "summarize_after": 50,
        "storage_dir": "conversation_history"
    }
    
    history_manager = get_instance(config)
    registry.register_component("conversation_history_manager", history_manager)
    
    # Register command handlers
    registry.register_command_handler("history:save", history_save_command, "conversation")
    registry.register_command_handler("history:list", history_list_command, "conversation")
    registry.register_command_handler("history:view", history_view_command, "conversation")
    
    logger.info("Conversation history commands registered")


def history_save_command(args: str) -> int:
    """
    Save the current conversation session.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Save the current conversation session")
    parser.add_argument("--session", help="Session ID (defaults to current)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get registry
    registry = sys.modules.get("__main__").registry
    
    # Get conversation interface
    conversation_interface = registry.get_component("conversational_interface")
    if not conversation_interface:
        print("Error: Conversational Interface not available")
        return 1
    
    # Get session ID
    session_id = cmd_args.session
    if not session_id:
        session_id = conversation_interface.current_session_id
    
    if not session_id:
        print("Error: No active conversation session")
        return 1
    
    # Get history manager
    history_manager = get_instance()
    
    # Save session
    if history_manager.save_session(session_id):
        print(f"Conversation session saved: {session_id}")
        return 0
    else:
        print(f"Error saving conversation session: {session_id}")
        return 1


def history_list_command(args: str) -> int:
    """
    List saved conversation sessions.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Get history manager
    history_manager = get_instance()
    
    # Get storage directory
    storage_dir = history_manager.storage_dir
    
    # List saved sessions
    if not os.path.exists(storage_dir):
        print(f"No saved sessions found (directory {storage_dir} does not exist)")
        return 0
    
    files = [f for f in os.listdir(storage_dir) if f.endswith('.json')]
    
    if not files:
        print(f"No saved sessions found in {storage_dir}")
        return 0
    
    print(f"\nSaved Conversation Sessions ({len(files)}):")
    print("-" * 60)
    
    for filename in sorted(files):
        filepath = os.path.join(storage_dir, filename)
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            session_id = session_data.get("session_id", "Unknown")
            created_at = datetime.fromtimestamp(session_data.get("created_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
            message_count = session_data.get("metadata", {}).get("message_count", 0)
            
            print(f"  {filename}")
            print(f"    ID: {session_id}")
            print(f"    Created: {created_at}")
            print(f"    Messages: {message_count}")
        except Exception as e:
            print(f"  {filename} (Error reading file: {e})")
    
    return 0


def history_view_command(args: str) -> int:
    """
    View a saved conversation session.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="View a saved conversation session")
    parser.add_argument("session_id", help="Session ID to view")
    parser.add_argument("--count", type=int, default=10, help="Number of messages to show")
    parser.add_argument("--all", action="store_true", help="Show all messages")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    session_id = cmd_args.session_id
    count = cmd_args.count
    show_all = cmd_args.all
    
    # Get history manager
    history_manager = get_instance()
    
    # Load session if not active
    if session_id not in history_manager.active_sessions:
        if not history_manager.load_session(session_id):
            print(f"Error: Session {session_id} not found")
            return 1
    
    # Get session messages
    messages = history_manager.get_session_history(session_id)
    
    if not messages:
        print(f"No messages found in session {session_id}")
        return 0
    
    # Display messages
    print(f"\nConversation Session: {session_id}")
    print("-" * 60)
    
    if show_all:
        display_messages = messages
    else:
        display_messages = messages[-count:]
    
    for i, msg in enumerate(display_messages, 1):
        role = msg["role"].capitalize()
        timestamp = datetime.fromtimestamp(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        content = msg["content"]
        
        print(f"\n[{timestamp}] {role}:")
        print(f"{content}")
    
    print(f"\nShowing {len(display_messages)} of {len(messages)} messages")
    
    return 0
