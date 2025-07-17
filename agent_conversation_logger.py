#!/usr/bin/env python3
"""
agent_conversation_logger.py
───────────────────────────
Conversation logging and storage system for agent interactions.

This module provides functionality to log and store all agent interactions,
including agent-user and agent-agent conversations, with the ability to
retrieve and analyze past conversations using LLM capabilities.
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentConversationLogger")

class ConversationLogger:
    """
    Conversation Logger for storing and analyzing agent interactions.
    
    This class provides methods to:
    1. Log agent-user and agent-agent conversations
    2. Store conversations in structured JSON format
    3. Retrieve past conversations for analysis
    4. Connect to auditor agent for deeper analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Conversation Logger.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.storage_path = Path(self.config.get("storage_path", ".triangulum/conversations"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Daily log file (for consolidated logging)
        self.daily_log_path = self.storage_path / f"conversations_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Sessions track multi-turn conversations
        self.active_sessions = {}
        
        # Thread lock for thread safety
        self._lock = threading.Lock()
        
        # Configure retention policy
        self.retention_days = self.config.get("retention_days", 30)
        
        # Register with auditor if available
        self.auditor = None
        self._register_with_auditor()
        
        logger.info("Conversation Logger initialized")
    
    def _register_with_auditor(self) -> None:
        """Register with the auditor agent if available."""
        try:
            # Try to import auditor module
            from agents.auditor.auditor_agent import get_instance as get_auditor
            self.auditor = get_auditor()
            logger.info("Successfully registered with Auditor Agent")
        except (ImportError, AttributeError):
            logger.warning("Auditor Agent not available, conversation analysis will be limited")
    
    def log_user_message(self, user_input: str, command: str, agent_id: str = None) -> str:
        """
        Log a message from a user to an agent.
        
        Args:
            user_input: The user's input message
            command: The command being executed
            agent_id: The target agent's ID (optional)
            
        Returns:
            Session ID for this conversation
        """
        if not self.enabled:
            return ""
            
        with self._lock:
            # Create a session ID if not provided
            session_id = self._get_or_create_session(agent_id)
            
            # Log the message
            message_data = {
                "timestamp": time.time(),
                "session_id": session_id,
                "direction": "user_to_agent",
                "user_input": user_input,
                "command": command,
                "target_agent": agent_id,
                "message_type": "command"
            }
            
            # Add to session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []
            self.active_sessions[session_id].append(message_data)
            
            # Save to storage
            self._save_message(message_data)
            
            # Log to auditor if available
            if self.auditor:
                try:
                    self.auditor.log_event("user_message", message_data)
                except Exception as e:
                    logger.error(f"Error logging to auditor: {e}")
            
            return session_id
    
    def log_agent_response(self, session_id: str, agent_id: str, response: Union[str, Dict], 
                           success: bool = True, llm_used: bool = False) -> None:
        """
        Log a response from an agent to a user.
        
        Args:
            session_id: The conversation session ID
            agent_id: The agent's ID
            response: The agent's response (text or structured data)
            success: Whether the response was successful
            llm_used: Whether LLM was used in generating the response
        """
        if not self.enabled:
            return
            
        with self._lock:
            # Format response
            if isinstance(response, dict):
                response_text = json.dumps(response)
            else:
                response_text = str(response)
            
            # Log the message
            message_data = {
                "timestamp": time.time(),
                "session_id": session_id,
                "direction": "agent_to_user",
                "agent_id": agent_id,
                "response": response_text,
                "success": success,
                "llm_used": llm_used,
                "message_type": "response"
            }
            
            # Add to session
            if session_id in self.active_sessions:
                self.active_sessions[session_id].append(message_data)
            
            # Save to storage
            self._save_message(message_data)
            
            # Log to auditor if available
            if self.auditor:
                try:
                    self.auditor.log_event("agent_response", message_data)
                except Exception as e:
                    logger.error(f"Error logging to auditor: {e}")
    
    def log_agent_to_agent(self, source_agent_id: str, target_agent_id: str, message: Dict[str, Any], 
                           session_id: Optional[str] = None) -> str:
        """
        Log a message from one agent to another.
        
        Args:
            source_agent_id: The source agent's ID
            target_agent_id: The target agent's ID
            message: The message being sent
            session_id: Optional session ID (will create if not provided)
            
        Returns:
            Session ID for this conversation
        """
        if not self.enabled:
            return ""
            
        with self._lock:
            # Create a session ID if not provided
            if not session_id:
                session_id = f"agent2agent_{int(time.time())}_{source_agent_id}_{target_agent_id}"
            
            # Log the message
            message_data = {
                "timestamp": time.time(),
                "session_id": session_id,
                "direction": "agent_to_agent",
                "source_agent": source_agent_id,
                "target_agent": target_agent_id,
                "message": message,
                "message_type": "inter_agent"
            }
            
            # Add to session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []
            self.active_sessions[session_id].append(message_data)
            
            # Save to storage
            self._save_message(message_data)
            
            # Log to auditor if available
            if self.auditor:
                try:
                    self.auditor.log_event("agent_to_agent", message_data)
                except Exception as e:
                    logger.error(f"Error logging to auditor: {e}")
            
            return session_id
    
    def log_llm_interaction(self, agent_id: str, prompt: str, response: str, 
                           session_id: Optional[str] = None) -> str:
        """
        Log an interaction between an agent and the LLM.
        
        Args:
            agent_id: The agent's ID
            prompt: The prompt sent to the LLM
            response: The response from the LLM
            session_id: Optional session ID (will create if not provided)
            
        Returns:
            Session ID for this conversation
        """
        if not self.enabled:
            return ""
            
        with self._lock:
            # Create a session ID if not provided
            if not session_id:
                session_id = f"llm_{int(time.time())}_{agent_id}"
            
            # Log the message
            message_data = {
                "timestamp": time.time(),
                "session_id": session_id,
                "direction": "agent_to_llm",
                "agent_id": agent_id,
                "prompt": prompt,
                "response": response,
                "message_type": "llm_interaction"
            }
            
            # Add to session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []
            self.active_sessions[session_id].append(message_data)
            
            # Save to storage
            self._save_message(message_data)
            
            # Log to auditor if available
            if self.auditor:
                try:
                    self.auditor.log_event("llm_interaction", message_data)
                except Exception as e:
                    logger.error(f"Error logging to auditor: {e}")
            
            return session_id
    
    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages in a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of message dictionaries in the session
        """
        # If session is active, return from memory
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Otherwise, load from storage
        return self._load_session(session_id)
    
    def close_session(self, session_id: str) -> None:
        """
        Close an active session.
        
        Args:
            session_id: The session ID to close
        """
        if session_id in self.active_sessions:
            # Perform analysis before closing if auditor is available
            if self.auditor:
                try:
                    session_data = self.active_sessions[session_id]
                    analysis_data = {
                        "session_id": session_id,
                        "message_count": len(session_data),
                        "session_data": session_data
                    }
                    self.auditor.analyze_with_llm("conversation", analysis_data)
                except Exception as e:
                    logger.error(f"Error analyzing session before closing: {e}")
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            logger.info(f"Closed session: {session_id}")
    
    def _get_or_create_session(self, agent_id: Optional[str] = None) -> str:
        """
        Get an existing session or create a new one.
        
        Args:
            agent_id: Optional agent ID to include in the session ID
            
        Returns:
            Session ID
        """
        # Create a new session ID
        agent_suffix = f"_{agent_id}" if agent_id else ""
        return f"session_{int(time.time())}{agent_suffix}"
    
    def _save_message(self, message_data: Dict[str, Any]) -> None:
        """
        Save a message to storage.
        
        Args:
            message_data: Message data dictionary
        """
        try:
            # Save to daily log file (append mode)
            with open(self.daily_log_path, 'a') as f:
                f.write(json.dumps(message_data) + '\n')
            
            # Save to session-specific file
            session_id = message_data.get("session_id")
            if session_id:
                session_path = self.storage_path / f"session_{session_id}.json"
                
                # Load existing session data if file exists
                session_data = []
                if session_path.exists():
                    try:
                        with open(session_path, 'r') as f:
                            session_data = json.load(f)
                    except:
                        # If file is corrupted, start fresh
                        session_data = []
                
                # Add message and save
                session_data.append(message_data)
                with open(session_path, 'w') as f:
                    json.dump(session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving message: {e}")
    
    def _load_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load a session from storage.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of message dictionaries in the session
        """
        try:
            session_path = self.storage_path / f"session_{session_id}.json"
            if session_path.exists():
                with open(session_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
        
        return []
    
    def clean_old_sessions(self) -> int:
        """
        Clean up old sessions based on retention policy.
        
        Returns:
            Number of sessions cleaned
        """
        if not self.enabled:
            return 0
            
        cleaned = 0
        try:
            # Calculate cutoff timestamp
            cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
            
            # Find and remove old session files
            for file_path in self.storage_path.glob("session_*.json"):
                try:
                    # Check file modification time
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned += 1
                except Exception:
                    pass
            
            # Find and remove old daily log files
            for file_path in self.storage_path.glob("conversations_*.jsonl"):
                try:
                    # Check file modification time
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned += 1
                except Exception:
                    pass
            
            logger.info(f"Cleaned {cleaned} old session files")
        except Exception as e:
            logger.error(f"Error cleaning old sessions: {e}")
        
        return cleaned
    
    def search_conversations(self, query: str, start_time: Optional[float] = None, 
                           end_time: Optional[float] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for conversations matching the query.
        
        Args:
            query: Search query string
            start_time: Optional start time timestamp
            end_time: Optional end time timestamp
            limit: Maximum number of results
            
        Returns:
            List of matching message dictionaries
        """
        if not self.enabled:
            return []
            
        results = []
        
        try:
            # Normalize query
            query = query.lower()
            
            # Set time bounds
            if not start_time:
                start_time = 0
            if not end_time:
                end_time = time.time()
            
            # Search through daily log files
            for file_path in sorted(self.storage_path.glob("conversations_*.jsonl"), reverse=True):
                # Stop if we've reached the limit
                if len(results) >= limit:
                    break
                
                # Process each line in the file
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            message = json.loads(line)
                            timestamp = message.get("timestamp", 0)
                            
                            # Check time bounds
                            if timestamp < start_time or timestamp > end_time:
                                continue
                            
                            # Check for query match
                            message_text = json.dumps(message).lower()
                            if query in message_text:
                                results.append(message)
                                
                                # Stop if we've reached the limit
                                if len(results) >= limit:
                                    break
                        except:
                            # Skip invalid lines
                            continue
            
            # Sort results by timestamp
            results.sort(key=lambda x: x.get("timestamp", 0))
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
        
        return results
    
    def analyze_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze a conversation session using LLM.
        
        Args:
            session_id: The session ID to analyze
            
        Returns:
            Analysis results dictionary
        """
        if not self.enabled or not self.auditor:
            return {"error": "Conversation analysis not available"}
            
        try:
            # Load the session
            session_data = self.get_session(session_id)
            
            if not session_data:
                return {"error": f"Session {session_id} not found or empty"}
            
            # Prepare analysis data
            analysis_data = {
                "session_id": session_id,
                "message_count": len(session_data),
                "session_data": session_data,
                "timestamp": time.time()
            }
            
            # Use auditor to analyze with LLM
            return self.auditor.analyze_with_llm("conversation", analysis_data)
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

# Create a singleton instance
_instance = None

def get_instance(config: Optional[Dict[str, Any]] = None) -> ConversationLogger:
    """
    Get the singleton instance of the Conversation Logger.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ConversationLogger instance
    """
    global _instance
    if _instance is None:
        _instance = ConversationLogger(config)
    return _instance
