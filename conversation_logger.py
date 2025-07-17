#!/usr/bin/env python3
"""
Conversation Logger

This module provides conversation storage, retrieval, and management.
"""

import os
import sys
import json
import time
import logging
import uuid
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger("ConversationLogger")

class ConversationLogger:
    """
    Conversation Logger for storing, managing, and retrieving conversations.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Conversation Logger.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configuration parameters
        self.logs_path = self.config.get("logs_path", "conversation_logs")
        self.max_conversation_size = self.config.get("max_conversation_size", 100)
        self.compression_enabled = self.config.get("compression_enabled", True)
        self.retention_days = self.config.get("retention_days", 90)
        
        # Active conversations
        self.active_conversations = {}
        
        # Register with registry
        registry.register_component("conversation_logger", self)
        
        logger.info("Conversation Logger initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the Conversation Logger.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Conversation Logger already initialized")
            return True
        
        try:
            # Create logs directory if it doesn't exist
            if not os.path.exists(self.logs_path):
                os.makedirs(self.logs_path)
            
            # Create conversation history directory
            history_path = os.path.join(self.logs_path, "history")
            if not os.path.exists(history_path):
                os.makedirs(history_path)
            
            # Clean up old conversations
            self._cleanup_old_conversations()
            
            self.initialized = True
            logger.info("Conversation Logger initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Conversation Logger: {e}")
            return False
    
    def _cleanup_old_conversations(self) -> None:
        """Clean up old conversations based on retention policy."""
        try:
            # Get current time
            now = time.time()
            
            # Calculate retention threshold
            retention_threshold = now - (self.retention_days * 24 * 60 * 60)
            
            # Get all conversation files
            history_path = os.path.join(self.logs_path, "history")
            conversation_files = []
            for root, _, files in os.walk(history_path):
                for file in files:
                    if file.endswith(".json"):
                        conversation_files.append(os.path.join(root, file))
            
            # Delete old conversations
            deleted_count = 0
            for file_path in conversation_files:
                try:
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    
                    # Delete if older than retention threshold
                    if mtime < retention_threshold:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old conversations")
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")
    
    def start_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Start a new conversation.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with conversation ID
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Conversation Logger initialization failed")
                return {"success": False, "error": "Conversation Logger initialization failed"}
        
        try:
            # Generate conversation ID
            conversation_id = str(uuid.uuid4())
            
            # Create conversation
            conversation = {
                "id": conversation_id,
                "user_id": user_id,
                "start_time": time.time(),
                "end_time": None,
                "messages": [],
                "summary": None
            }
            
            # Add to active conversations
            self.active_conversations[conversation_id] = conversation
            
            logger.info(f"Started conversation {conversation_id} for user {user_id}")
            
            return {
                "success": True,
                "conversation_id": conversation_id
            }
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            return {"success": False, "error": str(e)}
    
    def add_message(self, conversation_id: str, sender: str, content: str) -> Dict[str, Any]:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            sender: Message sender
            content: Message content
            
        Returns:
            Dictionary with result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Conversation Logger initialization failed")
                return {"success": False, "error": "Conversation Logger initialization failed"}
        
        try:
            # Check if conversation exists
            if conversation_id not in self.active_conversations:
                return {"success": False, "error": f"Conversation {conversation_id} not found"}
            
            # Add message
            message = {
                "id": str(uuid.uuid4()),
                "sender": sender,
                "content": content,
                "timestamp": time.time()
            }
            
            self.active_conversations[conversation_id]["messages"].append(message)
            
            # Check if conversation size exceeds limit
            if len(self.active_conversations[conversation_id]["messages"]) >= self.max_conversation_size:
                self.end_conversation(conversation_id)
            
            logger.info(f"Added message to conversation {conversation_id}")
            
            return {
                "success": True,
                "message_id": message["id"]
            }
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return {"success": False, "error": str(e)}
    
    def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        End a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary with result
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Conversation Logger initialization failed")
                return {"success": False, "error": "Conversation Logger initialization failed"}
        
        try:
            # Check if conversation exists
            if conversation_id not in self.active_conversations:
                return {"success": False, "error": f"Conversation {conversation_id} not found"}
            
            # End conversation
            conversation = self.active_conversations[conversation_id]
            conversation["end_time"] = time.time()
            
            # Generate summary
            conversation["summary"] = self._generate_summary(conversation)
            
            # Save conversation
            self._save_conversation(conversation)
            
            # Remove from active conversations
            del self.active_conversations[conversation_id]
            
            logger.info(f"Ended conversation {conversation_id}")
            
            return {
                "success": True
            }
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_summary(self, conversation: Dict[str, Any]) -> str:
        """
        Generate a summary of a conversation.
        
        Args:
            conversation: Conversation data
            
        Returns:
            Summary string
        """
        try:
            # Get neural matrix for summarization
            neural_matrix = self.registry.get_component("neural_matrix")
            if neural_matrix:
                # Create summary text
                messages_text = "\n".join([
                    f"{msg['sender']}: {msg['content']}"
                    for msg in conversation["messages"]
                ])
                
                # Generate summary
                summary = neural_matrix.summarize(messages_text)
                return summary
            else:
                # Fallback: simple summary
                return f"Conversation with {len(conversation['messages'])} messages"
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Conversation with {len(conversation['messages'])} messages"
    
    def _save_conversation(self, conversation: Dict[str, Any]) -> None:
        """
        Save a conversation to disk.
        
        Args:
            conversation: Conversation data
        """
        try:
            # Create conversation directory structure
            start_time = conversation["start_time"]
            date_str = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d")
            
            history_path = os.path.join(self.logs_path, "history", date_str)
            if not os.path.exists(history_path):
                os.makedirs(history_path)
            
            # Save conversation
            conversation_path = os.path.join(history_path, f"{conversation['id']}.json")
            with open(conversation_path, "w") as f:
                json.dump(conversation, f, indent=2)
            
            logger.info(f"Saved conversation {conversation['id']} to disk")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary with conversation data
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Conversation Logger initialization failed")
                return {"success": False, "error": "Conversation Logger initialization failed"}
        
        try:
            # Check if conversation is active
            if conversation_id in self.active_conversations:
                return {
                    "success": True,
                    "conversation": self.active_conversations[conversation_id]
                }
            
            # Search for conversation in history
            history_path = os.path.join(self.logs_path, "history")
            for root, _, files in os.walk(history_path):
                for file in files:
                    if file == f"{conversation_id}.json":
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            conversation = json.load(f)
                        return {
                            "success": True,
                            "conversation": conversation
                        }
            
            return {"success": False, "error": f"Conversation {conversation_id} not found"}
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return {"success": False, "error": str(e)}
    
    def get_conversations(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get conversations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of conversations to return
            
        Returns:
            Dictionary with conversation list
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Conversation Logger initialization failed")
                return {"success": False, "error": "Conversation Logger initialization failed"}
        
        try:
            # Collect conversations
            conversations = []
            
            # Add active conversations
            for conversation_id, conversation in self.active_conversations.items():
                if conversation["user_id"] == user_id:
                    conversations.append(conversation)
            
            # Add conversations from history
            history_path = os.path.join(self.logs_path, "history")
            for root, _, files in os.walk(history_path):
                for file in files:
                    if not file.endswith(".json"):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            conversation = json.load(f)
                        
                        if conversation["user_id"] == user_id:
                            conversations.append(conversation)
                    except Exception as e:
                        logger.error(f"Error loading conversation file {file_path}: {e}")
            
            # Sort by start time (most recent first)
            conversations.sort(key=lambda c: c["start_time"], reverse=True)
            
            # Limit number of conversations
            conversations = conversations[:limit]
            
            return {
                "success": True,
                "conversations": conversations,
                "count": len(conversations)
            }
        except Exception as e:
            logger.error(f"Error getting conversations: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self) -> None:
        """
        Shutdown the Conversation Logger.
        """
        if not self.initialized:
            return
        
        # End all active conversations
        for conversation_id in list(self.active_conversations.keys()):
            self.end_conversation(conversation_id)
        
        self.initialized = False
        logger.info("Conversation Logger shutdown complete")

# Singleton instance
_instance = None

def get_instance(registry, config: Optional[Dict[str, Any]] = None) -> ConversationLogger:
    """
    Get the singleton instance of the Conversation Logger.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        ConversationLogger instance
    """
    global _instance
    if _instance is None:
        _instance = ConversationLogger(registry, config)
    return _instance
