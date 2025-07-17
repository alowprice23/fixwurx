#!/usr/bin/env python3
"""
Standardized Message Format for Agent Communication

This module defines the standardized message format for communication between agents.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional, List

class Message:
    """
    Represents a standardized message for agent communication, enhanced for the AICC.
    """
    def __init__(self,
                 sender: str,
                 recipient: str,
                 intent: str,
                 payload: Optional[Dict[str, Any]] = None,
                 message_id: Optional[str] = None,
                 timestamp: Optional[float] = None,
                 context: Optional[Dict[str, Any]] = None,
                 priority: int = 0,
                 requires_response: bool = True):
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.sender = sender
        self.recipient = recipient
        self.intent = intent
        self.payload = payload or {}
        self.context = context or {}
        self.priority = priority
        self.requires_response = requires_response

    def to_json(self) -> str:
        """
        Serializes the message to a JSON string.
        """
        return json.dumps({
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipient": self.recipient,
            "intent": self.intent,
            "payload": self.payload,
            "context": self.context,
            "priority": self.priority,
            "requires_response": self.requires_response
        })

    @staticmethod
    def from_json(json_str: str) -> 'Message':
        """
        Deserializes a JSON string to a Message object.
        """
        data = json.loads(json_str)
        return Message(
            sender=data.get("sender"),
            recipient=data.get("recipient"),
            intent=data.get("intent"),
            payload=data.get("payload"),
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp"),
            context=data.get("context"),
            priority=data.get("priority", 0),
            requires_response=data.get("requires_response", True)
        )
