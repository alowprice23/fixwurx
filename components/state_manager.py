#!/usr/bin/env python3
"""
State Manager

This module provides a state manager for the FixWurx shell.
"""

import logging
import json
import time
import uuid
import os
from enum import Enum
from typing import Dict, Any, Optional, Callable

# Configure logger
logger = logging.getLogger("StateManager")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Event:
    """Event for the event system."""
    
    def __init__(self, event_type: str, data: Dict[str, Any] = None):
        """Initialize an event."""
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
    
    def to_dict(self):
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp
        }

class EventSystem:
    """Event system for the FixWurx shell."""
    
    def __init__(self):
        """Initialize the event system."""
        self.listeners = {}
        self.logger = logging.getLogger("EventSystem")
    
    def add_listener(self, event_type: str, listener: Callable):
        """
        Add a listener for an event type.
        
        Args:
            event_type: Type of event to listen for
            listener: Function to call when event occurs
        """
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)
        self.logger.info(f"Added listener for event type {event_type}")
    
    def emit_event(self, event: Event):
        """
        Emit an event.
        
        Args:
            event: Event to emit
        """
        event_type = event.event_type
        listeners = self.listeners.get(event_type, [])
        
        self.logger.info(f"Emitting event {event_type} with {len(listeners)} listeners")
        
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"Error in event listener: {e}")

class State(Enum):
    """
    Represents the state of the system.
    """
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_FEEDBACK = "awaiting_feedback"

class StateManager:
    """
    Manages the state of the FixWurx shell.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the State Manager.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.state = State.IDLE
        self.context = {}
        self.transition_hooks = {}
        self.event_system = EventSystem()
        
        # Register with registry
        registry.register_component("state_manager", self)
        
        logger.info("State Manager initialized")

    def get_state(self) -> State:
        """
        Get the current state.
        
        Returns:
            State: The current state
        """
        return self.state

    def set_state(self, state: State):
        """
        Set the current state.
        
        Args:
            state: The new state
        """
        old_state = self.state
        logger.info(f"State changed from {old_state.value} to {state.value}")
        self.state = state
        
        # Call transition hooks
        self._call_transition_hooks(old_state, state)
        
        # Emit state change event
        event = Event("state_change", {
            "old_state": old_state.value,
            "new_state": state.value,
            "context": self.context
        })
        self.event_system.emit_event(event)

    def register_transition_hook(self, from_state: State, to_state: State, callback: Callable):
        """
        Register a hook to be called when state transitions from one state to another.
        
        Args:
            from_state: Source state
            to_state: Target state
            callback: Function to call when transition occurs
        """
        key = (from_state, to_state)
        if key not in self.transition_hooks:
            self.transition_hooks[key] = []
        self.transition_hooks[key].append(callback)
        logger.info(f"Registered transition hook for {from_state.value} -> {to_state.value}")

    def _call_transition_hooks(self, from_state: State, to_state: State):
        """
        Call hooks for a state transition.
        
        Args:
            from_state: Source state
            to_state: Target state
        """
        key = (from_state, to_state)
        hooks = self.transition_hooks.get(key, [])
        
        for hook in hooks:
            try:
                hook(self.context)
            except Exception as e:
                logger.error(f"Error in transition hook: {e}")

    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context.
        
        Returns:
            Dict[str, Any]: The current context
        """
        return self.context

    def update_context(self, data: Dict[str, Any]):
        """
        Update the context with new data.
        
        Args:
            data: Data to update the context with
        """
        self.context.update(data)
        logger.info(f"Context updated: {data}")

    def reset(self):
        """
        Reset the state and context.
        """
        self.state = State.IDLE
        self.context = {}
        logger.info("State and context reset")

    def save_context(self, filename: str = None):
        """
        Save context to disk.
        
        Args:
            filename: Optional filename to save to
        """
        if not filename:
            timestamp = int(time.time())
            filename = f".triangulum/context_{timestamp}.json"
        
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, "w") as f:
                json.dump(self.context, f, indent=2, default=str)
            
            logger.info(f"Saved context to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving context: {e}")
            return False

    def load_context(self, filename: str):
        """
        Load context from disk.
        
        Args:
            filename: Filename to load from
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"Context file {filename} not found")
                return False
            
            with open(filename, "r") as f:
                loaded_context = json.load(f)
            
            self.context.update(loaded_context)
            logger.info(f"Loaded context from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            return False

# Singleton instance
_instance = None

def get_instance(registry, config: Optional[Dict[str, Any]] = None) -> StateManager:
    """
    Get the singleton instance of the State Manager.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        StateManager instance
    """
    global _instance
    if _instance is None:
        _instance = StateManager(registry, config)
    return _instance
