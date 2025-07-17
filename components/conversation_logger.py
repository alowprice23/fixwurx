"""
conversation_logger.py
──────────────────────
A stub for the Conversation Logger component.
"""

class ConversationLogger:
    """A placeholder for the conversation logger."""
    def __init__(self, config=None):
        self.config = config or {}
        self.initialized = False

    def initialize(self) -> bool:
        """Initializes the conversation logger."""
        self.initialized = True
        return True

    def start_conversation(self, user_id: str) -> dict:
        """Starts a new conversation."""
        return {"success": True, "conversation_id": "CONV-STUB-789"}

    def end_conversation(self, conversation_id: str) -> dict:
        """Ends a conversation."""
        return {"success": True}

    def add_message(self, conversation_id: str, role: str, content: str):
        """Adds a message to a conversation."""
        pass

    def shutdown(self):
        """Shuts down the conversation logger."""
        self.initialized = False

def get_instance(registry, config):
    """Returns an instance of the ConversationLogger."""
    return ConversationLogger(config)
