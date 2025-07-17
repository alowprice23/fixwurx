"""
Mock implementation of flask_socketio for testing purposes.

This is a minimal implementation that provides the necessary classes and functions
used by web_interface.py to avoid import errors during testing.
"""

class SocketIO:
    """Mock SocketIO class."""
    
    def __init__(self, app=None, **kwargs):
        """
        Initialize the SocketIO instance.
        
        Args:
            app: Flask application instance
            **kwargs: Additional keyword arguments
        """
        self.app = app
        self.options = kwargs
        self.handlers = {}
    
    def on(self, event):
        """
        Decorator for registering event handlers.
        
        Args:
            event: Event name
            
        Returns:
            Decorator function
        """
        def decorator(f):
            self.handlers[event] = f
            return f
        return decorator
    
    def emit(self, event, data=None, room=None, include_self=True, namespace=None):
        """
        Emit an event to connected clients.
        
        Args:
            event: Event name
            data: Event data
            room: Room name
            include_self: Whether to include the sender
            namespace: Namespace
        """
        pass
    
    def run(self, app, host=None, port=None, **kwargs):
        """
        Run the SocketIO server.
        
        Args:
            app: Flask application instance
            host: Host to bind to
            port: Port to listen on
            **kwargs: Additional keyword arguments
        """
        pass


# Export commonly used functions and classes
def emit(event, data=None, room=None, include_self=True, namespace=None):
    """
    Emit an event to connected clients.
    
    Args:
        event: Event name
        data: Event data
        room: Room name
        include_self: Whether to include the sender
        namespace: Namespace
    """
    pass


def join_room(room, namespace=None):
    """
    Join a room.
    
    Args:
        room: Room name
        namespace: Namespace
    """
    pass


def leave_room(room, namespace=None):
    """
    Leave a room.
    
    Args:
        room: Room name
        namespace: Namespace
    """
    pass
