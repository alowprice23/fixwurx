"""
Mock implementation of eventlet for testing purposes.

This is a minimal implementation that provides the necessary functions and classes
used by web_interface.py to avoid import errors during testing.
"""

def monkey_patch():
    """Mock function to patch standard library modules."""
    pass

def listen(sock, backlog=None):
    """Mock function to listen on a socket."""
    pass

def wrap_ssl(sock, *args, **kwargs):
    """Mock function to wrap a socket in SSL."""
    pass

class GreenPool:
    """Mock GreenPool class."""
    
    def __init__(self, size=1000):
        """Initialize the GreenPool."""
        self.size = size
    
    def spawn(self, func, *args, **kwargs):
        """Spawn a new greenlet."""
        pass
    
    def waitall(self):
        """Wait for all greenlets to complete."""
        pass

class Timeout:
    """Mock Timeout class."""
    
    def __init__(self, seconds=None, *args, **kwargs):
        """Initialize the timeout."""
        self.seconds = seconds
    
    def __enter__(self):
        """Enter the timeout context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the timeout context."""
        pass
    
    def cancel(self):
        """Cancel the timeout."""
        pass

class GreenThread:
    """Mock GreenThread class."""
    
    def __init__(self):
        """Initialize the green thread."""
        pass
    
    def wait(self):
        """Wait for the thread to complete."""
        pass
    
    def kill(self):
        """Kill the thread."""
        pass

def spawn(func, *args, **kwargs):
    """Spawn a new greenlet."""
    return GreenThread()
