"""
Mock registry for tests.

This module provides a simple mock registry implementation that can be used
in tests to avoid dependencies on the real registry.
"""

class MockRegistry:
    """Mock registry implementation for tests."""
    
    def __init__(self):
        """Initialize the mock registry."""
        self.components = {}
        self.config = {}
    
    def register_component(self, name, component):
        """
        Register a component in the registry.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
    
    def get_component(self, name):
        """
        Get a component from the registry.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None
        """
        return self.components.get(name)
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
