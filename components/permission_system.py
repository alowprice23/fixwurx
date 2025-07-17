#!/usr/bin/env python3
"""
Permission System

This module provides the PermissionSystem class for managing user permissions.
"""

from typing import Dict, Any, List

class PermissionSystem:
    """
    A placeholder for the permission system.
    """
    def __init__(self, registry, config: Dict[str, Any]):
        self.registry = registry
        self.config = config
        self.permissions = self._load_permissions()

    def _load_permissions(self) -> Dict[str, List[str]]:
        """
        Loads permissions from a configuration file.
        """
        # In a real implementation, this would load permissions from a database or a config file.
        return {
            "admin": ["*"],
            "user": ["read", "write", "execute"],
            "guest": ["read"]
        }

    def has_permission(self, user: str, permission: str) -> bool:
        """
        Checks if a user has a specific permission.
        """
        user_roles = self._get_user_roles(user)
        for role in user_roles:
            if "*" in self.permissions.get(role, []):
                return True
            if permission in self.permissions.get(role, []):
                return True
        return False

    def _get_user_roles(self, user: str) -> List[str]:
        """
        Gets the roles for a specific user.
        """
        # In a real implementation, this would get user roles from a database or an identity provider.
        if user == "admin":
            return ["admin"]
        elif user == "guest":
            return ["guest"]
        else:
            return ["user"]

def get_instance(registry, config):
    """Returns an instance of the PermissionSystem."""
    return PermissionSystem(registry, config)
