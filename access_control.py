#!/usr/bin/env python3
"""
access_control.py
────────────────
Role-based access control system for FixWurx.

Provides:
- User authentication and management
- Role-based permission system
- Access control for CLI operations and file operations
- Audit logging for all system modifications

Implements the principle of least privilege, ensuring users only have access
to the specific operations they need to perform their roles.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import logging
import os
import secrets
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Constants
ACCESS_DIR = Path(".triangulum") / "access"
USER_DB_PATH = ACCESS_DIR / "users.json"
ROLES_PATH = ACCESS_DIR / "roles.json"
AUDIT_LOG_PATH = ACCESS_DIR / "audit.log"
SESSION_DIR = ACCESS_DIR / "sessions"
TOKEN_EXPIRY = 8 * 60 * 60  # 8 hours in seconds

# Ensure directories exist
ACCESS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(AUDIT_LOG_PATH),
        logging.StreamHandler()
    ]
)
audit_logger = logging.getLogger("access_control")


class Permission(Enum):
    """Permissions that can be granted to roles."""
    # System operations
    SYSTEM_START = auto()         # Can start the system (tri run)
    SYSTEM_STATUS = auto()        # Can view system status (tri status)
    
    # Queue operations
    QUEUE_VIEW = auto()           # Can view the queue (tri queue)
    QUEUE_APPROVE = auto()        # Can approve items in the queue
    QUEUE_REJECT = auto()         # Can reject items in the queue
    
    # Rollback operations
    ROLLBACK_EXECUTE = auto()     # Can execute rollbacks (tri rollback)
    
    # Dashboard operations
    DASHBOARD_VIEW = auto()       # Can view the dashboard (tri dashboard)
    
    # Plan operations
    PLAN_VIEW = auto()            # Can view plans (tri plan --list)
    PLAN_MODIFY = auto()          # Can modify plans
    
    # Agent operations
    AGENT_VIEW = auto()           # Can view agent status (tri agents --status)
    AGENT_CONTROL = auto()        # Can control agents (tri agents --restart)
    
    # Entropy operations
    ENTROPY_VIEW = auto()         # Can view entropy metrics (tri entropy)
    
    # File operations
    FILE_READ = auto()            # Can read files
    FILE_WRITE = auto()           # Can write files
    FILE_DELETE = auto()          # Can delete files
    
    # User management
    USER_VIEW = auto()            # Can view users
    USER_CREATE = auto()          # Can create users
    USER_MODIFY = auto()          # Can modify users
    USER_DELETE = auto()          # Can delete users
    
    # Role management
    ROLE_VIEW = auto()            # Can view roles
    ROLE_CREATE = auto()          # Can create roles
    ROLE_MODIFY = auto()          # Can modify roles
    ROLE_DELETE = auto()          # Can delete roles
    
    # Audit operations
    AUDIT_VIEW = auto()           # Can view audit logs


class AccessControlError(Exception):
    """Base exception for access control errors."""
    pass


class AuthenticationError(AccessControlError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(AccessControlError):
    """Raised when a user doesn't have permission to perform an action."""
    pass


class UserManagementError(AccessControlError):
    """Raised for user management errors."""
    pass


class RoleManagementError(AccessControlError):
    """Raised for role management errors."""
    pass


def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using a secure algorithm.
    
    Args:
        password: The password to hash
        salt: Optional salt to use, generates a new one if not provided
        
    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use a secure hashing algorithm (SHA-256) with a salt
    hash_obj = hashlib.sha256()
    hash_obj.update(salt.encode('utf-8'))
    hash_obj.update(password.encode('utf-8'))
    
    return hash_obj.hexdigest(), salt


def _generate_token() -> str:
    """
    Generate a secure random token for session authentication.
    
    Returns:
        A secure random token
    """
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')


def _load_users() -> Dict[str, Dict[str, Any]]:
    """
    Load user database from file.
    
    Returns:
        Dictionary of users with their details
    """
    if not USER_DB_PATH.exists():
        return {}
    
    try:
        with open(USER_DB_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        audit_logger.error(f"Failed to load user database: {e}")
        return {}


def _save_users(users: Dict[str, Dict[str, Any]]) -> None:
    """
    Save user database to file.
    
    Args:
        users: Dictionary of users with their details
    """
    try:
        with open(USER_DB_PATH, 'w') as f:
            json.dump(users, f, indent=4)
    except IOError as e:
        audit_logger.error(f"Failed to save user database: {e}")
        raise UserManagementError(f"Failed to save user database: {e}")


def _load_roles() -> Dict[str, Dict[str, Any]]:
    """
    Load roles from file.
    
    Returns:
        Dictionary of roles with their permissions
    """
    if not ROLES_PATH.exists():
        # Create default roles if the file doesn't exist
        default_roles = {
            "admin": {
                "description": "Administrator with full access",
                "permissions": [p.name for p in Permission]
            },
            "operator": {
                "description": "System operator with limited access",
                "permissions": [
                    Permission.SYSTEM_STATUS.name,
                    Permission.QUEUE_VIEW.name,
                    Permission.DASHBOARD_VIEW.name,
                    Permission.PLAN_VIEW.name,
                    Permission.AGENT_VIEW.name,
                    Permission.ENTROPY_VIEW.name,
                    Permission.FILE_READ.name
                ]
            },
            "viewer": {
                "description": "Read-only access to system status",
                "permissions": [
                    Permission.SYSTEM_STATUS.name,
                    Permission.DASHBOARD_VIEW.name,
                    Permission.ENTROPY_VIEW.name
                ]
            }
        }
        
        # Save default roles
        try:
            with open(ROLES_PATH, 'w') as f:
                json.dump(default_roles, f, indent=4)
        except IOError as e:
            audit_logger.error(f"Failed to create default roles: {e}")
            return default_roles
            
        return default_roles
    
    try:
        with open(ROLES_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        audit_logger.error(f"Failed to load roles: {e}")
        return {}


def _save_roles(roles: Dict[str, Dict[str, Any]]) -> None:
    """
    Save roles to file.
    
    Args:
        roles: Dictionary of roles with their permissions
    """
    try:
        with open(ROLES_PATH, 'w') as f:
            json.dump(roles, f, indent=4)
    except IOError as e:
        audit_logger.error(f"Failed to save roles: {e}")
        raise RoleManagementError(f"Failed to save roles: {e}")


def _save_session(username: str, token: str) -> None:
    """
    Save a session token for a user.
    
    Args:
        username: The username
        token: The session token
    """
    session_data = {
        "username": username,
        "created_at": time.time(),
        "expires_at": time.time() + TOKEN_EXPIRY
    }
    
    try:
        session_path = SESSION_DIR / f"{token}.json"
        with open(session_path, 'w') as f:
            json.dump(session_data, f)
    except IOError as e:
        audit_logger.error(f"Failed to save session: {e}")
        raise AuthenticationError(f"Failed to save session: {e}")


def _load_session(token: str) -> Optional[Dict[str, Any]]:
    """
    Load a session token.
    
    Args:
        token: The session token
        
    Returns:
        Session data if valid, None otherwise
    """
    session_path = SESSION_DIR / f"{token}.json"
    if not session_path.exists():
        return None
    
    try:
        with open(session_path, 'r') as f:
            session = json.load(f)
            
        # Check if session has expired
        if session.get("expires_at", 0) < time.time():
            # Delete expired session
            session_path.unlink(missing_ok=True)
            return None
            
        return session
    except (json.JSONDecodeError, IOError):
        return None


def _delete_session(token: str) -> None:
    """
    Delete a session token.
    
    Args:
        token: The session token
    """
    session_path = SESSION_DIR / f"{token}.json"
    session_path.unlink(missing_ok=True)


def _cleanup_sessions() -> None:
    """Clean up expired sessions."""
    now = time.time()
    
    for session_file in SESSION_DIR.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session = json.load(f)
                
            if session.get("expires_at", 0) < now:
                session_file.unlink(missing_ok=True)
        except (json.JSONDecodeError, IOError):
            # Delete invalid session files
            session_file.unlink(missing_ok=True)


def create_user(
    username: str, 
    password: str, 
    role: str = "viewer", 
    full_name: Optional[str] = None, 
    email: Optional[str] = None,
    created_by: Optional[str] = "system"
) -> None:
    """
    Create a new user.
    
    Args:
        username: The username (must be unique)
        password: The user's password
        role: The user's role (must exist)
        full_name: Optional full name
        email: Optional email address
        created_by: Username of the creator (for audit)
        
    Raises:
        UserManagementError: If the user already exists or role is invalid
    """
    users = _load_users()
    roles = _load_roles()
    
    # Check if username already exists
    if username in users:
        raise UserManagementError(f"User '{username}' already exists")
    
    # Check if role exists
    if role not in roles:
        raise UserManagementError(f"Role '{role}' does not exist")
    
    # Hash the password
    hashed_password, salt = _hash_password(password)
    
    # Create user
    users[username] = {
        "password_hash": hashed_password,
        "salt": salt,
        "role": role,
        "full_name": full_name,
        "email": email,
        "created_at": time.time(),
        "created_by": created_by,
        "last_login": None
    }
    
    # Save users
    _save_users(users)
    
    # Log the creation
    audit_logger.info(f"User '{username}' created with role '{role}' by '{created_by}'")


def delete_user(username: str, deleted_by: str) -> None:
    """
    Delete a user.
    
    Args:
        username: The username to delete
        deleted_by: Username of the deleter (for audit)
        
    Raises:
        UserManagementError: If the user doesn't exist
    """
    users = _load_users()
    
    # Check if user exists
    if username not in users:
        raise UserManagementError(f"User '{username}' does not exist")
    
    # Delete user
    del users[username]
    
    # Save users
    _save_users(users)
    
    # Log the deletion
    audit_logger.info(f"User '{username}' deleted by '{deleted_by}'")
    
    # Delete any active sessions for this user
    for session_file in SESSION_DIR.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session = json.load(f)
                
            if session.get("username") == username:
                session_file.unlink(missing_ok=True)
        except (json.JSONDecodeError, IOError):
            pass


def update_user(username: str, updates: Dict[str, Any], updated_by: str) -> None:
    """
    Update a user's details.
    
    Args:
        username: The username to update
        updates: Dictionary of attributes to update
        updated_by: Username of the updater (for audit)
        
    Raises:
        UserManagementError: If the user doesn't exist or updates are invalid
    """
    users = _load_users()
    roles = _load_roles()
    
    # Check if user exists
    if username not in users:
        raise UserManagementError(f"User '{username}' does not exist")
    
    user = users[username]
    
    # Handle password update
    if "password" in updates:
        hashed_password, salt = _hash_password(updates["password"])
        user["password_hash"] = hashed_password
        user["salt"] = salt
        updates.pop("password")
    
    # Handle role update
    if "role" in updates and updates["role"] not in roles:
        raise UserManagementError(f"Role '{updates['role']}' does not exist")
    
    # Update other fields
    for key, value in updates.items():
        if key not in ["password_hash", "salt", "created_at", "created_by"]:
            user[key] = value
    
    # Save users
    _save_users(users)
    
    # Log the update
    audit_logger.info(f"User '{username}' updated by '{updated_by}'")


def create_role(
    role_name: str, 
    description: str, 
    permissions: List[str],
    created_by: str
) -> None:
    """
    Create a new role.
    
    Args:
        role_name: The role name (must be unique)
        description: Description of the role
        permissions: List of permission names
        created_by: Username of the creator (for audit)
        
    Raises:
        RoleManagementError: If the role already exists or permissions are invalid
    """
    roles = _load_roles()
    
    # Check if role already exists
    if role_name in roles:
        raise RoleManagementError(f"Role '{role_name}' already exists")
    
    # Validate permissions
    valid_permissions = {p.name for p in Permission}
    for perm in permissions:
        if perm not in valid_permissions:
            raise RoleManagementError(f"Invalid permission: '{perm}'")
    
    # Create role
    roles[role_name] = {
        "description": description,
        "permissions": permissions,
        "created_at": time.time(),
        "created_by": created_by
    }
    
    # Save roles
    _save_roles(roles)
    
    # Log the creation
    audit_logger.info(f"Role '{role_name}' created by '{created_by}'")


def update_role(
    role_name: str, 
    updates: Dict[str, Any],
    updated_by: str
) -> None:
    """
    Update a role.
    
    Args:
        role_name: The role name to update
        updates: Dictionary of attributes to update
        updated_by: Username of the updater (for audit)
        
    Raises:
        RoleManagementError: If the role doesn't exist or updates are invalid
    """
    roles = _load_roles()
    
    # Check if role exists
    if role_name not in roles:
        raise RoleManagementError(f"Role '{role_name}' does not exist")
    
    role = roles[role_name]
    
    # Handle permissions update
    if "permissions" in updates:
        valid_permissions = {p.name for p in Permission}
        for perm in updates["permissions"]:
            if perm not in valid_permissions:
                raise RoleManagementError(f"Invalid permission: '{perm}'")
    
    # Update fields
    for key, value in updates.items():
        if key not in ["created_at", "created_by"]:
            role[key] = value
    
    # Save roles
    _save_roles(roles)
    
    # Log the update
    audit_logger.info(f"Role '{role_name}' updated by '{updated_by}'")


def delete_role(role_name: str, deleted_by: str) -> None:
    """
    Delete a role.
    
    Args:
        role_name: The role name to delete
        deleted_by: Username of the deleter (for audit)
        
    Raises:
        RoleManagementError: If the role doesn't exist or is in use
    """
    roles = _load_roles()
    users = _load_users()
    
    # Check if role exists
    if role_name not in roles:
        raise RoleManagementError(f"Role '{role_name}' does not exist")
    
    # Check if role is in use
    for username, user in users.items():
        if user.get("role") == role_name:
            raise RoleManagementError(f"Role '{role_name}' is in use by user '{username}'")
    
    # Delete role
    del roles[role_name]
    
    # Save roles
    _save_roles(roles)
    
    # Log the deletion
    audit_logger.info(f"Role '{role_name}' deleted by '{deleted_by}'")


def authenticate(username: str, password: str) -> str:
    """
    Authenticate a user and create a session.
    
    Args:
        username: The username
        password: The password
        
    Returns:
        Session token if authentication succeeds
        
    Raises:
        AuthenticationError: If authentication fails
    """
    users = _load_users()
    
    # Check if user exists
    if username not in users:
        audit_logger.warning(f"Authentication failed: User '{username}' does not exist")
        raise AuthenticationError("Invalid username or password")
    
    user = users[username]
    
    # Check password
    hashed_password, _ = _hash_password(password, user["salt"])
    if hashed_password != user["password_hash"]:
        audit_logger.warning(f"Authentication failed: Incorrect password for user '{username}'")
        raise AuthenticationError("Invalid username or password")
    
    # Create session token
    token = _generate_token()
    _save_session(username, token)
    
    # Update last login time
    user["last_login"] = time.time()
    _save_users(users)
    
    # Log the authentication
    audit_logger.info(f"User '{username}' authenticated successfully")
    
    # Clean up expired sessions
    _cleanup_sessions()
    
    return token


def validate_token(token: str) -> Optional[str]:
    """
    Validate a session token.
    
    Args:
        token: The session token
        
    Returns:
        Username if token is valid, None otherwise
    """
    session = _load_session(token)
    if session:
        return session.get("username")
    return None


def logout(token: str) -> None:
    """
    Log out a user by invalidating their session token.
    
    Args:
        token: The session token
    """
    session = _load_session(token)
    if session:
        username = session.get("username")
        _delete_session(token)
        audit_logger.info(f"User '{username}' logged out")


def check_permission(username: str, permission: Union[Permission, str]) -> bool:
    """
    Check if a user has a specific permission.
    
    Args:
        username: The username
        permission: The permission to check
        
    Returns:
        True if the user has the permission, False otherwise
    """
    users = _load_users()
    roles = _load_roles()
    
    # Check if user exists
    if username not in users:
        return False
    
    user = users[username]
    role_name = user.get("role")
    
    # Check if role exists
    if role_name not in roles:
        return False
    
    role = roles[role_name]
    
    # Convert permission to string if it's an enum
    if isinstance(permission, Permission):
        permission = permission.name
    
    # Check if the role has the permission
    return permission in role.get("permissions", [])


def has_permission(token: str, permission: Union[Permission, str]) -> bool:
    """
    Check if a session has a specific permission.
    
    Args:
        token: The session token
        permission: The permission to check
        
    Returns:
        True if the session has the permission, False otherwise
    """
    username = validate_token(token)
    if not username:
        return False
        
    return check_permission(username, permission)


def require_permission(token: str, permission: Union[Permission, str]) -> str:
    """
    Require a specific permission, raising an error if not present.
    
    Args:
        token: The session token
        permission: The required permission
        
    Returns:
        Username if the permission check passes
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have the required permission
    """
    username = validate_token(token)
    if not username:
        raise AuthenticationError("Invalid or expired session")
    
    if not check_permission(username, permission):
        # Convert permission to string if it's an enum
        if isinstance(permission, Permission):
            permission = permission.name
            
        audit_logger.warning(f"Authorization failed: User '{username}' attempted to use permission '{permission}'")
        raise AuthorizationError(f"You don't have permission to perform this action ({permission})")
    
    return username


def log_action(username: str, action: str, target: Optional[str] = None, details: Optional[str] = None) -> None:
    """
    Log an action for audit purposes.
    
    Args:
        username: The username performing the action
        action: The action being performed
        target: Optional target of the action (e.g., filename, user, etc.)
        details: Optional additional details
    """
    audit_logger.info(f"ACTION: {username} - {action}" + 
                     (f" - Target: {target}" if target else "") +
                     (f" - Details: {details}" if details else ""))


def check_file_permission(token: str, path: Union[str, Path], operation: str) -> str:
    """
    Check if a user has permission to perform a file operation.
    
    Args:
        token: The session token
        path: The file path
        operation: The operation (read, write, delete)
        
    Returns:
        Username if the permission check passes
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have the required permission
    """
    permission_map = {
        "read": Permission.FILE_READ,
        "write": Permission.FILE_WRITE,
        "delete": Permission.FILE_DELETE
    }
    
    if operation not in permission_map:
        raise ValueError(f"Invalid file operation: {operation}")
    
    username = require_permission(token, permission_map[operation])
    
    # Path is normalized and converted to string for logging
    path_str = str(Path(path).resolve())
    
    # Log the file access
    log_action(username, f"FILE_{operation.upper()}", path_str)
    
    return username


def create_initial_admin_user() -> None:
    """
    Create an initial admin user if no users exist.
    Uses environment variables for credentials.
    """
    users = _load_users()
    if not users:
        # Get admin credentials from environment or use defaults
        admin_username = os.environ.get("FIXWURX_ADMIN_USER", "admin")
        admin_password = os.environ.get("FIXWURX_ADMIN_PASSWORD")
        
        if not admin_password:
            # Generate a random password if not provided
            admin_password = secrets.token_urlsafe(12)
            print(f"\n[ACCESS CONTROL] Created initial admin user:")
            print(f"  Username: {admin_username}")
            print(f"  Password: {admin_password}")
            print("  IMPORTANT: Please change this password immediately!\n")
        
        try:
            create_user(
                username=admin_username,
                password=admin_password,
                role="admin",
                full_name="System Administrator",
                created_by="system"
            )
            audit_logger.info(f"Created initial admin user '{admin_username}'")
        except Exception as e:
            audit_logger.error(f"Failed to create initial admin user: {e}")


# Initialize access control system
def initialize():
    """Initialize the access control system."""
    # Ensure directories exist
    ACCESS_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load roles to ensure defaults are created
    _load_roles()
    
    # Create initial admin user if needed
    create_initial_admin_user()
    
    # Clean up expired sessions
    _cleanup_sessions()
    
    audit_logger.info("Access control system initialized")


# Automatic initialization when module is imported
initialize()
