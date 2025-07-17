#!/usr/bin/env python3
"""
file_access.py
──────────────
Secure file access layer for FixWurx with permission checks.

Provides permission-controlled file operations:
- read_file: Read a file with FILE_READ permission check
- write_file: Write to a file with FILE_WRITE permission check
- delete_file: Delete a file with FILE_DELETE permission check
- list_directory: List directory contents with FILE_READ permission check

All operations require a valid session token and the appropriate permission.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union, BinaryIO, TextIO, Dict, Any

import access_control
from access_control import (
    Permission, 
    AuthenticationError, 
    AuthorizationError
)

# File access modes for different operations
class FileAccessMode:
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


class FileAccessError(Exception):
    """Exception raised for file access errors."""
    pass


def read_file(token: str, path: Union[str, Path], binary: bool = False) -> Union[str, bytes]:
    """
    Read a file with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        binary: Whether to read in binary mode
        
    Returns:
        File content as string or bytes
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If file operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.READ)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    try:
        if binary:
            return path_obj.read_bytes()
        else:
            return path_obj.read_text(encoding="utf-8")
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"FILE_READ_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to read file: {e}")


def write_file(token: str, path: Union[str, Path], content: Union[str, bytes], binary: bool = False) -> None:
    """
    Write to a file with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        content: Content to write
        binary: Whether to write in binary mode
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If file operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.WRITE)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    # Create parent directories if they don't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if binary:
            path_obj.write_bytes(content)
        else:
            path_obj.write_text(content, encoding="utf-8")
            
        # Log the success
        access_control.log_action(
            username=username,
            action=f"FILE_WRITE",
            target=str(path_obj),
            details=f"Size: {len(content)} bytes"
        )
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"FILE_WRITE_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to write file: {e}")


def append_to_file(token: str, path: Union[str, Path], content: Union[str, bytes], binary: bool = False) -> None:
    """
    Append to a file with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        content: Content to append
        binary: Whether to append in binary mode
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If file operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.WRITE)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    # Create parent directories if they don't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        mode = "ab" if binary else "a"
        with open(path_obj, mode) as f:
            if binary:
                f.write(content)
            else:
                f.write(content)
            
        # Log the success
        access_control.log_action(
            username=username,
            action=f"FILE_APPEND",
            target=str(path_obj),
            details=f"Size: {len(content)} bytes"
        )
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"FILE_APPEND_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to append to file: {e}")


def delete_file(token: str, path: Union[str, Path]) -> None:
    """
    Delete a file with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If file operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.DELETE)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    try:
        if path_obj.exists():
            path_obj.unlink()
            
            # Log the success
            access_control.log_action(
                username=username,
                action=f"FILE_DELETE",
                target=str(path_obj)
            )
        else:
            # Log the warning
            access_control.log_action(
                username=username,
                action=f"FILE_DELETE_WARNING",
                target=str(path_obj),
                details="File does not exist"
            )
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"FILE_DELETE_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to delete file: {e}")


def list_directory(token: str, path: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    List directory contents with permission check.
    
    Args:
        token: Session token
        path: Path to the directory
        recursive: Whether to list recursively
        
    Returns:
        List of paths
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If directory operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.READ)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    try:
        if not path_obj.exists():
            # Log the warning
            access_control.log_action(
                username=username,
                action=f"DIR_LIST_WARNING",
                target=str(path_obj),
                details="Directory does not exist"
            )
            return []
            
        if not path_obj.is_dir():
            # Log the warning
            access_control.log_action(
                username=username,
                action=f"DIR_LIST_WARNING",
                target=str(path_obj),
                details="Path is not a directory"
            )
            return []
            
        if recursive:
            # Recursively list all files and directories
            paths = list(path_obj.glob("**/*"))
        else:
            # List only top-level files and directories
            paths = list(path_obj.glob("*"))
            
        # Log the success
        access_control.log_action(
            username=username,
            action=f"DIR_LIST",
            target=str(path_obj),
            details=f"Found {len(paths)} items"
        )
        
        return paths
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"DIR_LIST_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to list directory: {e}")


def create_directory(token: str, path: Union[str, Path]) -> None:
    """
    Create a directory with permission check.
    
    Args:
        token: Session token
        path: Path to the directory
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If directory operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.WRITE)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Log the success
        access_control.log_action(
            username=username,
            action=f"DIR_CREATE",
            target=str(path_obj)
        )
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"DIR_CREATE_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to create directory: {e}")


def delete_directory(token: str, path: Union[str, Path], recursive: bool = False) -> None:
    """
    Delete a directory with permission check.
    
    Args:
        token: Session token
        path: Path to the directory
        recursive: Whether to delete recursively
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If directory operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.DELETE)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    try:
        if not path_obj.exists():
            # Log the warning
            access_control.log_action(
                username=username,
                action=f"DIR_DELETE_WARNING",
                target=str(path_obj),
                details="Directory does not exist"
            )
            return
            
        if not path_obj.is_dir():
            # Log the warning
            access_control.log_action(
                username=username,
                action=f"DIR_DELETE_WARNING",
                target=str(path_obj),
                details="Path is not a directory"
            )
            return
            
        if recursive:
            # Recursively delete directory and all contents
            shutil.rmtree(path_obj)
        else:
            # Delete empty directory
            path_obj.rmdir()
            
        # Log the success
        access_control.log_action(
            username=username,
            action=f"DIR_DELETE",
            target=str(path_obj),
            details=f"Recursive: {recursive}"
        )
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"DIR_DELETE_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to delete directory: {e}")


def file_exists(token: str, path: Union[str, Path]) -> bool:
    """
    Check if a file exists with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        
    Returns:
        True if the file exists, False otherwise
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.READ)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    exists = path_obj.exists()
    
    # Log the action
    access_control.log_action(
        username=username,
        action=f"FILE_EXISTS_CHECK",
        target=str(path_obj),
        details=f"Exists: {exists}"
    )
    
    return exists


def get_file_info(token: str, path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get file information with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        
    Returns:
        Dictionary with file information
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
        FileAccessError: If file operation fails
    """
    # Check permission
    username = access_control.check_file_permission(token, path, FileAccessMode.READ)
    
    # Convert path to Path object
    path_obj = Path(path)
    
    try:
        if not path_obj.exists():
            # Log the warning
            access_control.log_action(
                username=username,
                action=f"FILE_INFO_WARNING",
                target=str(path_obj),
                details="File does not exist"
            )
            return {"exists": False}
            
        # Get file stats
        stat = path_obj.stat()
        
        info = {
            "exists": True,
            "is_file": path_obj.is_file(),
            "is_dir": path_obj.is_dir(),
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "name": path_obj.name,
            "path": str(path_obj.resolve()),
            "extension": path_obj.suffix
        }
        
        # Log the success
        access_control.log_action(
            username=username,
            action=f"FILE_INFO",
            target=str(path_obj)
        )
        
        return info
    except IOError as e:
        # Log the error
        access_control.log_action(
            username=username,
            action=f"FILE_INFO_ERROR",
            target=str(path_obj),
            details=str(e)
        )
        raise FileAccessError(f"Failed to get file info: {e}")


def file_readable(token: str, path: Union[str, Path]) -> bool:
    """
    Check if a file is readable with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        
    Returns:
        True if the file is readable, False otherwise
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
    """
    # Check if the user has permission to read this file
    try:
        username = access_control.check_file_permission(token, path, FileAccessMode.READ)
    except (AuthenticationError, AuthorizationError):
        return False
    
    # Convert path to Path object
    path_obj = Path(path)
    
    # Check if the file exists and is readable
    readable = path_obj.exists() and os.access(path_obj, os.R_OK)
    
    # Log the action
    access_control.log_action(
        username=username,
        action=f"FILE_READABLE_CHECK",
        target=str(path_obj),
        details=f"Readable: {readable}"
    )
    
    return readable


def file_writable(token: str, path: Union[str, Path]) -> bool:
    """
    Check if a file is writable with permission check.
    
    Args:
        token: Session token
        path: Path to the file
        
    Returns:
        True if the file is writable, False otherwise
        
    Raises:
        AuthenticationError: If the token is invalid
        AuthorizationError: If the user doesn't have permission
    """
    # Check if the user has permission to write to this file
    try:
        username = access_control.check_file_permission(token, path, FileAccessMode.WRITE)
    except (AuthenticationError, AuthorizationError):
        return False
    
    # Convert path to Path object
    path_obj = Path(path)
    
    # Check if the file exists and is writable, or if the parent directory is writable
    writable = (path_obj.exists() and os.access(path_obj, os.W_OK)) or (
        not path_obj.exists() and path_obj.parent.exists() and os.access(path_obj.parent, os.W_OK)
    )
    
    # Log the action
    access_control.log_action(
        username=username,
        action=f"FILE_WRITABLE_CHECK",
        target=str(path_obj),
        details=f"Writable: {writable}"
    )
    
    return writable
