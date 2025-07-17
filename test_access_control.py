#!/usr/bin/env python3
"""
test_access_control.py
──────────────────────
Test script for the access control and file permission system.

This script demonstrates:
1. User authentication and management
2. Role management and permissions
3. File access control with different permission levels
4. Audit logging

Run this script to verify that the access control system is working correctly.
"""

import os
import time
from pathlib import Path

import access_control
from access_control import (
    Permission,
    AuthenticationError,
    AuthorizationError,
    UserManagementError,
    RoleManagementError
)

import file_access
from file_access import FileAccessError

# Test directory for file operations
TEST_DIR = Path(".triangulum/test")
TEST_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_test_result(name, success):
    """Print test result."""
    result = "✅ PASSED" if success else "❌ FAILED"
    print(f"{result} - {name}")


def ensure_test_admin():
    """Ensure a test admin user exists with a known password."""
    # Check if users exist
    users = access_control._load_users()
    
    # Create test admin if needed
    if "test_admin" not in users:
        try:
            # Load roles to ensure they exist
            roles = access_control._load_roles()
            
            # Create the admin user with a known password
            access_control.create_user(
                username="test_admin",
                password="test_admin_password",
                role="admin",
                full_name="Test Administrator",
                created_by="system"
            )
            print(f"Created test admin user 'test_admin' with password 'test_admin_password'")
        except Exception as e:
            print(f"Failed to create test admin: {e}")
    
    return "test_admin", "test_admin_password"


def cleanup_test_users():
    """Clean up test users and roles."""
    # Get admin credentials
    admin_user, admin_pass = ensure_test_admin()
    
    try:
        # Authenticate as admin
        admin_token = access_control.authenticate(admin_user, admin_pass)
        
        # Get users
        users = access_control._load_users()
        
        # Delete test users
        for username in list(users.keys()):
            if username.startswith("test_"):
                try:
                    access_control.delete_user(username, admin_user)
                    print(f"Cleaned up user: {username}")
                except Exception as e:
                    print(f"Failed to clean up user {username}: {e}")

        # Get roles
        roles = access_control._load_roles()
        
        # Delete test roles
        for role_name in list(roles.keys()):
            if role_name.startswith("test_"):
                try:
                    # Check if any users still have this role
                    has_users = False
                    for user in users.values():
                        if user.get("role") == role_name:
                            has_users = True
                            break
                    
                    if not has_users:
                        access_control.delete_role(role_name, admin_user)
                        print(f"Cleaned up role: {role_name}")
                except Exception as e:
                    print(f"Failed to clean up role {role_name}: {e}")
    
    except Exception as e:
        print(f"Failed to authenticate admin for cleanup: {e}")


def test_user_management():
    """Test user management functionality."""
    print_header("Testing User Management")
    
    # Test user creation
    admin_token = None
    try:
        # Get admin credentials
        admin_user, admin_pass = ensure_test_admin()
        
        # Authenticate as admin
        admin_token = access_control.authenticate(admin_user, admin_pass)
        print(f"Admin authenticated: {bool(admin_token)}")
        
        # Create a test user
        test_username = f"test_user_{int(time.time())}"
        access_control.create_user(
            username=test_username,
            password="testpassword",
            role="viewer",
            full_name="Test User",
            created_by=admin_user
        )
        print_test_result("User creation", True)
        
        # Verify the user exists
        users = access_control._load_users()
        user_exists = test_username in users
        print_test_result("User exists after creation", user_exists)
        
        if user_exists:
            # Authenticate as the test user
            test_token = access_control.authenticate(test_username, "testpassword")
            print_test_result("Test user authentication", bool(test_token))
            
            # Update the user
            access_control.update_user(
                username=test_username,
                updates={"full_name": "Updated Test User"},
                updated_by=admin_user
            )
            
            # Verify the update
            users = access_control._load_users()
            update_success = users[test_username]["full_name"] == "Updated Test User"
            print_test_result("User update", update_success)
            
            # Delete the user
            access_control.delete_user(test_username, admin_user)
            
            # Verify the user no longer exists
            users = access_control._load_users()
            delete_success = test_username not in users
            print_test_result("User deletion", delete_success)
            
    except Exception as e:
        print_test_result("User management operations", False)
        print(f"  Error: {e}")


def test_role_management():
    """Test role management functionality."""
    print_header("Testing Role Management")
    
    # Test role creation
    admin_token = None
    try:
        # Get admin credentials
        admin_user, admin_pass = ensure_test_admin()
        
        # Authenticate as admin
        admin_token = access_control.authenticate(admin_user, admin_pass)
        
        # Create a test role
        test_role = f"test_role_{int(time.time())}"
        access_control.create_role(
            role_name=test_role,
            description="Test role with limited permissions",
            permissions=[
                Permission.SYSTEM_STATUS.name,
                Permission.DASHBOARD_VIEW.name
            ],
            created_by=admin_user
        )
        print_test_result("Role creation", True)
        
        # Verify the role exists
        roles = access_control._load_roles()
        role_exists = test_role in roles
        print_test_result("Role exists after creation", role_exists)
        
        if role_exists:
            # Create a user with this role
            test_username = f"test_user_{int(time.time())}"
            access_control.create_user(
                username=test_username,
                password="testpassword",
                role=test_role,
                created_by=admin_user
            )
            print(f"Created test user '{test_username}' with role '{test_role}'")
            
            # Authenticate as the test user
            test_token = access_control.authenticate(test_username, "testpassword")
            
            # Check permissions
            has_dashboard_perm = access_control.has_permission(test_token, Permission.DASHBOARD_VIEW)
            print_test_result("User has expected permission", has_dashboard_perm)
            
            has_rollback_perm = access_control.has_permission(test_token, Permission.ROLLBACK_EXECUTE)
            print_test_result("User doesn't have unexpected permission", not has_rollback_perm)
            
            # Update the role
            access_control.update_role(
                role_name=test_role,
                updates={
                    "permissions": [
                        Permission.SYSTEM_STATUS.name,
                        Permission.DASHBOARD_VIEW.name,
                        Permission.ENTROPY_VIEW.name
                    ]
                },
                updated_by=admin_user
            )
            
            # Verify the role update
            roles = access_control._load_roles()
            update_success = Permission.ENTROPY_VIEW.name in roles[test_role]["permissions"]
            print_test_result("Role update", update_success)
            
            # Re-authenticate to refresh permissions
            test_token = access_control.authenticate(test_username, "testpassword")
            
            # Check if new permission is available
            has_entropy_perm = access_control.has_permission(test_token, Permission.ENTROPY_VIEW)
            print_test_result("User has new permission after role update", has_entropy_perm)
            
            # Delete the test user
            access_control.delete_user(test_username, admin_user)
            
            # Delete the role
            access_control.delete_role(test_role, admin_user)
            
            # Verify the role no longer exists
            roles = access_control._load_roles()
            delete_success = test_role not in roles
            print_test_result("Role deletion", delete_success)
            
    except Exception as e:
        print_test_result("Role management operations", False)
        print(f"  Error: {e}")


def test_file_permissions():
    """Test file access with different permission levels."""
    print_header("Testing File Access Permissions")
    
    # Create test files
    test_file = TEST_DIR / "test_file.txt"
    test_content = "This is a test file for permission checks."
    test_file.write_text(test_content)
    
    # Create test roles with different file permissions
    try:
        # Get admin credentials
        admin_user, admin_pass = ensure_test_admin()
        
        # Authenticate as admin
        admin_token = access_control.authenticate(admin_user, admin_pass)
        
        # Create reader role
        reader_role = f"test_reader_{int(time.time())}"
        access_control.create_role(
            role_name=reader_role,
            description="Role with read-only file access",
            permissions=[
                Permission.FILE_READ.name
            ],
            created_by=admin_user
        )
        
        # Create writer role
        writer_role = f"test_writer_{int(time.time())}"
        access_control.create_role(
            role_name=writer_role,
            description="Role with read and write file access",
            permissions=[
                Permission.FILE_READ.name,
                Permission.FILE_WRITE.name
            ],
            created_by=admin_user
        )
        
        # Create manager role
        manager_role = f"test_manager_{int(time.time())}"
        access_control.create_role(
            role_name=manager_role,
            description="Role with full file access",
            permissions=[
                Permission.FILE_READ.name,
                Permission.FILE_WRITE.name,
                Permission.FILE_DELETE.name
            ],
            created_by=admin_user
        )
        
        # Create users for each role
        reader_user = f"test_reader_{int(time.time())}"
        access_control.create_user(
            username=reader_user,
            password="password",
            role=reader_role,
            created_by=admin_user
        )
        
        writer_user = f"test_writer_{int(time.time())}"
        access_control.create_user(
            username=writer_user,
            password="password",
            role=writer_role,
            created_by=admin_user
        )
        
        manager_user = f"test_manager_{int(time.time())}"
        access_control.create_user(
            username=manager_user,
            password="password",
            role=manager_role,
            created_by=admin_user
        )
        
        # Authenticate as each user
        reader_token = access_control.authenticate(reader_user, "password")
        writer_token = access_control.authenticate(writer_user, "password")
        manager_token = access_control.authenticate(manager_user, "password")
        
        # Test read permissions
        print("\nTesting read permissions:")
        try:
            # Reader should be able to read
            content = file_access.read_file(reader_token, test_file)
            print_test_result("Reader can read files", content == test_content)
        except (AuthorizationError, FileAccessError) as e:
            print_test_result("Reader can read files", False)
            print(f"  Error: {e}")
            
        try:
            # Writer should be able to read
            content = file_access.read_file(writer_token, test_file)
            print_test_result("Writer can read files", content == test_content)
        except (AuthorizationError, FileAccessError) as e:
            print_test_result("Writer can read files", False)
            print(f"  Error: {e}")
            
        try:
            # Manager should be able to read
            content = file_access.read_file(manager_token, test_file)
            print_test_result("Manager can read files", content == test_content)
        except (AuthorizationError, FileAccessError) as e:
            print_test_result("Manager can read files", False)
            print(f"  Error: {e}")
            
        # Test write permissions
        print("\nTesting write permissions:")
        try:
            # Reader should not be able to write
            unauthorized = False
            try:
                file_access.write_file(reader_token, test_file, "Modified content")
            except AuthorizationError:
                unauthorized = True
            except Exception as e:
                print(f"Unexpected error: {e}")
            print_test_result("Reader cannot write files", unauthorized)
        except Exception as e:
            print_test_result("Reader write permission test", False)
            print(f"  Error: {e}")
            
        try:
            # Writer should be able to write
            writer_content = "Content modified by writer"
            file_access.write_file(writer_token, test_file, writer_content)
            content = test_file.read_text()
            print_test_result("Writer can write files", content == writer_content)
        except (AuthorizationError, FileAccessError) as e:
            print_test_result("Writer can write files", False)
            print(f"  Error: {e}")
            
        # Test delete permissions
        print("\nTesting delete permissions:")
        try:
            # Reader should not be able to delete
            unauthorized = False
            try:
                file_access.delete_file(reader_token, test_file)
            except AuthorizationError:
                unauthorized = True
            except Exception as e:
                print(f"Unexpected error: {e}")
            print_test_result("Reader cannot delete files", unauthorized)
        except Exception as e:
            print_test_result("Reader delete permission test", False)
            print(f"  Error: {e}")
            
        try:
            # Writer should not be able to delete
            unauthorized = False
            try:
                file_access.delete_file(writer_token, test_file)
            except AuthorizationError:
                unauthorized = True
            except Exception as e:
                print(f"Unexpected error: {e}")
            print_test_result("Writer cannot delete files", unauthorized)
        except Exception as e:
            print_test_result("Writer delete permission test", False)
            print(f"  Error: {e}")
            
        try:
            # Manager should be able to delete
            file_access.delete_file(manager_token, test_file)
            delete_success = not test_file.exists()
            print_test_result("Manager can delete files", delete_success)
        except (AuthorizationError, FileAccessError) as e:
            print_test_result("Manager can delete files", False)
            print(f"  Error: {e}")
            
        # Clean up test users and roles
        access_control.delete_user(reader_user, admin_user)
        access_control.delete_user(writer_user, admin_user)
        access_control.delete_user(manager_user, admin_user)
        access_control.delete_role(reader_role, admin_user)
        access_control.delete_role(writer_role, admin_user)
        access_control.delete_role(manager_role, admin_user)
        
    except Exception as e:
        print_test_result("File permission tests", False)
        print(f"  Error: {e}")
        

def test_directory_operations():
    """Test directory operations with permission checks."""
    print_header("Testing Directory Operations")
    
    # Create test directory
    test_subdir = TEST_DIR / "test_subdir"
    
    try:
        # Get admin credentials
        admin_user, admin_pass = ensure_test_admin()
        
        # Authenticate as admin
        admin_token = access_control.authenticate(admin_user, admin_pass)
        
        # Create test role with directory permissions
        dir_manager_role = f"test_dir_manager_{int(time.time())}"
        access_control.create_role(
            role_name=dir_manager_role,
            description="Role with directory management permissions",
            permissions=[
                Permission.FILE_READ.name,
                Permission.FILE_WRITE.name,
                Permission.FILE_DELETE.name
            ],
            created_by=admin_user
        )
        
        # Create user with directory permissions
        dir_manager_user = f"test_dir_manager_{int(time.time())}"
        access_control.create_user(
            username=dir_manager_user,
            password="password",
            role=dir_manager_role,
            created_by=admin_user
        )
        
        # Authenticate as directory manager
        dir_manager_token = access_control.authenticate(dir_manager_user, "password")
        
        # Test create directory
        file_access.create_directory(dir_manager_token, test_subdir)
        create_success = test_subdir.exists() and test_subdir.is_dir()
        print_test_result("Create directory", create_success)
        
        # Test list directory
        test_file1 = test_subdir / "test1.txt"
        test_file2 = test_subdir / "test2.txt"
        
        file_access.write_file(dir_manager_token, test_file1, "Test file 1")
        file_access.write_file(dir_manager_token, test_file2, "Test file 2")
        
        dir_contents = file_access.list_directory(dir_manager_token, test_subdir)
        list_success = len(dir_contents) == 2
        print_test_result("List directory", list_success)
        
        # Test get file info
        file_info = file_access.get_file_info(dir_manager_token, test_file1)
        info_success = (
            file_info["exists"] and 
            file_info["is_file"] and 
            file_info["name"] == "test1.txt"
        )
        print_test_result("Get file info", info_success)
        
        # Test file exists check
        exists_check = file_access.file_exists(dir_manager_token, test_file1)
        print_test_result("File exists check", exists_check)
        
        # Test file readable check
        readable_check = file_access.file_readable(dir_manager_token, test_file1)
        print_test_result("File readable check", readable_check)
        
        # Test file writable check
        writable_check = file_access.file_writable(dir_manager_token, test_file1)
        print_test_result("File writable check", writable_check)
        
        # Test delete directory (recursive)
        file_access.delete_directory(dir_manager_token, test_subdir, recursive=True)
        delete_success = not test_subdir.exists()
        print_test_result("Delete directory (recursive)", delete_success)
        
        # Clean up test user and role
        access_control.delete_user(dir_manager_user, admin_user)
        access_control.delete_role(dir_manager_role, admin_user)
        
    except Exception as e:
        print_test_result("Directory operations", False)
        print(f"  Error: {e}")


def main():
    print_header("ACCESS CONTROL SYSTEM TEST SUITE")
    
    # First try to clean up any leftover test users/roles
    cleanup_test_users()
    
    try:
        test_user_management()
        test_role_management()
        test_file_permissions()
        test_directory_operations()
    finally:
        # Final cleanup
        cleanup_test_users()
    
    print("\n")
    print_header("TEST SUITE COMPLETE")


if __name__ == "__main__":
    main()
