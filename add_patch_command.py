#!/usr/bin/env python3
"""
Add Patch Command Script

This script adds the missing patch_command function to auditor_commands.py
to fix the 'name patch_command is not defined' error.
"""

import os
import sys

def main():
    # Check if auditor_commands.py exists
    if not os.path.exists('auditor_commands.py'):
        print("Error: auditor_commands.py not found")
        return 1
    
    # Check if auditor_missing_functions.py exists
    if not os.path.exists('auditor_missing_functions.py'):
        print("Error: auditor_missing_functions.py not found")
        return 1
    
    # Read the missing functions
    with open('auditor_missing_functions.py', 'r') as f:
        missing_functions = f.read()
    
    # Find the patch_command function in auditor_commands.py
    with open('auditor_commands.py', 'r') as f:
        auditor_commands = f.read()
    
    # Check if patch_command already exists
    if 'def patch_command(' in auditor_commands:
        print("Note: patch_command already exists in auditor_commands.py")
        return 0
    
    # Create the patch_command function
    patch_command_function = """
def patch_command(args: str) -> int:
    \"\"\"
    Manage and apply patches.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    \"\"\"
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage and apply patches")
    parser.add_argument("action", nargs="?", choices=["list", "view", "apply", "revert"], default="list", 
                        help="Action to perform")
    parser.add_argument("patch_id", nargs="?", help="Patch ID for view/apply/revert actions")
    parser.add_argument("--status", choices=["pending", "applied", "failed", "reverted", "all"], default="pending", 
                       help="Filter by patch status")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    patch_id = cmd_args.patch_id
    status = cmd_args.status
    output_format = cmd_args.format
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # List patches
        if action == "list":
            # Get patches
            patches_filter = {}
            if status != "all":
                patches_filter["status"] = status
            
            patches = auditor.get_patches(filter=patches_filter)
            
            # Display patches
            if output_format == "json":
                print(json.dumps(patches, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(patches, default_flow_style=False))
            else:  # text format
                print("\\nPatches:")
                
                if not patches:
                    print("No patches found matching the criteria")
                    return 0
                
                print(f"Found {len(patches)} patches:")
                for i, patch in enumerate(patches, 1):
                    patch_id = patch.get("id", "Unknown")
                    status = patch.get("status", "Unknown")
                    target = patch.get("target", "Unknown")
                    created_at = patch.get("created_at", "Unknown")
                    description = patch.get("description", "No description")
                    
                    print(f"  {i}. {patch_id} - {status}")
                    print(f"     Target: {target}")
                    print(f"     Created: {created_at}")
                    print(f"     Description: {description}")
        
        # View patch details
        elif action == "view" and patch_id:
            # Get patch details
            patch = auditor.get_patch_details(patch_id)
            
            if not patch:
                print(f"Patch '{patch_id}' not found")
                return 1
            
            # Display patch
            if output_format == "json":
                print(json.dumps(patch, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(patch, default_flow_style=False))
            else:  # text format
                print(f"\\nPatch Details - {patch_id}")
                print("=" * 60)
                
                # Print patch information
                print(f"Status: {patch.get('status', 'Unknown')}")
                print(f"Target: {patch.get('target', 'Unknown')}")
                print(f"Created: {patch.get('created_at', 'Unknown')}")
                print(f"Applied: {patch.get('applied_at', 'Never')}")
                print(f"Description: {patch.get('description', 'No description')}")
                
                # Print patch changes
                if "changes" in patch:
                    print("\\nChanges:")
                    changes = patch["changes"]
                    if isinstance(changes, list):
                        for i, change in enumerate(changes, 1):
                            print(f"  {i}. {change.get('file', 'Unknown')}:")
                            print(f"     Type: {change.get('type', 'Unknown')}")
                            if "line" in change:
                                print(f"     Line: {change['line']}")
                            if "content" in change:
                                print(f"     Content: {change['content']}")
                    elif isinstance(changes, dict):
                        for file, file_changes in changes.items():
                            print(f"  {file}:")
                            if isinstance(file_changes, list):
                                for change in file_changes:
                                    print(f"     {change}")
                            else:
                                print(f"     {file_changes}")
                
                # Print patch verification
                if "verification" in patch:
                    print("\\nVerification:")
                    verification = patch["verification"]
                    if isinstance(verification, dict):
                        for key, value in verification.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {verification}")
        
        # Apply a patch
        elif action == "apply" and patch_id:
            # Apply the patch
            result = auditor.apply_patch(patch_id)
            
            # Display result
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                success = result.get("success", False)
                print(f"\\nApply result for patch {patch_id}:")
                
                if success:
                    print("[SUCCESS] Patch applied successfully")
                else:
                    print("[FAILED] Failed to apply patch")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
                
                # Print application details
                if "details" in result:
                    print("\\nDetails:")
                    details = result["details"]
                    if isinstance(details, str):
                        print(details)
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            print(f"  {key}: {value}")
                    elif isinstance(details, list):
                        for item in details:
                            print(f"  - {item}")
        
        # Revert a patch
        elif action == "revert" and patch_id:
            # Revert the patch
            result = auditor.revert_patch(patch_id)
            
            # Display result
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                success = result.get("success", False)
                print(f"\\nRevert result for patch {patch_id}:")
                
                if success:
                    print("[SUCCESS] Patch reverted successfully")
                else:
                    print("[FAILED] Failed to revert patch")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
                
                # Print revert details
                if "details" in result:
                    print("\\nDetails:")
                    details = result["details"]
                    if isinstance(details, str):
                        print(details)
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            print(f"  {key}: {value}")
                    elif isinstance(details, list):
                        for item in details:
                            print(f"  - {item}")
        
        # Unknown action
        else:
            print("Error: Invalid action or missing required arguments")
            print("Usage examples:")
            print("  patch list [--status <status>]")
            print("  patch view <patch_id>")
            print("  patch apply <patch_id>")
            print("  patch revert <patch_id>")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error managing patches: {e}")
        return 1
"""
    
    # Append the patch_command function to auditor_commands.py
    with open('auditor_commands.py', 'a') as f:
        f.write(patch_command_function)
    
    print("Successfully added patch_command function to auditor_commands.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
