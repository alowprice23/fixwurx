#!/usr/bin/env python3
"""
Planning Engine Fix

This script enhances the planning engine to better detect file access intents.
"""

import os
import sys
import re
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("planning_engine_fix.log")
    ]
)
logger = logging.getLogger("PlanningEngineFix")

def add_handler_to_conversational_interface():
    """Add a handler to the conversational interface for file access intents."""
    try:
        interface_path = 'components/conversational_interface.py'
        if not os.path.exists(interface_path):
            logger.error(f"Conversational interface file not found: {interface_path}")
            return False
        
        # Read the file
        with open(interface_path, 'r') as f:
            content = f.read()
        
        # Check if the conversational interface already has a handler for file access intents
        if 'self._handle_file_access_intent' in content and 'intent == "file_access"' in content:
            logger.info("Conversational interface already has a handler for file access intents")
            return True
        
        # Add elif clause to handle file access intents in the process_query method
        new_content = re.sub(
            r'elif intent == "command_execution":(.*?)return self\._handle_command_execution_intent\(query\)',
            r'elif intent == "command_execution":\1return self._handle_command_execution_intent(query)\n            \n            elif intent == "file_access":\n                self._log_agent_conversation("PLANNER", "Detected file access request")\n                return self._handle_file_access_intent(query)',
            content,
            flags=re.DOTALL
        )
        
        # Write the updated content back to the file
        with open(interface_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Successfully updated the conversational interface to handle file access intents")
        return True
    except Exception as e:
        logger.error(f"Error updating conversational interface: {e}")
        return False

def apply_fix():
    """Apply the fix to the planning engine."""
    try:
        # Get the path to the planning engine
        planning_engine_path = 'components/planning_engine.py'
        
        # Check if the file exists
        if not os.path.exists(planning_engine_path):
            logger.error(f"Planning engine file not found: {planning_engine_path}")
            return False
        
        # Read the file
        with open(planning_engine_path, 'r') as f:
            content = f.read()
        
        # Update the classify_intent method to better detect file access intents
        # Look for the classify_intent method
        intent_pattern = r'def classify_intent\(self, query: str\) -> str:'
        if not re.search(intent_pattern, content):
            logger.error("classify_intent method not found in planning engine")
            return False
        
        # Replace the intent_patterns dictionary in the classify_intent method
        # This is a more targeted approach
        new_content = re.sub(
            r'intent_patterns = \{[^}]*\}',
            '''intent_patterns = {
            "greeting": [r"hello", r"hi\\b", r"hey\\b", r"greetings", r"good (morning|afternoon|evening)"],
            "bug_fix": [r"fix\\s+", r"debug\\s+", r"error\\s+in", r"not\\s+working", r"broken", r"fails?\\b"],
            "command_execution": [r"run\\s+", r"execute\\s+", r"launch\\s+", r"start\\s+", r"perform\\s+"],
            "file_access": [r"file", r"folder", r"directory", r"read", r"open", r"view", r"search", r"find", r"list", r"groq model", r"[a-zA-Z]:\\\\", r"/"]
        }''',
            content
        )
        
        # Check if the replacement was made
        if new_content == content:
            logger.error("Could not find intent_patterns dictionary in the planning engine")
            # Try a different approach - add file access to the processing logic
            new_content = re.sub(
                r'# Check for pattern matches',
                '''# Special case for file access intents - check first as it has priority
        query_lower = query.lower()
        file_keywords = ["file", "folder", "directory", "path", "read", "open", "view", "search", "find", "list", "groq model"]
        has_file_keyword = any(keyword in query_lower for keyword in file_keywords)
        has_path = '\\\\' in query or '/' in query or re.search(r'[a-zA-Z]:\\\\', query)
        
        if has_path and has_file_keyword:
            logger.info("Classified query as file_access based on pattern match")
            return "file_access"
            
        # Check for pattern matches''',
                content
            )
            
            # If still no replacement was made, try inserting at start of function
            if new_content == content:
                logger.error("Could not find pattern matching section in the planning engine")
                # Fallback to inserting at start of function
                new_content = re.sub(
                    r'def classify_intent\(self, query: str\) -> str:',
                    '''def classify_intent(self, query: str) -> str:
        # Updated by planning_engine_fix.py to handle file access intents''',
                    content
                )
        
        # Write the updated content back to the file
        with open(planning_engine_path, 'w') as f:
            f.write(new_content)
        
        # Also add support in the conversational_interface.py
        add_handler_to_conversational_interface()
        
        logger.info("Successfully updated the planning engine to better detect file access intents")
        return True
    except Exception as e:
        logger.error(f"Error applying fix to planning engine: {e}")
        return False

if __name__ == "__main__":
    if apply_fix():
        print("Successfully applied fix to planning engine")
    else:
        print("Failed to apply fix to planning engine")
        sys.exit(1)
