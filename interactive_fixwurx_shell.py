#!/usr/bin/env python3
"""
FixWurx Interactive Shell with Intent Classification

This script provides an interactive shell for the FixWurx system with advanced
intent classification. It allows users to enter natural language commands which
are processed through the intent classification system and executed.
"""

import os
import sys
import time
import json
import random
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fixwurx_shell.log"), logging.StreamHandler()]
)
logger = logging.getLogger("FixWurxShell")

# Sample buggy code to debug
SAMPLE_BUGGY_CODE = {
    "main.py": """
def main():
    print("Starting application...")
    data = load_data("data.json")  # Bug 1: No error handling
    processed = process_data(data)
    save_results(processed)  # Bug 2: No return value check
    print("Application finished.")

def load_data(filename):
    with open(filename, 'r') as f:  # Bug 3: No file existence check
        return json.load(f)

def process_data(data):
    result = []
    for item in data:
        if item['active']:
            value = item['value'] * 2  # Bug 4: No key check
            result.append({'id': item['id'], 'processed_value': value})
    return result

def save_results(results):
    with open('results.json', 'w') as f:
        json.dump(results, f)
    """,
    
    "utils.py": """
import os
import logging

logger = logging.getLogger(__name__)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")
    
def clean_string(s):
    return s.strip().lower()  # Bug 5: No None check

def calculate_average(values):
    return sum(values) / len(values)  # Bug 6: No empty list check

def validate_config(config):
    required_keys = ['api_key', 'endpoint', 'timeout']
    for key in required_keys:
        if key not in config:  # Bug 7: No type check for config
            logger.error(f"Missing required config key: {key}")
            return False
    
    if config['timeout'] <= 0:  # Bug 8: No type check for timeout
        logger.error("Timeout must be positive")
        return False
        
    return True
    """,
    
    "api.py": """
import requests
import json
import logging
from utils import validate_config

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:  # Bug 9: No error handling
            self.config = json.load(f)
        
        if not validate_config(self.config):
            raise ValueError("Invalid configuration")
            
        self.base_url = self.config['endpoint']
        self.api_key = self.config['api_key']
        self.timeout = self.config['timeout']
        
    def get_data(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()  # Bug 10: No specific error handling
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None  # Bug 11: Inconsistent error handling
    
    def post_data(self, endpoint, data):
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise  # Bug 12: Inconsistent with get_data error handling
    """
}

# Create a class to represent different components of the system
class Component:
    def __init__(self, name: str, status: str = "healthy"):
        self.name = name
        self.status = status
        self.start_time = time.time()
        self.load = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        # Simulate fluctuations in load
        self.load = min(1.0, max(0.0, self.load + random.uniform(-0.1, 0.15)))
        
        return {
            "name": self.name,
            "status": self.status,
            "uptime": time.time() - self.start_time,
            "load": self.load
        }
    
    def execute(self, command: str, *args, **kwargs) -> Dict[str, Any]:
        # Simulate execution
        time.sleep(random.uniform(0.05, 0.2))
        return {"success": True, "result": f"Executed {command} on {self.name}"}

class IntentClassifier:
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Check cache
        if query in self.cache:
            self.cache_hits += 1
            print(f"[Intent Cache] Hit for: {query}")
            return self.cache[query]
        
        self.cache_misses += 1
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        # Classify based on keywords for debugging
        if "debug" in query.lower() or "analyze" in query.lower() or "check" in query.lower():
            intent_type = "system_debugging"
            confidence = random.uniform(0.8, 0.95)
            parameters = {}
            
            # Extract folder path if present
            import re
            folder_match = re.search(r'folder[s]?\s+(["\'])?(.*?)\1', query)
            if folder_match:
                parameters["folder_path"] = folder_match.group(2)
            else:
                folder_match = re.search(r'(["\'])(.*?)\1', query)
                if folder_match:
                    parameters["folder_path"] = folder_match.group(2)
                else:
                    parameters["folder_path"] = "sample_buggy_code"
            
            # Extract specific file patterns or bug types
            if "syntax" in query.lower():
                parameters["bug_type"] = "syntax"
            elif "logic" in query.lower():
                parameters["bug_type"] = "logic"
            elif "security" in query.lower():
                parameters["bug_type"] = "security"
            elif "performance" in query.lower():
                parameters["bug_type"] = "performance"
            else:
                parameters["bug_type"] = "all"
            
            # Determine execution path
            execution_path = "agent_collaboration"
            required_agents = ["analyzer", "debugger", "executor"]
            
        elif "fix" in query.lower() or "repair" in query.lower() or "solve" in query.lower():
            intent_type = "code_repair"
            confidence = random.uniform(0.75, 0.9)
            parameters = {}
            
            # Extract folder path if present
            import re
            folder_match = re.search(r'folder[s]?\s+(["\'])?(.*?)\1', query)
            if folder_match:
                parameters["folder_path"] = folder_match.group(2)
            else:
                folder_match = re.search(r'(["\'])(.*?)\1', query)
                if folder_match:
                    parameters["folder_path"] = folder_match.group(2)
                else:
                    parameters["folder_path"] = "sample_buggy_code"
            
            execution_path = "agent_collaboration"
            required_agents = ["developer", "debugger", "tester"]
            
        elif "show" in query.lower() or "display" in query.lower() or "list" in query.lower():
            intent_type = "file_access"
            confidence = random.uniform(0.85, 0.98)
            parameters = {}
            
            # Extract folder path if present
            import re
            folder_match = re.search(r'folder[s]?\s+(["\'])?(.*?)\1', query)
            if folder_match:
                parameters["folder_path"] = folder_match.group(2)
            else:
                folder_match = re.search(r'(["\'])(.*?)\1', query)
                if folder_match:
                    parameters["folder_path"] = folder_match.group(2)
                else:
                    parameters["folder_path"] = "sample_buggy_code"
            
            execution_path = "direct"
            required_agents = ["file_handler"]
            
        else:
            intent_type = "generic"
            confidence = random.uniform(0.6, 0.8)
            parameters = {}
            execution_path = "planning"
            required_agents = ["planner", "executor"]
        
        # Create intent object
        intent = {
            "query": query,
            "type": intent_type,
            "confidence": confidence,
            "parameters": parameters,
            "execution_path": execution_path,
            "required_agents": required_agents
        }
        
        # Cache the result
        self.cache[query] = intent
        
        return intent

class FileSystem:
    def __init__(self):
        self.files = {}
        
    def create_file(self, path: str, content: str) -> bool:
        self.files[path] = content
        return True
        
    def read_file(self, path: str) -> Optional[str]:
        return self.files.get(path)
        
    def list_files(self, folder: str = "") -> List[str]:
        if not folder:
            return list(self.files.keys())
        
        return [f for f in self.files.keys() if f.startswith(folder)]
        
    def file_exists(self, path: str) -> bool:
        return path in self.files

class DebugEngine:
    def __init__(self):
        pass
        
    def analyze_code(self, code: str, bug_type: str = "all") -> List[Dict[str, Any]]:
        """Analyze code for bugs."""
        bugs = []
        
        # Simple pattern-based bug detection
        if bug_type in ["all", "syntax"]:
            # Check for missing parentheses
            if code.count('(') != code.count(')'):
                bugs.append({
                    "type": "syntax",
                    "description": "Mismatched parentheses",
                    "severity": "high"
                })
                
            # Check for missing colons in function definitions or if statements
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if (line.strip().startswith('def ') or 
                    line.strip().startswith('if ') or 
                    line.strip().startswith('else') or 
                    line.strip().startswith('for ') or 
                    line.strip().startswith('while ')) and not line.strip().endswith(':'):
                    bugs.append({
                        "type": "syntax",
                        "description": f"Missing colon at line {i+1}",
                        "line": i+1,
                        "severity": "high"
                    })
        
        if bug_type in ["all", "logic"]:
            # Check for potential null pointer dereferences
            if ".strip()" in code and "None" not in code:
                bugs.append({
                    "type": "logic",
                    "description": "Potential null pointer dereference - calling methods on possibly None values",
                    "severity": "medium"
                })
                
            # Check for division by zero
            if "/" in code and "len(" in code and "if" not in code:
                bugs.append({
                    "type": "logic",
                    "description": "Potential division by zero when calculating averages or ratios",
                    "severity": "high"
                })
                
            # Check for unchecked dictionary access
            if "[" in code and "in" not in code:
                bugs.append({
                    "type": "logic",
                    "description": "Unchecked dictionary access - potential KeyError",
                    "severity": "medium"
                })
        
        if bug_type in ["all", "security"]:
            # Check for hardcoded credentials
            if "password" in code.lower() or "api_key" in code.lower():
                if "=" in code and ("\"" in code or "'" in code):
                    bugs.append({
                        "type": "security",
                        "description": "Potential hardcoded credentials",
                        "severity": "critical"
                    })
                    
            # Check for SQL injection
            if "sql" in code.lower() and "%" in code:
                bugs.append({
                    "type": "security",
                    "description": "Potential SQL injection vulnerability",
                    "severity": "critical"
                })
        
        if bug_type in ["all", "performance"]:
            # Check for inefficient list operations
            if "for" in code and "append" in code:
                bugs.append({
                    "type": "performance",
                    "description": "Potentially inefficient list operations - consider list comprehensions",
                    "severity": "low"
                })
                
            # Check for repeated computations
            if "for" in code and "if" in code and any(op in code for op in ["sum(", "len(", "max(", "min("]):
                bugs.append({
                    "type": "performance",
                    "description": "Potential repeated computations in loop",
                    "severity": "low"
                })
        
        return bugs
        
    def debug_folder(self, files: Dict[str, str], bug_type: str = "all") -> Dict[str, List[Dict[str, Any]]]:
        """Debug all files in a folder."""
        results = {}
        
        for file_path, content in files.items():
            bugs = self.analyze_code(content, bug_type)
            if bugs:
                results[file_path] = bugs
                
        return results
        
    def suggest_fixes(self, code: str, bugs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Suggest fixes for the bugs."""
        fixes = {}
        
        for bug in bugs:
            if bug["type"] == "syntax":
                if "missing colon" in bug["description"].lower():
                    # Simple fix for missing colons
                    lines = code.split('\n')
                    line_idx = bug.get("line", 1) - 1
                    lines[line_idx] = lines[line_idx].rstrip() + ":"
                    fixes["Add missing colon"] = "\n".join(lines)
                    
            elif bug["type"] == "logic":
                if "null pointer" in bug["description"].lower():
                    # Fix for potential null pointer dereference
                    fixes["Add null check"] = code.replace(".strip()", "if x is not None else ''")
                
                if "division by zero" in bug["description"].lower():
                    # Fix for division by zero
                    if "len(" in code:
                        fixes["Add empty list check"] = code.replace(
                            "sum(values) / len(values)", 
                            "sum(values) / len(values) if values else 0"
                        )
                
                if "dictionary access" in bug["description"].lower():
                    # Fix for unchecked dictionary access
                    fixes["Add key check"] = "Add checks using the 'in' operator before accessing dictionary keys"
            
            elif bug["type"] == "security":
                if "hardcoded credentials" in bug["description"].lower():
                    fixes["Use environment variables"] = "Replace hardcoded credentials with environment variables or a secure vault"
                
                if "sql injection" in bug["description"].lower():
                    fixes["Use parameterized queries"] = "Replace string formatting with parameterized queries"
            
            elif bug["type"] == "performance":
                if "list operations" in bug["description"].lower():
                    fixes["Use list comprehension"] = "Replace for loop and append with list comprehension"
                
                if "repeated computations" in bug["description"].lower():
                    fixes["Cache repeated computations"] = "Calculate values once before the loop and reuse them"
        
        return fixes

class FixWurxSystem:
    def __init__(self):
        # Initialize components
        self.components = {
            "intent_classifier": IntentClassifier(),
            "debug_engine": DebugEngine(),
            "file_system": FileSystem(),
            "neural_matrix": Component("neural_matrix"),
            "triangulum": Component("triangulum"),
            "agent_system": Component("agent_system")
        }
        
        # Initialize sample buggy code
        self.setup_sample_code()
        
        # System state
        self.conversation_history = []
        self.current_folder = "sample_buggy_code"
        self.running = True
    
    def setup_sample_code(self):
        """Set up sample buggy code in the file system."""
        fs = self.components["file_system"]
        for filename, content in SAMPLE_BUGGY_CODE.items():
            fs.create_file(f"sample_buggy_code/{filename}", content)
    
    def process_input(self, user_input: str) -> str:
        """Process user input through the intent classification system."""
        # Skip empty input
        if not user_input.strip():
            return ""
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "q"]:
            self.running = False
            return "Exiting FixWurx shell..."
        
        # Record in history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Get context
        context = {
            "history": self.conversation_history,
            "current_folder": self.current_folder
        }
        
        # Classify intent
        intent = self.components["intent_classifier"].classify(user_input, context)
        
        # Log the classified intent
        print(f"\n[Intent Classification] Type: {intent['type']}, Confidence: {intent['confidence']:.2f}")
        print(f"[Intent Classification] Execution Path: {intent['execution_path']}")
        print(f"[Intent Classification] Required Agents: {', '.join(intent['required_agents'])}")
        if intent['parameters']:
            print(f"[Intent Classification] Parameters: {json.dumps(intent['parameters'], indent=2)}")
        
        # Process based on intent type
        if intent["type"] == "system_debugging":
            return self.handle_debugging_intent(intent)
        elif intent["type"] == "code_repair":
            return self.handle_repair_intent(intent)
        elif intent["type"] == "file_access":
            return self.handle_file_access_intent(intent)
        else:
            return self.handle_generic_intent(intent)
    
    def handle_debugging_intent(self, intent: Dict[str, Any]) -> str:
        """Handle debugging intent."""
        folder_path = intent["parameters"].get("folder_path", "sample_buggy_code")
        bug_type = intent["parameters"].get("bug_type", "all")
        
        # Get files from the folder
        fs = self.components["file_system"]
        files = {}
        for file_path in fs.list_files(folder_path):
            content = fs.read_file(file_path)
            if content:
                files[file_path] = content
        
        if not files:
            return f"No files found in folder '{folder_path}'."
        
        # Debug the files
        debug_engine = self.components["debug_engine"]
        results = debug_engine.debug_folder(files, bug_type)
        
        if not results:
            return f"No bugs found in folder '{folder_path}'."
        
        # Format the results
        output = [f"Debugging Results for '{folder_path}':\n"]
        
        total_bugs = sum(len(bugs) for bugs in results.values())
        output.append(f"Found {total_bugs} potential issues in {len(results)} files:\n")
        
        for file_path, bugs in results.items():
            output.append(f"File: {file_path}")
            output.append("  " + "-" * 40)
            
            for i, bug in enumerate(bugs, 1):
                output.append(f"  Issue #{i}: {bug['description']}")
                output.append(f"    Type: {bug['type']}")
                output.append(f"    Severity: {bug['severity']}")
                if "line" in bug:
                    output.append(f"    Line: {bug['line']}")
                output.append("")
            
            # Suggest fixes
            content = fs.read_file(file_path)
            fixes = debug_engine.suggest_fixes(content, bugs)
            
            if fixes:
                output.append("  Suggested Fixes:")
                for fix_name, fix_details in fixes.items():
                    output.append(f"    - {fix_name}: {fix_details}")
            
            output.append("")
        
        return "\n".join(output)
    
    def handle_repair_intent(self, intent: Dict[str, Any]) -> str:
        """Handle code repair intent."""
        folder_path = intent["parameters"].get("folder_path", "sample_buggy_code")
        
        # Get files from the folder
        fs = self.components["file_system"]
        files = {}
        for file_path in fs.list_files(folder_path):
            content = fs.read_file(file_path)
            if content:
                files[file_path] = content
        
        if not files:
            return f"No files found in folder '{folder_path}'."
        
        # Debug the files first
        debug_engine = self.components["debug_engine"]
        results = debug_engine.debug_folder(files)
        
        if not results:
            return f"No bugs found in folder '{folder_path}'. Nothing to fix."
        
        # Fix the bugs
        output = [f"Fixing bugs in '{folder_path}':\n"]
        
        for file_path, bugs in results.items():
            output.append(f"File: {file_path}")
            content = fs.read_file(file_path)
            
            # Apply fixes
            fixes = debug_engine.suggest_fixes(content, bugs)
            
            if not fixes:
                output.append("  No automatic fixes available for this file.")
                continue
            
            output.append("  Applied fixes:")
            
            # For simplicity, just apply the first fix
            for fix_name, fixed_content in fixes.items():
                if isinstance(fixed_content, str) and len(fixed_content) > 10:
                    # It's actual code, so update the file
                    fs.create_file(file_path, fixed_content)
                    output.append(f"    - {fix_name}")
                    break
                else:
                    # It's just a suggestion
                    output.append(f"    - {fix_name}: {fixed_content}")
            
            output.append("")
        
        return "\n".join(output)
    
    def handle_file_access_intent(self, intent: Dict[str, Any]) -> str:
        """Handle file access intent."""
        folder_path = intent["parameters"].get("folder_path", "sample_buggy_code")
        
        # List files in the folder
        fs = self.components["file_system"]
        files = fs.list_files(folder_path)
        
        if not files:
            return f"No files found in folder '{folder_path}'."
        
        # Format the output
        output = [f"Files in '{folder_path}':\n"]
        
        for file_path in files:
            output.append(f"- {file_path}")
            
            # Show a preview of the file
            content = fs.read_file(file_path)
            if content:
                lines = content.split('\n')
                preview = '\n'.join(lines[:5])
                output.append(f"\n  Preview:\n  {preview.replace('\n', '\n  ')}\n")
        
        return "\n".join(output)
    
    def handle_generic_intent(self, intent: Dict[str, Any]) -> str:
        """Handle generic intent."""
        # Generate a plan
        plan = {
            "goal": "Process user query",
            "steps": [
                {"description": "Understand user intent", "status": "completed"},
                {"description": "Generate appropriate response", "status": "completed"}
            ],
            "response": f"I understand you want to '{intent['query']}'. Could you provide more specific instructions? For example:\n\n- 'Debug folder sample_buggy_code for syntax errors'\n- 'Fix all bugs in the utils.py file'\n- 'Show me all files in the sample_buggy_code folder'"
        }
        
        return plan["response"]
    
    def run_shell(self):
        """Run the interactive shell."""
        print("\n=== FixWurx Shell with Advanced Intent Classification ===")
        print("Type natural language commands to interact with the system.")
        print("For example: 'debug the sample_buggy_code folder for logic errors'")
        print("Type 'exit', 'quit', or 'q' to exit the shell.\n")
        
        while self.running:
            try:
                # Get user input
                user_input = input("> ")
                
                # Process the input
                output = self.process_input(user_input)
                
                # Display the output
                if output:
                    print(f"\n{output}\n")
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to exit the shell.")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nFixWurx shell closed. Thank you for using the system.")

def main():
    """Main entry point."""
    system = FixWurxSystem()
    system.run_shell()

if __name__ == "__main__":
    main()
