#!/usr/bin/env python3
"""
Shell Scripting Module

This module provides advanced shell scripting capabilities for the FixWurx shell environment,
including variables, conditionals, loops, functions, and error handling.
"""

import os
import sys
import json
import time
import logging
import shlex
import re
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

logger = logging.getLogger("ShellScripting")

class SimpleScriptInterpreter:
    """A simplified script interpreter for FixWurx shell scripts."""
    
    def __init__(self, registry=None):
        """
        Initialize the interpreter.
        
        Args:
            registry: Component registry
        """
        self.registry = registry
        self.variables = {}
        self.functions = {}
        self.exit_code = 0
    
    def execute(self, script: str) -> int:
        """
        Execute a script.
        
        Args:
            script: Script to execute
            
        Returns:
            Exit code
        """
        # Reset state
        self.exit_code = 0
        
        # Split script into lines
        lines = script.split('\n')
        
        # Parse and execute each line
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Handle if statements
            if line.startswith('if '):
                i = self._handle_if_statement(lines, i)
                continue
            
            # Handle for loops
            if line.startswith('for '):
                i = self._handle_for_loop(lines, i)
                continue
            
            # Handle while loops
            if line.startswith('while '):
                i = self._handle_while_loop(lines, i)
                continue
            
            # Handle function definitions
            if line.startswith('function '):
                i = self._handle_function_definition(lines, i)
                continue
            
            # Handle variable assignments
            if '=' in line and not line.startswith(('if ', 'while ', 'for ')):
                self._handle_assignment(line)
                i += 1
                continue
            
            # Handle commands
            if not line.startswith(('if ', 'else', 'elif ', 'fi', 'for ', 'in ', 'do', 'done', 'while ', 'until ', 'function ')):
                self._handle_command(line)
                i += 1
                continue
            
            # Skip unknown lines
            i += 1
        
        return self.exit_code
    
    def _handle_if_statement(self, lines: List[str], start_idx: int) -> int:
        """
        Handle an if statement.
        
        Args:
            lines: Script lines
            start_idx: Start index
            
        Returns:
            End index
        """
        line = lines[start_idx].strip()
        
        # Extract condition
        condition = line[3:].strip()
        if condition.endswith(' then'):
            condition = condition[:-5].strip()
        
        # Evaluate condition
        condition_result = self._evaluate_expression(condition)
        
        # Find end of if statement
        end_idx = self._find_block_end(lines, start_idx, 'if', 'fi')
        
        # Find else block
        else_idx = self._find_else(lines, start_idx, end_idx)
        
        if condition_result:
            # Execute then block
            block_start = start_idx + 1
            block_end = else_idx if else_idx != -1 else end_idx
            
            for i in range(block_start, block_end):
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line and not line.startswith(('if ', 'while ', 'for ')):
                    self._handle_assignment(line)
                elif not line.startswith(('else', 'elif', 'fi')):
                    self._handle_command(line)
        else:
            # Execute else block if it exists
            if else_idx != -1:
                for i in range(else_idx + 1, end_idx):
                    line = lines[i].strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line and not line.startswith(('if ', 'while ', 'for ')):
                        self._handle_assignment(line)
                    elif not line == 'fi':
                        self._handle_command(line)
        
        return end_idx + 1
    
    def _handle_for_loop(self, lines: List[str], start_idx: int) -> int:
        """
        Handle a for loop.
        
        Args:
            lines: Script lines
            start_idx: Start index
            
        Returns:
            End index
        """
        line = lines[start_idx].strip()
        
        # Extract variable and iterable
        match = re.match(r'for\s+(\w+)\s+in\s+(.+)\s+do', line)
        
        if not match:
            logger.error(f"Invalid for loop: {line}")
            return start_idx + 1
        
        var_name = match.group(1)
        iterable_expr = match.group(2)
        
        # Evaluate iterable
        iterable = self._evaluate_expression(iterable_expr)
        
        # Find end of for loop
        end_idx = self._find_block_end(lines, start_idx, 'for', 'done')
        
        # Execute loop block
        loop_lines = lines[start_idx + 1:end_idx]
        
        for item in iterable:
            # Set loop variable
            self.variables[var_name] = item
            
            # Execute loop body
            for loop_line in loop_lines:
                loop_line = loop_line.strip()
                
                if not loop_line or loop_line.startswith('#'):
                    continue
                
                if '=' in loop_line and not loop_line.startswith(('if ', 'while ', 'for ')):
                    self._handle_assignment(loop_line)
                else:
                    self._handle_command(loop_line)
        
        return end_idx + 1
    
    def _handle_while_loop(self, lines: List[str], start_idx: int) -> int:
        """
        Handle a while loop.
        
        Args:
            lines: Script lines
            start_idx: Start index
            
        Returns:
            End index
        """
        line = lines[start_idx].strip()
        
        # Extract condition
        match = re.match(r'while\s+(.+)\s+do', line)
        
        if not match:
            logger.error(f"Invalid while loop: {line}")
            return start_idx + 1
        
        condition = match.group(1)
        
        # Find end of while loop
        end_idx = self._find_block_end(lines, start_idx, 'while', 'done')
        
        # Execute loop block
        loop_lines = lines[start_idx + 1:end_idx]
        
        while self._evaluate_expression(condition):
            # Execute loop body
            for loop_line in loop_lines:
                loop_line = loop_line.strip()
                
                if not loop_line or loop_line.startswith('#'):
                    continue
                
                if '=' in loop_line and not loop_line.startswith(('if ', 'while ', 'for ')):
                    self._handle_assignment(loop_line)
                else:
                    self._handle_command(loop_line)
        
        return end_idx + 1
    
    def _handle_function_definition(self, lines: List[str], start_idx: int) -> int:
        """
        Handle a function definition.
        
        Args:
            lines: Script lines
            start_idx: Start index
            
        Returns:
            End index
        """
        line = lines[start_idx].strip()
        
        # Extract function name and parameters
        match = re.match(r'function\s+(\w+)\s*\((.*)\)', line)
        
        if not match:
            logger.error(f"Invalid function definition: {line}")
            return start_idx + 1
        
        func_name = match.group(1)
        params = [p.strip() for p in match.group(2).split(',') if p.strip()]
        
        # Find end of function definition
        end_idx = self._find_block_end(lines, start_idx, 'function', 'end')
        
        # Store function definition
        self.functions[func_name] = {
            'params': params,
            'body': lines[start_idx + 1:end_idx]
        }
        
        return end_idx + 1
    
    def _handle_assignment(self, line: str) -> None:
        """
        Handle a variable assignment.
        
        Args:
            line: Assignment line
        """
        parts = line.split('=', 1)
        var_name = parts[0].strip()
        
        # Remove 'var' prefix if present
        if var_name.startswith('var '):
            var_name = var_name[4:].strip()
        
        if len(parts) > 1:
            value = parts[1].strip()
            
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            # Handle variable references
            if value.startswith('$'):
                ref_var = value[1:]
                if ref_var in self.variables:
                    value = self.variables[ref_var]
                else:
                    value = ""
            
            # Convert numeric values
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            # Store variable
            self.variables[var_name] = value
    
    def _handle_command(self, line: str) -> None:
        """
        Handle a command.
        
        Args:
            line: Command line
        """
        # Handle function calls
        if '(' in line and ')' in line:
            func_call = line.split('(')[0].strip()
            
            if func_call in self.functions:
                self._call_function(line)
                return
        
        # Handle system commands
        parts = shlex.split(line)
        
        if not parts:
            return
        
        command = parts[0]
        args = ' '.join(parts[1:])
        
        # Replace variables in arguments
        for var_name, var_value in self.variables.items():
            args = args.replace(f"${var_name}", str(var_value))
        
        # Handle special commands
        if command == 'echo':
            print(args)
            return
        
        if command == 'exit':
            try:
                self.exit_code = int(args) if args else 0
            except ValueError:
                self.exit_code = 1
            return
        
        # Execute command through registry
        if self.registry:
            handler_info = self.registry.get_command_handler(command)
            
            if handler_info:
                handler = handler_info["handler"]
                self.exit_code = handler(args)
                return
        
        # Unknown command
        logger.error(f"Unknown command: {command}")
        self.exit_code = 1
    
    def _call_function(self, line: str) -> Any:
        """
        Call a function.
        
        Args:
            line: Function call line
            
        Returns:
            Function return value
        """
        # Extract function name and arguments
        match = re.match(r'(\w+)\s*\((.*)\)', line)
        
        if not match:
            logger.error(f"Invalid function call: {line}")
            return None
        
        func_name = match.group(1)
        args_str = match.group(2)
        
        # Parse arguments
        args = []
        if args_str:
            args = [arg.strip() for arg in args_str.split(',')]
        
        # Get function definition
        if func_name not in self.functions:
            logger.error(f"Function not defined: {func_name}")
            return None
        
        func_def = self.functions[func_name]
        params = func_def['params']
        body = func_def['body']
        
        # Create function environment
        old_vars = self.variables.copy()
        
        # Set parameters
        for i, param in enumerate(params):
            if i < len(args):
                self.variables[param] = args[i]
            else:
                self.variables[param] = None
        
        # Execute function body
        for func_line in body:
            func_line = func_line.strip()
            
            if not func_line or func_line.startswith('#'):
                continue
            
            # Handle return
            if func_line.startswith('return '):
                expr = func_line[7:].strip()
                ret_val = self._evaluate_expression(expr)
                
                # Restore variables
                self.variables = old_vars
                
                return ret_val
            
            # Handle assignment
            if '=' in func_line and not func_line.startswith(('if ', 'while ', 'for ')):
                self._handle_assignment(func_line)
            else:
                # Handle command
                self._handle_command(func_line)
        
        # Restore variables
        self.variables = old_vars
        
        return None
    
    def _evaluate_expression(self, expr: str) -> Any:
        """
        Evaluate an expression.
        
        Args:
            expr: Expression
            
        Returns:
            Expression value
        """
        # Handle variable references
        if expr.startswith('$'):
            var_name = expr[1:]
            return self.variables.get(var_name, "")
        
        # Handle simple comparisons
        if ' == ' in expr:
            left, right = expr.split(' == ', 1)
            return self._evaluate_expression(left) == self._evaluate_expression(right)
        
        if ' != ' in expr:
            left, right = expr.split(' != ', 1)
            return self._evaluate_expression(left) != self._evaluate_expression(right)
        
        if ' < ' in expr:
            left, right = expr.split(' < ', 1)
            return self._evaluate_expression(left) < self._evaluate_expression(right)
        
        if ' > ' in expr:
            left, right = expr.split(' > ', 1)
            return self._evaluate_expression(left) > self._evaluate_expression(right)
        
        if ' <= ' in expr:
            left, right = expr.split(' <= ', 1)
            return self._evaluate_expression(left) <= self._evaluate_expression(right)
        
        if ' >= ' in expr:
            left, right = expr.split(' >= ', 1)
            return self._evaluate_expression(left) >= self._evaluate_expression(right)
        
        # Handle logical operators
        if ' and ' in expr:
            left, right = expr.split(' and ', 1)
            return self._evaluate_expression(left) and self._evaluate_expression(right)
        
        if ' or ' in expr:
            left, right = expr.split(' or ', 1)
            return self._evaluate_expression(left) or self._evaluate_expression(right)
        
        if expr.startswith('not '):
            return not self._evaluate_expression(expr[4:])
        
        # Handle literals
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        
        if expr.startswith("'") and expr.endswith("'"):
            return expr[1:-1]
        
        if expr.isdigit():
            return int(expr)
        
        try:
            return float(expr)
        except ValueError:
            pass
        
        # Handle function calls
        if '(' in expr and ')' in expr:
            return self._call_function(expr)
        
        # Handle variable references without $
        if expr in self.variables:
            return self.variables[expr]
        
        # Return as is
        return expr
    
    def _find_block_end(self, lines: List[str], start_idx: int, start_keyword: str, end_keyword: str) -> int:
        """
        Find the end of a block.
        
        Args:
            lines: Script lines
            start_idx: Start index
            start_keyword: Start keyword
            end_keyword: End keyword
            
        Returns:
            End index
        """
        depth = 1
        current_idx = start_idx + 1
        
        while current_idx < len(lines) and depth > 0:
            line = lines[current_idx].strip()
            
            if line.startswith(start_keyword):
                depth += 1
            elif line == end_keyword or line.startswith(f"{end_keyword} "):
                depth -= 1
            
            if depth == 0:
                return current_idx
            
            current_idx += 1
        
        # Return last line if end not found
        return len(lines) - 1
    
    def _find_else(self, lines: List[str], start_idx: int, end_idx: int) -> int:
        """
        Find an else block.
        
        Args:
            lines: Script lines
            start_idx: Start index
            end_idx: End index
            
        Returns:
            Else index or -1 if not found
        """
        current_idx = start_idx + 1
        
        while current_idx < end_idx:
            line = lines[current_idx].strip()
            
            if line == 'else' or line.startswith('else '):
                return current_idx
            
            current_idx += 1
        
        return -1


class ShellScripting:
    """Shell scripting module for FixWurx shell environment."""
    
    def __init__(self):
        """Initialize the shell scripting module."""
        self.registry = None
        self.interpreter = SimpleScriptInterpreter()
    
    def set_registry(self, registry):
        """Set the component registry."""
        self.registry = registry
        self.interpreter.registry = registry
    
    def execute_script(self, script: str) -> int:
        """
        Execute a shell script.
        
        Args:
            script: Shell script
            
        Returns:
            Exit code
        """
        try:
            return self.interpreter.execute(script)
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return 1
    
    def execute_script_file(self, script_file: str) -> int:
        """
        Execute a shell script file.
        
        Args:
            script_file: Shell script file path
            
        Returns:
            Exit code
        """
        try:
            # Read script file
            with open(script_file, 'r') as f:
                script = f.read()
            
            # Execute script
            return self.execute_script(script)
        except FileNotFoundError:
            logger.error(f"Script file not found: {script_file}")
            return 1
        except Exception as e:
            logger.error(f"Error executing script file: {e}")
            return 1


# Command handler for shell scripting
def shell_script_command(args: str) -> int:
    """
    Shell script command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Shell script commands")
    parser.add_argument("action", choices=["run"], 
                        help="Action to perform")
    parser.add_argument("file", help="Script file to run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get registry
    registry = sys.modules.get("__main__").registry
    
    # Get shell scripting module
    shell_scripting = registry.get_component("shell_scripting")
    
    if not shell_scripting:
        print("Error: Shell scripting module not available")
        return 1
    
    # Perform action
    if cmd_args.action == "run":
        if cmd_args.debug:
            logging.getLogger("ShellScripting").setLevel(logging.DEBUG)
        
        return shell_scripting.execute_script_file(cmd_args.file)
    
    else:
        print(f"Unknown action: {cmd_args.action}")
        return 1
