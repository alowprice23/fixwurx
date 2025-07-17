#!/usr/bin/env python3
"""
Response Formatter

This module provides utilities for formatting and presenting responses from
the agent system to the user in a clear, consistent, and visually appealing way.
It follows the conventions specified in the interactive_coding_examples.md.
"""

import os
import sys
import json
import logging
import textwrap
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("response_formatter.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ResponseFormatter")

# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
    "bg_black": "\033[40m",
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
    "bg_white": "\033[47m",
}

# Terminal width for text wrapping
DEFAULT_TERM_WIDTH = 80
TERM_WIDTH = os.get_terminal_size().columns if sys.stdout.isatty() else DEFAULT_TERM_WIDTH

class ResponseFormatter:
    """
    Formats responses from the agent system for presentation to the user.
    
    This class provides methods for formatting various types of content:
    - Plain text
    - Code blocks
    - Tables
    - Lists
    - Command output
    - Errors
    - Warnings
    """
    
    def __init__(self, verbosity: str = "normal", use_colors: bool = True):
        """
        Initialize the response formatter.
        
        Args:
            verbosity: Verbosity level ('concise', 'normal', or 'verbose')
            use_colors: Whether to use ANSI color codes in the output
        """
        self.verbosity = verbosity
        self.use_colors = use_colors and sys.stdout.isatty()
        self.term_width = TERM_WIDTH
        
        logger.info(f"Response Formatter initialized with verbosity={verbosity}, use_colors={use_colors}")
    
    def color(self, text: str, color: str) -> str:
        """
        Add color to text if colors are enabled.
        
        Args:
            text: Text to colorize
            color: Color name from the COLORS dictionary
            
        Returns:
            Colorized text
        """
        if not self.use_colors:
            return text
        
        color_code = COLORS.get(color, "")
        reset_code = COLORS["reset"]
        
        return f"{color_code}{text}{reset_code}"
    
    def wrap_text(self, text: str, width: Optional[int] = None, indent: int = 0) -> str:
        """
        Wrap text to the specified width.
        
        Args:
            text: Text to wrap
            width: Width to wrap to (defaults to terminal width)
            indent: Number of spaces to indent each line
            
        Returns:
            Wrapped text
        """
        width = width or self.term_width
        indentation = " " * indent
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=indentation,
            subsequent_indent=indentation,
            break_long_words=False,
            break_on_hyphens=False
        )
        
        # Split by lines, then wrap each line
        lines = text.split("\n")
        wrapped_lines = []
        
        for line in lines:
            if not line.strip():
                wrapped_lines.append("")
            else:
                wrapped_lines.extend(wrapper.wrap(line))
        
        return "\n".join(wrapped_lines)
    
    def format_text(self, text: str, wrap: bool = True) -> str:
        """
        Format plain text.
        
        Args:
            text: Text to format
            wrap: Whether to wrap the text
            
        Returns:
            Formatted text
        """
        if wrap:
            return self.wrap_text(text)
        return text
    
    def format_code_block(self, code: str, language: str = "") -> str:
        """
        Format a code block.
        
        Args:
            code: Code to format
            language: Programming language for syntax highlighting
            
        Returns:
            Formatted code block
        """
        header = self.color(f"```{language}", "bright_cyan")
        footer = self.color("```", "bright_cyan")
        
        if self.use_colors:
            # Basic syntax highlighting for some common languages
            if language in ["python", "py"]:
                code = self._highlight_python(code)
            elif language in ["javascript", "js"]:
                code = self._highlight_javascript(code)
            elif language in ["bash", "sh"]:
                code = self._highlight_bash(code)
        
        return f"\n{header}\n{code}\n{footer}\n"
    
    def format_table(self, headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
        """
        Format a table.
        
        Args:
            headers: List of column headers
            rows: List of rows, where each row is a list of strings
            title: Optional table title
            
        Returns:
            Formatted table
        """
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Create header row
        header_row = " | ".join(
            self.color(header.ljust(col_widths[i]), "bold")
            for i, header in enumerate(headers)
        )
        
        # Create separator row
        separator = "-+-".join("-" * width for width in col_widths)
        
        # Create data rows
        data_rows = []
        for row in rows:
            data_row = " | ".join(
                str(cell).ljust(col_widths[i]) if i < len(col_widths) else str(cell)
                for i, cell in enumerate(row)
            )
            data_rows.append(data_row)
        
        # Assemble table
        table = []
        if title:
            table.append(self.color(title, "bright_green"))
            table.append("")
        
        table.append(header_row)
        table.append(separator)
        table.extend(data_rows)
        
        return "\n".join(table)
    
    def format_list(self, items: List[str], ordered: bool = False, nested_level: int = 0) -> str:
        """
        Format a list.
        
        Args:
            items: List of items
            ordered: Whether the list is ordered or unordered
            nested_level: Nesting level for indentation
            
        Returns:
            Formatted list
        """
        result = []
        indent = 2 * nested_level
        
        for i, item in enumerate(items):
            if ordered:
                bullet = f"{i+1}."
            else:
                bullet = "-"
            
            # Handle nested lists (item can contain newlines with nested lists)
            item_lines = item.split("\n")
            first_line = item_lines[0]
            
            result.append(f"{' ' * indent}{bullet} {first_line}")
            
            # Add remaining lines with proper indentation
            if len(item_lines) > 1:
                for line in item_lines[1:]:
                    result.append(f"{' ' * indent}  {line}")
        
        return "\n".join(result)
    
    def format_command_output(self, output: str, command: str, exit_code: int = 0, summary: bool = True) -> str:
        """
        Format command output.
        
        Args:
            output: Command output
            command: The command that was executed
            exit_code: Command exit code
            summary: Whether to summarize the output based on verbosity
            
        Returns:
            Formatted command output
        """
        # Determine how much of the output to show based on verbosity
        if summary and self.verbosity == "concise" and len(output.splitlines()) > 5:
            # For concise mode, show only first and last two lines
            lines = output.splitlines()
            if len(lines) > 5:
                shown_lines = lines[:2] + ["..."] + lines[-2:]
                output = "\n".join(shown_lines)
        
        # Format header
        header = self.color(f"$ {command}", "bright_green")
        
        # Format exit code
        if exit_code == 0:
            exit_status = self.color(f"(Exit: {exit_code})", "green")
        else:
            exit_status = self.color(f"(Exit: {exit_code})", "red")
        
        # Format output
        formatted_output = f"\n{header} {exit_status}\n"
        
        if output:
            if self.use_colors:
                # Simple output highlighting
                lines = []
                for line in output.splitlines():
                    if line.startswith(("ERROR", "Error", "error")):
                        lines.append(self.color(line, "red"))
                    elif line.startswith(("WARNING", "Warning", "warning")):
                        lines.append(self.color(line, "yellow"))
                    elif line.startswith(("SUCCESS", "Success", "success")):
                        lines.append(self.color(line, "green"))
                    else:
                        lines.append(line)
                
                output = "\n".join(lines)
            
            formatted_output += f"{output}\n"
        
        return formatted_output
    
    def format_error(self, message: str, details: Optional[str] = None) -> str:
        """
        Format an error message.
        
        Args:
            message: Error message
            details: Optional error details
            
        Returns:
            Formatted error message
        """
        error_msg = self.color(f"ERROR: {message}", "red")
        
        if details and self.verbosity != "concise":
            error_msg += f"\n\n{details}"
        
        return error_msg
    
    def format_warning(self, message: str, details: Optional[str] = None) -> str:
        """
        Format a warning message.
        
        Args:
            message: Warning message
            details: Optional warning details
            
        Returns:
            Formatted warning message
        """
        warning_msg = self.color(f"WARNING: {message}", "yellow")
        
        if details and self.verbosity != "concise":
            warning_msg += f"\n\n{details}"
        
        return warning_msg
    
    def format_success(self, message: str, details: Optional[str] = None) -> str:
        """
        Format a success message.
        
        Args:
            message: Success message
            details: Optional success details
            
        Returns:
            Formatted success message
        """
        success_msg = self.color(f"SUCCESS: {message}", "green")
        
        if details and self.verbosity != "concise":
            success_msg += f"\n\n{details}"
        
        return success_msg
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """
        Format a complete response from the agent system.
        
        Args:
            response: Response dictionary with various sections
            
        Returns:
            Formatted response
        """
        sections = []
        
        # Add main message
        if "message" in response:
            sections.append(self.format_text(response["message"]))
        
        # Add code blocks
        if "code_blocks" in response:
            for code_block in response["code_blocks"]:
                language = code_block.get("language", "")
                code = code_block.get("code", "")
                sections.append(self.format_code_block(code, language))
        
        # Add tables
        if "tables" in response:
            for table in response["tables"]:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                title = table.get("title")
                sections.append(self.format_table(headers, rows, title))
        
        # Add lists
        if "lists" in response:
            for list_item in response["lists"]:
                items = list_item.get("items", [])
                ordered = list_item.get("ordered", False)
                sections.append(self.format_list(items, ordered))
        
        # Add command outputs
        if "command_outputs" in response:
            for cmd_output in response["command_outputs"]:
                output = cmd_output.get("output", "")
                command = cmd_output.get("command", "")
                exit_code = cmd_output.get("exit_code", 0)
                sections.append(self.format_command_output(output, command, exit_code))
        
        # Add errors
        if "errors" in response:
            for error in response["errors"]:
                message = error.get("message", "")
                details = error.get("details")
                sections.append(self.format_error(message, details))
        
        # Add warnings
        if "warnings" in response:
            for warning in response["warnings"]:
                message = warning.get("message", "")
                details = warning.get("details")
                sections.append(self.format_warning(message, details))
        
        return "\n\n".join(sections)
    
    def _highlight_python(self, code: str) -> str:
        """
        Apply basic syntax highlighting for Python code.
        
        Args:
            code: Python code
            
        Returns:
            Highlighted code
        """
        # Very simple syntax highlighting
        keywords = [
            "import", "from", "def", "class", "if", "elif", "else", "for", "while",
            "try", "except", "finally", "with", "as", "return", "yield", "break",
            "continue", "pass", "assert", "raise", "in", "is", "not", "and", "or",
            "True", "False", "None", "lambda", "global", "nonlocal", "async", "await"
        ]
        
        lines = []
        for line in code.splitlines():
            # Comments
            if "#" in line:
                comment_start = line.find("#")
                comment = line[comment_start:]
                line = line[:comment_start] + self.color(comment, "dim")
            
            # Strings
            line = self._highlight_strings(line)
            
            # Keywords
            for keyword in keywords:
                # Only match whole words
                line = line.replace(f" {keyword} ", f" {self.color(keyword, 'bright_yellow')} ")
                line = line.replace(f" {keyword}:", f" {self.color(keyword, 'bright_yellow')}:")
                line = line.replace(f" {keyword}(", f" {self.color(keyword, 'bright_yellow')}(")
                
                # Handle keyword at start of line
                if line.startswith(f"{keyword} "):
                    line = f"{self.color(keyword, 'bright_yellow')} {line[len(keyword)+1:]}"
                if line.startswith(f"{keyword}:"):
                    line = f"{self.color(keyword, 'bright_yellow')}:{line[len(keyword)+1:]}"
                if line.startswith(f"{keyword}("):
                    line = f"{self.color(keyword, 'bright_yellow')}{line[len(keyword):]}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _highlight_javascript(self, code: str) -> str:
        """
        Apply basic syntax highlighting for JavaScript code.
        
        Args:
            code: JavaScript code
            
        Returns:
            Highlighted code
        """
        # Very simple syntax highlighting
        keywords = [
            "var", "let", "const", "function", "class", "if", "else", "for", "while",
            "try", "catch", "finally", "with", "return", "break", "continue", "switch",
            "case", "default", "throw", "typeof", "instanceof", "in", "of", "new", "this",
            "super", "extends", "async", "await", "import", "export", "from", "as",
            "true", "false", "null", "undefined"
        ]
        
        lines = []
        for line in code.splitlines():
            # Comments
            if "//" in line:
                comment_start = line.find("//")
                comment = line[comment_start:]
                line = line[:comment_start] + self.color(comment, "dim")
            
            # Strings
            line = self._highlight_strings(line)
            
            # Keywords
            for keyword in keywords:
                # Only match whole words
                line = line.replace(f" {keyword} ", f" {self.color(keyword, 'bright_yellow')} ")
                line = line.replace(f" {keyword}:", f" {self.color(keyword, 'bright_yellow')}:")
                line = line.replace(f" {keyword}(", f" {self.color(keyword, 'bright_yellow')}(")
                
                # Handle keyword at start of line
                if line.startswith(f"{keyword} "):
                    line = f"{self.color(keyword, 'bright_yellow')} {line[len(keyword)+1:]}"
                if line.startswith(f"{keyword}:"):
                    line = f"{self.color(keyword, 'bright_yellow')}:{line[len(keyword)+1:]}"
                if line.startswith(f"{keyword}("):
                    line = f"{self.color(keyword, 'bright_yellow')}{line[len(keyword):]}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _highlight_bash(self, code: str) -> str:
        """
        Apply basic syntax highlighting for Bash code.
        
        Args:
            code: Bash code
            
        Returns:
            Highlighted code
        """
        # Very simple syntax highlighting
        keywords = [
            "if", "then", "else", "elif", "fi", "for", "in", "do", "done",
            "while", "until", "case", "esac", "function", "return", "exit",
            "export", "local", "readonly", "source", "alias", "unalias",
            "true", "false", "echo", "read", "set", "unset", "shift"
        ]
        
        lines = []
        for line in code.splitlines():
            # Comments
            if "#" in line:
                comment_start = line.find("#")
                comment = line[comment_start:]
                line = line[:comment_start] + self.color(comment, "dim")
            
            # Strings
            line = self._highlight_strings(line)
            
            # Keywords
            for keyword in keywords:
                # Only match whole words
                line = line.replace(f" {keyword} ", f" {self.color(keyword, 'bright_yellow')} ")
                line = line.replace(f" {keyword};", f" {self.color(keyword, 'bright_yellow')};")
                
                # Handle keyword at start of line
                if line.startswith(f"{keyword} "):
                    line = f"{self.color(keyword, 'bright_yellow')} {line[len(keyword)+1:]}"
                if line.startswith(f"{keyword};"):
                    line = f"{self.color(keyword, 'bright_yellow')};{line[len(keyword)+1:]}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _highlight_strings(self, line: str) -> str:
        """
        Highlight string literals in a line of code.
        
        Args:
            line: Line of code
            
        Returns:
            Line with highlighted strings
        """
        # This is a very simple approach and won't handle all cases correctly,
        # especially nested quotes, escaped quotes, etc.
        
        # Single quotes
        in_single_quote = False
        new_line = ""
        for i, char in enumerate(line):
            if char == "'" and (i == 0 or line[i-1] != "\\"):
                in_single_quote = not in_single_quote
                if in_single_quote:
                    new_line += self.color("'", "bright_green")
                else:
                    new_line += self.color("'", "bright_green")
            elif in_single_quote:
                new_line += self.color(char, "bright_green")
            else:
                new_line += char
        
        line = new_line
        
        # Double quotes
        in_double_quote = False
        new_line = ""
        for i, char in enumerate(line):
            if char == '"' and (i == 0 or line[i-1] != "\\"):
                in_double_quote = not in_double_quote
                if in_double_quote:
                    new_line += self.color('"', "bright_green")
                else:
                    new_line += self.color('"', "bright_green")
            elif in_double_quote:
                new_line += self.color(char, "bright_green")
            else:
                new_line += char
        
        return new_line


# Singleton instance
_instance = None

def get_instance(verbosity: str = "normal", use_colors: bool = True) -> ResponseFormatter:
    """
    Get the singleton instance of the response formatter.
    
    Args:
        verbosity: Verbosity level ('concise', 'normal', or 'verbose')
        use_colors: Whether to use ANSI color codes in the output
        
    Returns:
        ResponseFormatter instance
    """
    global _instance
    if _instance is None or _instance.verbosity != verbosity or _instance.use_colors != use_colors:
        _instance = ResponseFormatter(verbosity, use_colors)
    return _instance


# Example usage
if __name__ == "__main__":
    formatter = get_instance(verbosity="normal", use_colors=True)
    
    # Example text
    print(formatter.format_text("This is a simple text message that will be wrapped to the terminal width."))
    
    # Example code block
    print(formatter.format_code_block(
        "def hello_world():\n    print('Hello, world!')\n\nhello_world()",
        language="python"
    ))
    
    # Example table
    print(formatter.format_table(
        headers=["Name", "Age", "Occupation"],
        rows=[
            ["Alice", "32", "Engineer"],
            ["Bob", "28", "Designer"],
            ["Charlie", "45", "Manager"]
        ],
        title="Employee List"
    ))
    
    # Example list
    print(formatter.format_list(
        items=["First item", "Second item", "Third item with\nmultiple lines"],
        ordered=True
    ))
    
    # Example command output
    print(formatter.format_command_output(
        output="Hello, world!\nThis is the output of a command.",
        command="echo 'Hello, world!'",
        exit_code=0
    ))
    
    # Example error
    print(formatter.format_error(
        message="File not found",
        details="The file 'missing.txt' could not be found in the current directory."
    ))
    
    # Example warning
    print(formatter.format_warning(
        message="Deprecated function used",
        details="The function 'old_function()' is deprecated and will be removed in a future version."
    ))
    
    # Example success
    print(formatter.format_success(
        message="Operation completed successfully",
        details="All 42 files were processed in 3.5 seconds."
    ))
    
    # Example complete response
    print(formatter.format_response({
        "message": "Here's the result of your request:",
        "code_blocks": [
            {
                "language": "python",
                "code": "def hello_world():\n    print('Hello, world!')\n\nhello_world()"
            }
        ],
        "tables": [
            {
                "headers": ["Name", "Value"],
                "rows": [["foo", "42"], ["bar", "99"]],
                "title": "Configuration"
            }
        ],
        "command_outputs": [
            {
                "command": "ls -la",
                "output": "total 12\ndrwxr-xr-x 2 user user 4096 Jul 15 14:30 .\ndrwxr-xr-x 4 user user 4096 Jul 15 14:29 ..",
                "exit_code": 0
            }
        ]
    }))
