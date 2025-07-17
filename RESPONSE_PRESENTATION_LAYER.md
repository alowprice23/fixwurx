# Response Presentation Layer Implementation

This document outlines the implementation of Task CI-3: Develop the Response Presentation Layer from the LLM Shell Integration Plan v4.

## Overview

The Response Presentation Layer is responsible for formatting and displaying responses from the agent system to the user in a clear, consistent, and visually appealing way. It supports multiple content types and verbosity levels to enhance the user experience.

## Components Implemented

### 1. Response Formatter (`response_formatter.py`)

The core formatting engine that handles different types of content:

- **Text formatting**: Clean presentation of plain text with proper wrapping
- **Code block formatting**: Syntax highlighting for multiple languages
- **Table formatting**: Structured display of tabular data
- **List formatting**: Both ordered and unordered lists
- **Command output formatting**: Terminal command results with exit codes
- **Error and warning formatting**: Distinctive styling for errors and warnings
- **Structured response formatting**: Complex responses with multiple sections

### 2. Verbosity Control

Three levels of verbosity to control the amount of detail shown:

- **Concise**: Shows minimal information, summarizes long outputs
- **Normal**: Balanced level of detail, suitable for most interactions
- **Verbose**: Shows all available information for in-depth analysis

### 3. Integration with Conversational Interface

Updated the Conversational Interface to use the Response Formatter:

- Automatic detection of content types in responses
- Proper formatting of code blocks with language-specific highlighting
- Special handling for error messages
- Support for changing verbosity on-the-fly

## Key Features

1. **Syntax Highlighting**: Code blocks are highlighted based on the programming language
2. **ANSI Color Support**: Terminal colors enhance readability (when supported)
3. **Adaptive Output**: Content formatting adapts to verbosity level
4. **Command Output Processing**: Command outputs show command, exit code, and formatted output
5. **Error Highlighting**: Errors and warnings are visually distinct
6. **Table Formatting**: Clean presentation of tabular data

## Implementation Details

### Singleton Pattern

The formatter uses a singleton pattern for efficient reuse:

```python
def get_instance(verbosity: str = "normal", use_colors: bool = True) -> ResponseFormatter:
    global _instance
    if _instance is None or _instance.verbosity != verbosity or _instance.use_colors != use_colors:
        _instance = ResponseFormatter(verbosity, use_colors)
    return _instance
```

### Terminal Width Detection

Automatically adapts to the terminal width for optimal text wrapping:

```python
TERM_WIDTH = os.get_terminal_size().columns if sys.stdout.isatty() else DEFAULT_TERM_WIDTH
```

### Smart Content Processing

The display_response method in ConversationalInterface intelligently processes different content types:

```python
def display_response(self, response: str) -> None:
    # Use response formatter to format the response
    if "```" in response:
        # Response contains code blocks, extract and format them
        # ...
    elif response.startswith("Error:") or response.startswith("I'm sorry"):
        # Error message handling
        # ...
    else:
        # Regular response
        print(self.formatter.format_text(response))
```

## Testing

Comprehensive test suite implemented in `test_response_formatter.py` to verify:
- Correct formatting of all content types
- Proper behavior across different verbosity levels
- Integration with the conversational interface
- Color handling and syntax highlighting

## Usage Example

```python
# Get a formatter instance
formatter = get_response_formatter(verbosity="normal")

# Format and display a code block
code = "def hello_world():\n    print('Hello, world!')"
formatted = formatter.format_code_block(code, "python")
print(formatted)

# Format an error message
error_msg = formatter.format_error("File not found", "The file 'config.json' could not be found")
print(error_msg)
```

## Future Enhancements

1. Add support for more content types (e.g., images, graphs)
2. Enhance syntax highlighting for additional languages
3. Implement theme support for consistent styling
4. Add markdown rendering capabilities
