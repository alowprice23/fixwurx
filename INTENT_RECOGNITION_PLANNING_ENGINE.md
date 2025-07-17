# Intent Recognition and Planning Engine Implementation

This document outlines the implementation of the Intent Recognition and Planning Engine (IRPE) components from the LLM Shell Integration Plan v4.

## Overview

The Intent Recognition and Planning Engine translates high-level user goals into concrete, executable plans or scripts. It combines deterministic decision trees with LLM-based generative planning, providing a hybrid approach to planning that is both efficient and flexible.

## Components Implemented

### 1. Planning Engine (`planning_engine.py`)

The core planning engine with the following features:

- **Hybrid Planning Strategy**: Attempts to find matches in the decision tree before falling back to LLM-based generative planning
- **Goal Deconstruction**: Breaks down high-level goals into logical sequences of steps
- **Script Generation**: Converts steps into executable `.fx` scripts
- **Command Validation**: Validates generated scripts against the command lexicon
- **Script Library Management**: Stores, retrieves, and manages reusable scripts

### 2. LLM Client (`llm_client.py`)

A mock LLM client for testing purposes:

- **Text Generation**: Simulates LLM responses for goal deconstruction and script generation
- **Pattern Matching**: Returns different predefined responses based on the prompt type
- **Chat Interface**: Provides a chat completion interface compatible with standard LLM APIs

### 3. Testing Infrastructure (`test_planning_engine.py`)

Comprehensive test suite for the planning engine:

- **Goal Deconstruction Tests**: Tests the ability to break down user goals into steps
- **Script Generation Tests**: Tests the generation of scripts from decomposed steps
- **Script Library Tests**: Tests adding, retrieving, and listing scripts in the library

## Key Features

1. **Two-Stage Planning Process**: First decompose the goal into steps, then generate a script
2. **Script Validation**: Validate generated scripts against the command lexicon to catch LLM hallucinations
3. **Script Library**: Reuse previously successful scripts for similar tasks
4. **Script Fixing**: Attempt to fix scripts that fail validation
5. **Command Lexicon Integration**: Parse and utilize the command lexicon for validation

## Implementation Details

### Goal Deconstruction

```python
def _deconstruct_goal(self, goal: str, context: Dict[str, Any]) -> List[str]:
    """
    Deconstruct a user goal into a logical sequence of steps.
    """
    # Prepare the prompt for goal deconstruction
    prompt = self._create_goal_deconstruction_prompt(goal, context)
    
    # Call the LLM to deconstruct the goal
    response = self.llm_client.generate(prompt, temperature=0.2)
    
    # Extract steps from the response
    steps = self._extract_steps_from_response(response)
    
    return steps
```

### Script Generation

```python
def _generate_fx_script(self, goal: str, steps: List[str], context: Dict[str, Any]) -> str:
    """
    Generate an .fx script from the goal and steps.
    """
    # Prepare the prompt for script generation
    prompt = self._create_script_generation_prompt(goal, steps, context)
    
    # Call the LLM to generate the script
    response = self.llm_client.generate(prompt, temperature=0.2)
    
    # Extract the script from the response
    script = self._extract_script_from_response(response)
    
    return script
```

### Command Validation

```python
def _validate_script(self, script: str) -> Dict[str, Any]:
    """
    Validate a script against the command lexicon.
    """
    # Parse the script line by line
    lines = script.split('\n')
    issues = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Check if the line contains a valid command
        if not self._is_valid_command(line):
            issues.append({
                "line": i + 1,
                "content": line,
                "issue": "Invalid command or syntax"
            })
    
    valid = len(issues) == 0
    
    return {
        "valid": valid,
        "issues": issues
    }
```

### Script Library Management

```python
def add_script_to_library(self, script: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Add a script to the script library.
    """
    # Create a unique ID for the script
    timestamp = int(time.time())
    script_id = f"script_{timestamp}"
    
    # Extract existing metadata from the script
    existing_metadata = self._extract_metadata_from_script(script)
    
    # Merge with provided metadata
    if metadata:
        for key, value in metadata.items():
            existing_metadata[key] = value
    
    # Format the metadata as comments
    metadata_comments = [
        "### BEGIN METADATA ###",
        f"# Description: {existing_metadata.get('description', '')}",
        f"# Author: {existing_metadata.get('author', '')}",
        f"# Version: {existing_metadata.get('version', '1.0')}",
        f"# Tags: {', '.join(existing_metadata.get('tags', []))}",
        "### END METADATA ###",
        ""
    ]
    
    # Add metadata to the script
    script_with_metadata = "\n".join(metadata_comments) + script
    
    # Write the script to the file
    with open(script_path, 'w') as f:
        f.write(script_with_metadata)
    
    return {
        "success": True,
        "script_id": script_id,
        "path": script_path,
        "metadata": existing_metadata
    }
```

## Integration with Other Components

- **Decision Tree Integration**: The planning engine can utilize the decision tree system if available
- **Command Lexicon Integration**: The planning engine parses and uses the command lexicon for validation
- **Shell Environment Integration**: Generated scripts are designed to be executed by the shell environment

## Test Results

The tests verified the following functionality:

1. **Goal Deconstruction**: Successfully breaks down goals into logical steps
2. **Script Generation**: Generates scripts based on decomposed steps
3. **Script Library Management**: Correctly adds, retrieves, and lists scripts

The script validation tests identified an important area for improvement: we need to update the command lexicon to include basic shell commands (like `echo`, `mkdir`, etc.) in addition to the FixWurx-specific commands.

## Future Enhancements

1. **Decision Tree Growth**: Add logic to expand the decision tree based on successful planning
2. **Script Tagging and Categorization**: Enhance the script library with better tagging and search
3. **Context-Aware Planning**: Improve planning based on user context and history
4. **Performance Optimization**: Implement caching for common goal patterns
5. **Plan Visualization**: Add tools to visualize the planning process for better transparency
