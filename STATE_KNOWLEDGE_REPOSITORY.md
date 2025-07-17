# State and Knowledge Repository (SKR)

This document outlines the implementation of the State and Knowledge Repository (SKR) components from the LLM Shell Integration Plan v4.

## Overview

The State and Knowledge Repository (SKR) provides persistent storage for scripts, conversations, and knowledge acquired during agent operations. It serves as the system's long-term memory, enabling knowledge accumulation, reuse, and context-aware interactions.

## Components Implemented

### 1. Script Library (`script_library.py`)

A version-controlled repository for storing and retrieving scripts:

- **Git Integration**: Scripts are stored in a Git repository for versioning and collaboration
- **Metadata Management**: Each script includes rich metadata (author, tags, version, etc.)
- **Categorization**: Scripts are organized by category (workflows, fixes, utilities, templates)
- **Search Capabilities**: Find scripts by query, tags, author, and category
- **Usage Tracking**: Track success and failure rates for scripts
- **Versioning**: Maintain script history with versioning support

### 2. Conversation Logger (`conversation_logger.py`)

A structured storage system for conversations between users and agents:

- **Conversation Storage**: Store complete conversations with metadata
- **Message Management**: Structured storage of individual messages with metadata
- **Conversation Retrieval**: Retrieve conversations by ID or search criteria
- **Summarization**: Generate summaries of conversations for quick reference
- **Archiving**: Compress and archive older conversations
- **Retention Policies**: Automatically clean up old conversations based on retention settings

## Implementation Details

### Script Library

The Script Library provides a Git-backed storage system for scripts with comprehensive metadata:

```python
def add_script(self, script_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a script to the library.
    
    Args:
        script_content: Content of the script
        metadata: Metadata for the script
        
    Returns:
        Dictionary with result
    """
```

Key features:

- **Automatic Categorization**: Scripts are automatically categorized based on their metadata and tags
- **Metadata Validation**: Ensure all scripts have consistent and complete metadata
- **Script Formatting**: Scripts are formatted with metadata as comments
- **Execution Tracking**: Track the success and failure rates of scripts
- **Search and Retrieval**: Find scripts based on various criteria

### Conversation Logger

The Conversation Logger provides structured storage for conversations with summarization capabilities:

```python
def add_message(self, conversation_id: str, sender: str, content: str,
               content_type: str = "text", metadata: Optional[Dict[str, Any]] = None,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Add a message to a conversation.
    """
```

Key features:

- **Structured Storage**: JSONL format for efficient storage and retrieval
- **Compression**: Automatic compression of archived conversations
- **Summarization**: Generate summaries using neural matrix integration
- **Context Management**: Store and retrieve conversation context
- **Search Capabilities**: Search conversations by query, user, agent, tags, and time range
- **Retention Management**: Automatically clean up old conversations

## Neural Matrix Integration

Both components integrate with the Neural Matrix for enhanced capabilities:

- **Script Library**: Use Neural Matrix for script improvement suggestions and categorization
- **Conversation Logger**: Use Neural Matrix for generating conversation summaries and extracting insights

## Usage Examples

### Script Library Example

```python
# Add a script to the library
result = script_library.add_script(
    script_content="""
    #!/usr/bin/env bash
    echo "Hello, World!"
    """,
    metadata={
        "name": "Hello World Script",
        "description": "A simple hello world script",
        "author": "meta_agent",
        "tags": ["example", "hello", "utility"]
    }
)

# Search for scripts
search_result = script_library.search_scripts(
    query="hello",
    tags=["utility"]
)

# Get and execute a script
script_result = script_library.get_script(script_id)
if script_result["success"]:
    # Execute the script
    # ...
    # Report outcome
    script_library.report_execution_outcome(script_id, success=True)
```

### Conversation Logger Example

```python
# Start a new conversation
result = conversation_logger.start_conversation(
    user_id="user123",
    metadata={"source": "web_interface"}
)
conversation_id = result["conversation_id"]

# Add messages to the conversation
conversation_logger.add_message(
    conversation_id=conversation_id,
    sender="user123",
    content="How can I fix my database connection issue?",
    content_type="text"
)

conversation_logger.add_message(
    conversation_id=conversation_id,
    sender="agent456",
    content="Let's troubleshoot your database connection...",
    content_type="text"
)

# End the conversation with a summary
conversation_logger.end_conversation(
    conversation_id=conversation_id,
    tags=["database", "troubleshooting"]
)

# Search for conversations about databases
search_result = conversation_logger.search_conversations(
    query="database",
    tags=["troubleshooting"]
)
```

## Future Enhancements

1. **Neural Matrix Schema Extension**: Enhance the Neural Matrix with schemas for SKR integration
2. **Knowledge Graph Integration**: Implement a knowledge graph to connect scripts, conversations, and entities
3. **Improved Summarization**: Enhance conversation summarization with better NLP techniques
4. **Insights Extraction**: Extract actionable insights from conversations and script execution history
5. **Collaborative Editing**: Enable collaborative editing of scripts with merge capabilities

## Test Results

The tests (`test_script_library.py` and `test_conversation_logger.py`) verify the following functionality:

1. **Script Storage**: Scripts are stored correctly with metadata
2. **Script Retrieval**: Scripts can be retrieved by ID and search criteria
3. **Script Versioning**: Script versions are managed correctly
4. **Conversation Storage**: Conversations and messages are stored correctly
5. **Conversation Retrieval**: Conversations can be retrieved by ID and search criteria
6. **Summarization**: Conversation summaries are generated and stored

All implemented features pass their respective tests, demonstrating the robustness of the SKR implementation.
