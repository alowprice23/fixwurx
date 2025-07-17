# Conversation Logging System

## Overview

The Conversation Logging System provides a comprehensive framework for logging, storing, and analyzing all interactions between agents, users, and LLMs within the FixWurx environment. It ensures that all communication is properly recorded and can be analyzed for insights, debugging, and auditing purposes.

## Key Features

- **Complete Conversation Tracking**: Logs all user-agent, agent-agent, and agent-LLM interactions
- **Structured Storage**: Stores conversations in well-organized JSON files with daily rolling logs
- **Search Capabilities**: Allows searching through conversation history
- **LLM-Powered Analysis**: Leverages the Auditor Agent's LLM capabilities to analyze conversations
- **Shell Integration**: Seamlessly integrates with the shell environment's command system
- **Retention Policy**: Configurable retention period for conversation data

## Components

The system consists of three primary components:

1. **`agent_conversation_logger.py`**: Core logging engine that handles recording and storing conversations
2. **`conversation_commands.py`**: Shell commands for interacting with the conversation logs
3. **`agent_shell_integration.py`**: Integration with the shell environment and command wrapping

## Architecture

The conversation logger uses a singleton pattern to ensure a single instance is shared across the system. It automatically connects with the Auditor Agent to provide advanced analysis capabilities for conversations.

Storage is organized into:
- **Daily Log Files**: JSONL files containing all conversations for a given day
- **Session Files**: Individual JSON files for each conversation session
- **.triangulum/conversations**: Default storage location (configurable)

## Usage Examples

### Shell Commands

```
# List recent conversations
fx> conversation:list
fx> convs  # Alias

# Show details of a specific conversation
fx> conversation:show <session_id>
fx> conv <session_id>  # Alias

# Search for specific content in conversations
fx> conversation:search "error message"

# Analyze a conversation with LLM
fx> conversation:analyze <session_id>

# Clean up old conversations
fx> conversation:clean
```

### Programmatic Usage

```python
# Get the conversation logger instance
from agent_conversation_logger import get_instance
logger = get_instance()

# Log a user message
session_id = logger.log_user_message(
    user_input="Fix this bug",
    command="bug:fix",
    agent_id="planner"
)

# Log an agent response
logger.log_agent_response(
    session_id=session_id,
    agent_id="planner",
    response={"status": "success", "plan": "..."},
    success=True,
    llm_used=True
)

# Log agent-to-agent communication
logger.log_agent_to_agent(
    source_agent_id="meta",
    target_agent_id="planner",
    message={"action": "generate_plan", "bug_id": "123"}
)

# Log LLM interaction
logger.log_llm_interaction(
    agent_id="meta",
    prompt="Generate a plan...",
    response="Here's a plan..."
)
```

## LLM Integration

The conversation logging system integrates deeply with the LLM capabilities:

1. **Direct Logging**: All LLM interactions (prompts and responses) are logged
2. **Command Detection**: Automatically detects LLM usage in agent commands
3. **Conversation Analysis**: Uses the Auditor Agent's LLM to analyze conversations
4. **Context Preservation**: Maintains full context for complex multi-turn conversations

## Benefits

- **Auditability**: Complete record of all system interactions
- **Debugging**: Easily trace issues through conversation history
- **Learning**: Analyze conversations to improve agent performance
- **Transparency**: Clear visibility into all agent operations
- **Security**: Track all command executions and responses

## Configuration

The conversation logger can be configured with:

```python
config = {
    "enabled": True,                          # Enable/disable logging
    "storage_path": ".triangulum/conversations", # Storage location
    "retention_days": 30                      # Data retention period
}
```

## Integration with Auditor Agent

When a conversation session is closed, the system automatically triggers an analysis using the Auditor Agent's LLM capabilities. This analysis examines patterns, identifies issues, and provides insights about the conversation.

## Session Data Structure

Each conversation session contains structured message data:

```json
{
  "timestamp": 1625123456.789,
  "session_id": "session_1625123456_agent",
  "direction": "user_to_agent",
  "user_input": "What's the status?",
  "command": "agent:status",
  "target_agent": "agent",
  "message_type": "command"
}
```

## Testing

A test script (`test_conversation_logging.py`) is provided to verify the functionality of the conversation logging system. Run it with:

```
python test_conversation_logging.py
