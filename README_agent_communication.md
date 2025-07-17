# Agent Communication System

## Overview

The Agent Communication System enables direct agent-to-user communication in the FixWurx shell environment. It provides a rich, interactive experience where agents can speak directly to users, report progress on long-running tasks, and collaborate with each other to solve complex problems.

## Components

1. **Agent Communication System** (`agent_communication_system.py`)
   - Direct agent-to-user communication with rich formatting
   - Support for multiple message types (info, success, warning, error)
   - Colored and emoji-enhanced outputs for better readability
   - Session-based conversation tracking

2. **Progress Tracking System** (`agent_progress_tracking.py`)
   - Real-time progress reporting for long-running tasks
   - Estimated completion time calculations
   - Support for concurrent task tracking across multiple agents
   - Task timeout detection and stalled task warnings

3. **Conversation Logging** (`agent_conversation_logger.py`)
   - Logs all agent-user interactions
   - Provides searchable conversation history
   - Enables analysis of past conversations

4. **Shell Integration** (`agent_shell_integration.py`)
   - Seamless integration with existing shell environment
   - Automatic agent selection based on command context
   - Command wrapping to inject communication capabilities

## Quick Start

### Basic Agent Communication

Make an agent speak directly to the user:

```bash
agent:speak launchpad "Starting system initialization"
agent:speak triangulum "Analysis complete" -t success
agent:speak auditor "Missing configuration file" -t warning
agent:speak neural_matrix "Training failed" -t error
```

### Progress Tracking

Track progress of a long-running task:

```bash
# Start tracking a task
tracker_id=$(agent:progress start --agent triangulum --task "analysis" --description "Code analysis" --steps 5)

# Update progress
agent:progress update --tracker $tracker_id --step 3 --message "Halfway done"

# Complete the task
agent:progress complete --tracker $tracker_id --success
```

### Conversation Logging

View and search past conversations:

```bash
# List recent conversations
conversation:list

# Search for specific content
conversation:search "error"

# View a specific conversation
conversation:show session_1234567890
```

## Demonstration

For a complete demonstration of all features, run:

```bash
./agent_communication_demo.fx
```

This will showcase all aspects of the agent communication system, including:
- Direct agent communication with different message types
- Progress tracking for a multi-step task
- Multi-agent collaboration

## Working with Documentation Examples

When copying code examples from documentation, you may encounter errors with Markdown code fences (like \`\`\` at the beginning and end of code blocks). The included `shell_script_parser.py` utility helps resolve these issues:

```bash
# Parse and execute a script containing Markdown code blocks
python shell_script_parser.py script_with_markdown.fx

# Just print the processed script without executing
python shell_script_parser.py script_with_markdown.fx --print
```

## Available Agents

The following agents are registered by default:

1. **launchpad**: Orchestrates overall system operation (green text)
2. **orchestrator**: Coordinates resources and task execution (purple text)
3. **triangulum**: Analyzes problems and generates solutions (blue text)
4. **auditor**: Monitors and verifies system operation (yellow text)
5. **neural_matrix**: Handles learning and prediction (blue text)

## Message Types

The agent:speak command supports several message types:

1. **default**: Standard agent message (blue text with robot emoji)
2. **success**: Success message (green text with check mark emoji)
3. **warning**: Warning message (yellow text with warning emoji)
4. **error**: Error message (red text with X emoji)
5. **info**: Informational message (cyan text with info emoji)

## Comprehensive Documentation

For complete command reference, see:

- [Agent Command Reference](agent_command_reference.md) - Detailed command documentation with examples

## Integration with Other Components

The agent communication system integrates with other FixWurx components:

- **Bug Detection System**: Agents can report bug detection progress and findings
- **Neural Matrix**: Training progress can be tracked and reported
- **Triangulum Engine**: Analysis results can be communicated directly to users
- **Auditor System**: Security and compliance findings can be reported in real-time
