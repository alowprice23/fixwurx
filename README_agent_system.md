# Agent System for FixWurx

The Agent System provides a family tree of specialized agents for debugging and fixing issues, with the Planner Agent as the root. This document explains how to use the Agent System through the shell environment.

## Overview

The Agent System consists of:

1. **Meta Agent**: Provides oversight and coordination across all agents
2. **Planner Agent (Root)**: Coordinates the overall debugging strategy and manages other agents
3. **Observer Agent**: Analyzes bugs and reproduces issues
4. **Analyst Agent**: Generates patches and solutions
5. **Verifier Agent**: Tests fixes and ensures correctness

## Getting Started

### Initialization

The Agent System is automatically initialized when you load the shell environment. If you need to manually initialize it, use:

```
fx> agent init
```

### Viewing Status

To view the current status of the Agent System:

```
fx> agent status
```

This will display metrics like active bugs, paths, and success/failure rates.

### Family Tree

To view the agent family tree:

```
fx> agent tree
```

Or use the alias:

```
fx> agents
```

## Bug Management

### Creating a Bug

```
fx> bug create <bug-id> [title]
```

Example:
```
fx> bug create login-bug "User cannot log in with correct credentials"
```

### Listing Bugs

```
fx> bug list [status]
```

or use the alias:

```
fx> bugs
```

To filter by status:
```
fx> bug list analyzed
```

### Viewing Bug Details

```
fx> bug show <bug-id>
```

### Updating Bug Properties

```
fx> bug update <bug-id> property=value
```

Valid properties: title, description, severity, status, tag

Examples:
```
fx> bug update login-bug severity=high
fx> bug update login-bug status=in-progress
fx> bug update login-bug tag=critical
```

## Solution Path Management

### Generating Solution Paths

```
fx> plan generate <bug-id>
```

This will generate multiple potential solution paths for fixing the bug.

### Selecting the Best Path

```
fx> plan select <bug-id>
```

This will display the highest-priority solution path for the bug.

## Agent Interaction

### Observer Agent

```
fx> observe analyze <bug-id>
```

The Observer Agent will analyze the bug and document its behavior.

### Analyst Agent

```
fx> analyze patch <bug-id>
```

The Analyst Agent will generate a patch for the bug.

### Verifier Agent

```
fx> verify test <bug-id>
```

The Verifier Agent will test the patch to ensure it fixes the bug.

## Meta Agent Commands

The Meta Agent provides oversight and coordination capabilities for the agent system.

### Meta Agent Status

```
fx> meta status
```

This will display the status of the Meta Agent including agent count, coordination events, conflict resolutions, and other metrics.

### Agent Network Visualization

```
fx> agent network
```

This will display the agent network, showing the connections between agents and their current status.

To generate a visualization file:

```
fx> agent network visualize
```

### Meta Insights

```
fx> meta insights
```

This will generate a new meta-insight about the agent system, analyzing patterns and suggesting optimizations.

### Agent Coordination

```
fx> meta coordinate agent1,agent2,agent3 task-id task-type
```

This will create a coordination plan for multiple agents to work together on a task.

### View Registered Agents

```
fx> meta agents
```

This will list all agents registered with the Meta Agent, including their type, status, and registration timestamp.

### View Agent Conflicts

```
fx> meta conflicts
```

This will show any detected conflicts between agents and their resolution status.

## Advanced Usage

### Viewing Detailed Metrics

```
fx> agent metrics verbose
```

This will show detailed metrics including neural weights for different agent types.

### Agent Network Analysis

```
fx> agent network
```

This shows the network of agent relationships, including connection strength between agents.

## Integration with Other Components

The Agent System integrates with:

1. **Shell Environment**: Through command handlers for user interaction
2. **Triangulation Engine**: For core decision-making logic
3. **Neural Matrix**: For pattern recognition and learning
4. **Auditor**: For logging and monitoring

## Implementation Details

The Agent System is implemented across multiple files:

- `agent_commands.py`: Command handlers for the shell
- `planner_agent.py`: Core implementation of the Planner Agent
- `agent_shell_integration.py`: Integration with the shell environment
- `specialized_agents.py`: Implementation of specialized agents
- `data_structures.py`: Data structures for bugs and solution paths

## Example Workflow

1. Create a bug:
   ```
   fx> bug create api-bug "API returns 500 on valid requests"
   ```

2. Analyze the bug:
   ```
   fx> observe analyze api-bug
   ```

3. Generate solution paths:
   ```
   fx> plan generate api-bug
   ```

4. Select the best path:
   ```
   fx> plan select api-bug
   ```

5. Generate a patch:
   ```
   fx> analyze patch api-bug
   ```

6. Verify the fix:
   ```
   fx> verify test api-bug
   ```

7. Check the final status:
   ```
   fx> bug show api-bug
