# FixWurx Agent Communication and Progress Tracking Reference

## Introduction

This document provides a comprehensive reference for the agent communication and progress tracking capabilities in the FixWurx shell environment. These features enable direct agent-to-user communication, progress reporting, and multi-agent collaboration.

## Table of Contents

1. [Agent Communication Commands](#agent-communication-commands)
2. [Progress Tracking Commands](#progress-tracking-commands)
3. [Conversation Logging Commands](#conversation-logging-commands)
4. [Script Examples](#script-examples)
5. [Integration with Other Components](#integration-with-other-components)

---

## Agent Communication Commands

Commands for agent-to-user and agent-to-agent communication.

### agent:speak

Make an agent speak directly to the user.

**Usage:**
```
agent:speak <agent_id> <message> [--type/-t <message_type>] [--raw/-r]
```

**Parameters:**
- `agent_id`: The ID of the agent (e.g., "launchpad", "triangulum", "auditor")
- `message`: The message to speak
- `--type/-t`: Message type (default, error, success, warning, info)
- `--raw/-r`: Output raw message without formatting

**Examples:**
```
agent:speak launchpad "Starting bug detection process"           # Standard message
agent:speak triangulum "Analysis complete" -t success            # Success message
agent:speak auditor "Missing configuration file" -t warning      # Warning message
agent:speak neural_matrix "Training failed" -t error             # Error message
speak orchestrator "Coordinating resources"                      # Using the 'speak' alias
```

### Message Types

The agent:speak command supports several message types that determine how the message is formatted:

1. **default**: Standard agent message (blue text with robot emoji)
2. **success**: Success message (green text with check mark emoji)
3. **warning**: Warning message (yellow text with warning emoji)
4. **error**: Error message (red text with X emoji)
5. **info**: Informational message (cyan text with info emoji)

### Available Agents

The following agents are registered by default:

1. **launchpad**: Orchestrates overall system operation (green text)
2. **orchestrator**: Coordinates resources and task execution (purple text)
3. **triangulum**: Analyzes problems and generates solutions (blue text)
4. **auditor**: Monitors and verifies system operation (yellow text)
5. **neural_matrix**: Handles learning and prediction (blue text)

---

## Progress Tracking Commands

Commands for tracking progress of long-running tasks.

### agent:progress

Manage progress tracking for agent tasks.

**Usage:**
```
agent:progress <action> [options]
```

**Actions:**
- `start`: Start tracking a task
- `update`: Update task progress
- `complete`: Complete a task
- `pause`: Pause a task
- `resume`: Resume a task
- `list`: List tasks
- `show`: Show task details

### Start Tracking a Task

**Usage:**
```
agent:progress start --agent <agent_id> --task <task_id> --description <description> [--steps <total_steps>]
```

**Examples:**
```
agent:progress start --agent triangulum --task "code_analysis" --description "Analyzing codebase" --steps 10
```

### Update Task Progress

**Usage:**
```
agent:progress update --tracker <tracker_id> --step <current_step> [--message <status_message>]
```

**Examples:**
```
agent:progress update --tracker triangulum_code_analysis_1234567890 --step 5 --message "Halfway done"
```

### Complete a Task

**Usage:**
```
agent:progress complete --tracker <tracker_id> [--success] [--fail] [--message <completion_message>]
```

**Examples:**
```
agent:progress complete --tracker triangulum_code_analysis_1234567890 --success
agent:progress complete --tracker triangulum_code_analysis_1234567890 --fail --message "Failed due to missing data"
```

### Pause and Resume Tasks

**Usage:**
```
agent:progress pause --tracker <tracker_id> [--message <pause_message>]
agent:progress resume --tracker <tracker_id> [--message <resume_message>]
```

**Examples:**
```
agent:progress pause --tracker triangulum_code_analysis_1234567890 --message "Waiting for resources"
agent:progress resume --tracker triangulum_code_analysis_1234567890
```

### List Tasks

**Usage:**
```
agent:progress list [--agent <agent_id>] [--all/-A]
```

**Examples:**
```
agent:progress list                       # List all active tasks
agent:progress list --agent launchpad     # List tasks for launchpad agent
agent:progress list --all                 # Include completed tasks
tasks                                     # Using the 'tasks' alias
```

### Show Task Details

**Usage:**
```
agent:progress show --tracker <tracker_id>
```

**Examples:**
```
agent:progress show --tracker triangulum_code_analysis_1234567890
```

---

## Conversation Logging Commands

Commands for managing and analyzing agent conversations.

### conversation:list

List conversation sessions.

**Usage:**
```
conversation:list [--limit/-n <count>] [--all/-a] [--agent/-g <agent_id>] [--type/-t <type>]
```

**Examples:**
```
conversation:list                      # List recent sessions
conversation:list -n 20                # List 20 most recent sessions
conversation:list -a                   # List all sessions
conversation:list -g triangulum        # List sessions for triangulum agent
convs                                  # Using the 'convs' alias
```

### conversation:show

Show a conversation session.

**Usage:**
```
conversation:show <session_id> [--json/-j] [--full/-f]
```

**Examples:**
```
conversation:show session_1234567890   # Show session details
conversation:show session_1234567890 -f # Show full messages
conv session_1234567890                # Using the 'conv' alias
```

### conversation:search

Search for conversations.

**Usage:**
```
conversation:search <query> [--limit/-n <count>] [--start-time/-s <time>] [--end-time/-e <time>]
```

**Examples:**
```
conversation:search "error"                          # Search for "error"
conversation:search "bug fix" -n 20                  # Find 20 matches for "bug fix"
conversation:search "permission" -s "2025-07-10"     # From specific date
```

### conversation:analyze

Analyze a conversation session with LLM.

**Usage:**
```
conversation:analyze <session_id>
```

**Examples:**
```
conversation:analyze session_1234567890   # Analyze session
```

### conversation:clean

Clean old conversation sessions.

**Usage:**
```
conversation:clean [--days <days>] [--force/-f]
```

**Examples:**
```
conversation:clean              # Clean sessions older than retention period
conversation:clean --days 7     # Clean sessions older than 7 days
conversation:clean -f           # Force cleaning without confirmation
```

---

## Script Examples

### Basic Agent Communication

```
# Script demonstrating basic agent communication
echo "Starting agent communication demo"

# Direct agent communication
agent:speak launchpad "Starting system initialization"
agent:speak orchestrator "Allocating resources" -t info
agent:speak triangulum "Analyzing system configuration" 
agent:speak auditor "Verifying security settings" -t warning
agent:speak neural_matrix "Training models" -t info

echo "Communication demo completed"
```

### Progress Tracking

```
# Script demonstrating progress tracking
echo "Starting progress tracking demo"

# Start tracking a task
tracker_id=$(agent:progress start --agent triangulum --task "analysis" --description "Code analysis" --steps 5)

# Update progress
for i in 1 2 3 4 5 do
    # Simulate work
    sleep 1
    
    # Update progress
    agent:progress update --tracker $tracker_id --step $i --message "Step $i complete"
    
    # Have the agent report interesting findings
    if $i == 3 then
        agent:speak triangulum "Found interesting pattern in code" -t info
    fi
done

# Complete the task
agent:progress complete --tracker $tracker_id --success --message "Analysis completed"

echo "Progress tracking demo completed"
```

### Multi-Agent Collaboration

```
# Script demonstrating multi-agent collaboration
echo "Starting multi-agent collaboration demo"

# Initialize a workflow with the Launchpad agent
agent:speak launchpad "Starting bug detection and fix workflow"

# Start progress tracking
workflow_tracker=$(agent:progress start --agent launchpad --task "workflow" --description "Bug workflow" --steps 100)

# Triangulum analyzes the code
agent:speak triangulum "Beginning code analysis..."

analysis_tracker=$(agent:progress start --agent triangulum --task "analysis" --description "Code analysis" --steps 5)

# Update analysis progress
for i in 1 2 3 4 5 do
    sleep 1
    agent:progress update --tracker $analysis_tracker --step $i --message "Analyzing module $i"
done

agent:progress complete --tracker $analysis_tracker --success

# Report findings
agent:speak triangulum "Analysis complete: Found 3 bugs" -t success

# Orchestrator prioritizes fixes
agent:speak orchestrator "Prioritizing bugs based on severity"
agent:progress update --tracker $workflow_tracker --step 30 --message "Bugs prioritized"

# Neural Matrix generates solutions
agent:speak neural_matrix "Generating fix patterns"
solution_tracker=$(agent:progress start --agent neural_matrix --task "solutions" --description "Generating solutions" --steps 3)

for i in 1 2 3 do
    sleep 1
    agent:progress update --tracker $solution_tracker --step $i
    agent:speak neural_matrix "Generated solution $i"
done

agent:progress complete --tracker $solution_tracker --success

# Auditor verifies fixes
agent:speak auditor "Verifying fixes"
verify_tracker=$(agent:progress start --agent auditor --task "verify" --description "Verifying fixes" --steps 3)

for i in 1 2 3 do
    sleep 1
    agent:progress update --tracker $verify_tracker --step $i
done

agent:progress complete --tracker $verify_tracker --success
agent:speak auditor "All fixes verified" -t success

# Complete workflow
agent:progress update --tracker $workflow_tracker --step 100
agent:progress complete --tracker $workflow_tracker --success
agent:speak launchpad "Bug detection and fix workflow completed successfully" -t success

echo "Multi-agent collaboration demo completed"
```

---

## Integration with Other Components

The agent communication and progress tracking systems can be integrated with other FixWurx components:

### Integration with Bug Detection

```
# Detect bugs with agent communication
agent:speak triangulum "Starting bug detection in target directory"

# Run bug detection
detect_bugs $target_dir

# Process each bug with progress tracking
bugs=$(cat bug_report.json | jq -r '.bugs[].id')
total_bugs=$(echo $bugs | wc -w)

tracker_id=$(agent:progress start --agent triangulum --task "bug_fixing" --description "Fixing bugs" --steps $total_bugs)
current=0

for bug_id in $bugs do
    current=$current + 1
    
    # Update progress
    agent:progress update --tracker $tracker_id --step $current --message "Fixing bug $bug_id"
    
    # Analyze and fix the bug
    agent:speak triangulum "Analyzing bug $bug_id"
    analyze_bug $bug_id
    
    fix_status=fix_bug $bug_id
    
    if $fix_status == 0 then
        agent:speak triangulum "Bug $bug_id fixed successfully" -t success
    else
        agent:speak triangulum "Failed to fix bug $bug_id" -t error
    fi
done

# Complete tracking
agent:progress complete --tracker $tracker_id --success
agent:speak triangulum "Bug fixing completed" -t success
```

### Integration with Neural Matrix

```
# Train neural matrix with progress reporting
agent:speak neural_matrix "Starting model training"

# Start progress tracking
tracker_id=$(agent:progress start --agent neural_matrix --task "training" --description "Training neural model" --steps 100)

# Start training
train_process=train --model="code_fix" --epochs=100 --background

# Monitor training progress
while true do
    # Check if training is still running
    if not is_running($train_process) then
        break
    fi
    
    # Get current progress
    progress=$(get_training_progress)
    
    # Update progress tracker
    agent:progress update --tracker $tracker_id --step $progress
    
    # Sleep briefly
    sleep 5
done

# Complete tracking
agent:progress complete --tracker $tracker_id --success
agent:speak neural_matrix "Model training completed successfully" -t success
```

### Integration with Auditor

```
# Run system audit with agent communication
agent:speak auditor "Starting system audit"

# Start tracking
tracker_id=$(agent:progress start --agent auditor --task "audit" --description "System audit" --steps 5)

# Update progress for each audit phase
agent:progress update --tracker $tracker_id --step 1 --message "Checking configurations"
audit --check-config

agent:progress update --tracker $tracker_id --step 2 --message "Verifying permissions"
audit --check-permissions

agent:progress update --tracker $tracker_id --step 3 --message "Scanning for vulnerabilities"
audit --scan-vulnerabilities

agent:progress update --tracker $tracker_id --step 4 --message "Checking compliance"
audit --check-compliance

agent:progress update --tracker $tracker_id --step 5 --message "Generating report"
audit_report --full > audit_results.txt

# Complete tracking
agent:progress complete --tracker $tracker_id --success
agent:speak auditor "System audit completed, report saved to audit_results.txt" -t success
