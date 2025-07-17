#!/usr/bin/env python3
# agent_communication_demo.fx
# A clean demonstration script for agent communication without Markdown formatting
# This script can be directly executed with fx

echo "Starting agent communication demo"

# Basic agent communication
agent:speak launchpad "Starting system initialization"
sleep 1

agent:speak orchestrator "Allocating resources" -t info
sleep 1

agent:speak triangulum "Analyzing system configuration" 
sleep 1

agent:speak auditor "Verifying security settings" -t warning
sleep 1

agent:speak neural_matrix "Training models" -t info
sleep 1

echo "Basic communication demo completed"

# Progress tracking demonstration
echo "Starting progress tracking demo"

# Start tracking a task
tracker_id=$(agent:progress start --agent triangulum --task "analysis" --description "Code analysis" --steps 5)

# Update progress
for i in 1 2 3 4 5; do
    # Sleep to simulate work
    sleep 1
    
    # Update progress
    agent:progress update --tracker $tracker_id --step $i --message "Step $i complete"
    
    # Have the agent report interesting findings at step 3
    if [ $i -eq 3 ]; then
        agent:speak triangulum "Found interesting pattern in code" -t info
    fi
done

# Complete the task
agent:progress complete --tracker $tracker_id --success --message "Analysis completed"

echo "Progress tracking demo completed"

# List active tasks
echo "Listing active tasks:"
agent:progress list

echo "Agent communication demo completed successfully"
