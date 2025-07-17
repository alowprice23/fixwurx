# test_agent_conversations.fx
# FixWurx shell script to test agent conversation logging capabilities

# Import modules
import json
import time

# Display header
echo "========================================================"
echo "      TESTING AGENT CONVERSATIONS IN FIXWURX SHELL      "
echo "========================================================"

# Step 1: Simulate agent interactions
echo "\n[Step 1] Simulating agent interactions..."

# Define a function to simulate agent-to-agent communication
function simulate_agent_messages() {
    # Get the conversation logger
    logger = conversation_logger
    
    # Meta to Planner communication
    echo "  Creating Meta → Planner communication..."
    session1 = logger.log_agent_to_agent(
        source_agent_id="meta",
        target_agent_id="planner",
        message={
            "action": "plan_generate",
            "bug_id": "bug-fx-123",
            "priority": "high",
            "timeout": 300
        }
    )
    
    # Planner response
    logger.log_agent_to_agent(
        source_agent_id="planner",
        target_agent_id="meta",
        message={
            "action": "plan_generated",
            "plan_id": "plan-fx-456",
            "steps": [
                {"id": 1, "action": "analyze_bug", "assigned_to": "observer"},
                {"id": 2, "action": "generate_fix", "assigned_to": "analyst"},
                {"id": 3, "action": "verify_fix", "assigned_to": "verifier"}
            ],
            "estimated_time": 90
        },
        session_id=session1
    )
    
    # Observer to Analyst communication
    echo "  Creating Observer → Analyst communication..."
    session2 = logger.log_agent_to_agent(
        source_agent_id="observer",
        target_agent_id="analyst",
        message={
            "action": "bug_analysis",
            "bug_id": "bug-fx-123",
            "analysis": {
                "file": "example_app.py",
                "line": 42,
                "type": "null_reference",
                "severity": "high"
            }
        }
    )
    
    # LLM interaction
    echo "  Creating LLM interaction from Meta agent..."
    session3 = logger.log_llm_interaction(
        agent_id="meta",
        prompt="Generate a plan to fix null reference bug in example_app.py line 42",
        response="The bug can be fixed by adding a null check before accessing the object. This is a common error in Python applications where an object may be None but code attempts to access its attributes or methods."
    )
    
    # Return the session IDs
    return [session1, session2, session3]
}

# Execute the simulation and capture session IDs
echo "  Running simulation..."
sessions = simulate_agent_messages()
echo "  Created " + str(len(sessions)) + " conversation sessions"
echo "  Session IDs: " + str(sessions)

# Wait briefly for all logs to be written
echo "  Waiting for logs to be written..."
sleep 2

# Step 2: List all conversations
echo "\n[Step 2] Listing all conversations..."
conversation:list

# Step 3: Show details of the first session
echo "\n[Step 3] Showing details of the first session..."
if len(sessions) > 0:
    conversation:show sessions[0]

# Step 4: Search for conversations mentioning a specific bug
echo "\n[Step 4] Searching for conversations mentioning 'bug-fx-123'..."
conversation:search "bug-fx-123"

# Step 5: Analyze a conversation with LLM
echo "\n[Step 5] Analyzing the first conversation with LLM..."
if len(sessions) > 0:
    conversation:analyze sessions[0]

# Step 6: Show the storage structure
echo "\n[Step 6] Showing conversation storage structure..."
# Use the execute command to list files in the storage directory
exec ls -la .triangulum/conversations

# Step 7: Clean up (don't actually delete, just show the command)
echo "\n[Step 7] Example of how to clean old conversations (not executed)..."
echo "  To clean old conversations, you would run: conversation:clean"

# Done
echo "\n========================================================"
echo "              TEST COMPLETED SUCCESSFULLY                "
echo "========================================================"
