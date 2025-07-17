# test_real_agent_collaboration.fx
# Tests actual agent-to-agent collaboration with the conversation logging system

# Display header
echo "=========================================================="
echo "  TESTING REAL AGENT COLLABORATION WITH LOGGING SYSTEM"
echo "=========================================================="

# Step 1: Create a sample buggy file for agents to work on
echo "\n[Step 1] Creating a sample buggy file..."
cat > sample_buggy_calc.py << 'EOF'
def divide(a, b):
    # Bug: No check for zero division
    return a / b

def calculate_average(numbers):
    # Bug: No check for empty list
    total = sum(numbers)
    return divide(total, len(numbers))

def process_data(data_points):
    # Function that uses the buggy code
    results = []
    for point in data_points:
        avg = calculate_average(point)
        results.append(avg)
    return results

# Test cases that would trigger the bugs
test_data = [
    [1, 2, 3],     # Works fine
    [],            # Will cause division by zero in calculate_average
    [5, 0, 15]     # Works fine
]

try:
    results = process_data(test_data)
    print(f"Results: {results}")
except Exception as e:
    print(f"Error occurred: {e}")
EOF

echo "Created sample_buggy_calc.py with division by zero bug"

# Step 2: Create a bug report to initiate the agent collaboration
echo "\n[Step 2] Creating a bug report to initiate agent collaboration..."
bug:create "Division by zero in calculate_average" "The calculate_average function fails when given an empty list because it attempts to divide by zero. It needs proper error checking."

# Step 3: Get the bug ID and initiate a fix
echo "\n[Step 3] Getting the bug ID and initiating a fix..."
bug:list

# Step 4: Have the meta agent coordinate fixing the bug (this will trigger agent collaboration)
echo "\n[Step 4] Initiating meta agent coordination..."
# This triggers coordination between agents
meta coordinate "fix-division-by-zero" "Division by zero bug needs fixing in sample_buggy_calc.py"

# Step 5: Have the planner create a plan for fixing the bug
echo "\n[Step 5] Having planner generate a solution plan..."
# This will cause the planner to communicate with other agents
plan:generate "Fix division by zero in calculate_average function in sample_buggy_calc.py"

# Step 6: Have the observer analyze the bug
echo "\n[Step 6] Having observer analyze the bug..."
# This will cause the observer to communicate with the analyst
observe:analyze "sample_buggy_calc.py"

# Step 7: Listing conversations to see agent-to-agent communications
echo "\n[Step 7] Listing all conversations to see agent interactions..."
conversation:list

# Step 8: Search for agent-to-agent conversations
echo "\n[Step 8] Searching for agent-to-agent conversations..."
conversation:search "agent_to_agent"

# Step 9: Show details of a specific agent conversation
echo "\n[Step 9] Getting meta agent conversation details..."
echo "To show a specific conversation: conversation:show <session_id>"
echo "From the list above, you can choose a session ID to examine"

# Step 10: Clean up
echo "\n[Step 10] Cleaning up test files..."
# Keep this commented out to preserve the file for inspection
# rm sample_buggy_calc.py

# Done
echo "\n=========================================================="
echo "  AGENT COLLABORATION TEST COMPLETED"
echo "=========================================================="
