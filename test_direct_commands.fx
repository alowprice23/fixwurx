# test_direct_commands.fx - Simple test of conversation commands

# Display header
echo "===== TESTING CONVERSATION COMMANDS IN FIXWURX ====="

# Step 1: Check if conversation commands are available
echo "\n[Step 1] Checking available conversation commands..."
help | grep conversation

# Step 2: List current conversations (should be empty or show existing)
echo "\n[Step 2] Listing current conversations..."
conversation:list

# Step 3: Create a bug for testing
echo "\n[Step 3] Creating a test bug..."
bug:create "Test bug for conversation logging" "This is a test bug to demonstrate conversation logging"

# Step 4: List bugs to see the created bug
echo "\n[Step 4] Listing bugs..."
bug:list

# Step 5: Show bug details
echo "\n[Step 5] Showing bug details (which should trigger logging)..."
bug:show 1

# Step 6: List conversations again (should now include the bug commands)
echo "\n[Step 6] Listing conversations after bug commands..."
conversation:list

# Step 7: Search for conversations related to bugs
echo "\n[Step 7] Searching for conversations with 'bug' keyword..."
conversation:search "bug"

# Step 8: Show the first conversation from the list if available
echo "\n[Step 8] If we have a conversation, showing details..."
echo "To view a conversation, you would run: conversation:show <session_id>"

# Done
echo "\n===== CONVERSATION COMMAND TEST COMPLETE ====="
