# FixWurx Test Script
# This script runs all the component tests

echo "Running FixWurx component tests..."
echo "=================================="

# Run Triangulation Engine test
test:engine

# Run Neural Matrix test
test:matrix

# Run Integration test
test:integration

echo "All tests complete!"
