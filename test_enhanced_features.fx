#!/usr/bin/env fx
# Test script for enhanced shell features

# Test pipeline
echo "Testing pipeline..."
echo "This is a test of pipeline functionality" | uppercase

# Test redirection
echo "Testing redirection..."
echo "This line should be in the file" > test_output.txt
cat test_output.txt

# Test background execution
echo "Testing background execution..."
echo "Starting background task..."
sleep 2 &
echo "Background task started, continuing with script..."

# Test combined features
echo "Testing combined features..."
echo "This line should be uppercase and in a file" | uppercase > test_combined.txt
cat test_combined.txt

echo "Script completed successfully!"
