#!/usr/bin/env fx
# Example FixWurx Shell Script
# This script demonstrates the features of the FixWurx shell scripting system

# Variables
var name = "FixWurx"
var version = "1.0"
var count = 10
var items = ["apple", "banana", "cherry", "date", "elderberry"]

# Output
echo "Welcome to ${name} Shell Scripting v${version}"
echo "----------------"

# Function declaration
function greet(person) 
    echo "Hello, ${person}!"
    
    if person == "admin" then
        echo "You have administrator privileges."
    else
        echo "You have standard user privileges."
    fi
    
    return "Greeted ${person}"
end

# Function call
greet("user")
greet("admin")

# Conditional
if count > 5 then
    echo "Count is greater than 5"
else
    echo "Count is 5 or less"
fi

# For loop
echo "Items in the list:"
for item in items do
    echo "- ${item}"
done

# While loop
var i = 0
echo "Countdown:"
while i < 5 do
    echo "${5 - i}..."
    i = i + 1
done
echo "Launch!"

# Command execution
echo "System information:"
version
echo "Current directory:"
ls
