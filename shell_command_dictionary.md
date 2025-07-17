# FixWurx Shell Command and Scripting Dictionary
## THIS FILE CANNOT BE EDITED ##
## Introduction

This comprehensive guide documents all available commands and scripting capabilities in the FixWurx shell environment. The FixWurx shell is a powerful and extensible command-line interface that integrates various system components and provides a unified environment for interacting with the system.

## Table of Contents

1. [Basic Shell Commands](#basic-shell-commands)
2. [File and Directory Management](#file-and-directory-management)
3. [System Commands](#system-commands)
4. [Pipeline, Redirection, and Background Execution](#pipeline-redirection-and-background-execution)
5. [Shell Scripting](#shell-scripting)
6. [Component-Specific Commands](#component-specific-commands)
   - [Auditor Commands](#auditor-commands)
   - [Neural Matrix Commands](#neural-matrix-commands)
   - [Triangulum Commands](#triangulum-commands)
   - [Agent Commands](#agent-commands)
   - [Bug Detection Commands](#bug-detection-commands)
   - [Security Commands](#security-commands)

---

## Basic Shell Commands

These commands provide core shell functionality.

### help

Display help information for available commands.

**Usage:**
```
help [command]
```

**Examples:**
```
help                # List all available commands
help ls             # Show help for the 'ls' command
```

### exit / quit

Exit the shell.

**Usage:**
```
exit [exit_code]
quit [exit_code]
```

**Examples:**
```
exit                # Exit with code 0
exit 1              # Exit with code 1
```

### alias

Manage command aliases.

**Usage:**
```
alias list
alias add <alias> <command>
alias remove <alias>
```

**Examples:**
```
alias list                  # List all aliases
alias add ll "ls -l"        # Create an alias 'll' for 'ls -l'
alias remove ll             # Remove the 'll' alias
```

### history

Display command history.

**Usage:**
```
history [--count/-n <count>] [--clear]
```

**Examples:**
```
history                     # Show last 10 commands
history -n 20               # Show last 20 commands
history --clear             # Clear command history
```

### version

Display version information.

**Usage:**
```
version
```

### echo

Echo the given text.

**Usage:**
```
echo <text>
```

**Examples:**
```
echo Hello, world!          # Print "Hello, world!"
```

### clear

Clear the terminal screen.

**Usage:**
```
clear
```

### reload

Reload command modules.

**Usage:**
```
reload [module] [--all]
```

**Examples:**
```
reload                      # Reload all modules
reload auditor              # Reload the auditor module
```

### event

Trigger an event.

**Usage:**
```
event <event_type> [--data <json_data>] [--file <file_path>]
```

**Examples:**
```
event system.startup                    # Trigger system.startup event
event error.reported --data '{"level":"error","message":"Test error"}'
```

---

## File and Directory Management

These commands help you navigate and manipulate the file system.

### ls

List directory contents.

**Usage:**
```
ls [path] [-l/--long] [-a/--all]
```

**Examples:**
```
ls                          # List current directory
ls -l                       # Long format listing
ls -a                       # Show all files including hidden ones
ls /path/to/directory       # List specific directory
```

### cd

Change the current working directory.

**Usage:**
```
cd [directory]
```

**Examples:**
```
cd                          # Change to home directory
cd -                        # Change to previous directory
cd /path/to/directory       # Change to specific directory
cd ..                       # Move up one directory
```

### pwd

Print current working directory.

**Usage:**
```
pwd
```

### cat

Display file contents.

**Usage:**
```
cat <files...> [-n/--number]
```

**Examples:**
```
cat file.txt                    # Display contents of file.txt
cat file1.txt file2.txt         # Display contents of multiple files
cat -n file.txt                 # Display with line numbers
```

---

## System Commands

These commands allow interaction with the system.

### exec

Execute a shell command.

**Usage:**
```
exec <command> [args...]
```

**Examples:**
```
exec dir                    # Execute the 'dir' command
exec python --version       # Check Python version
```

### python

Execute Python code.

**Usage:**
```
python [-c/--code <code>] [-f/--file <file>] [-i/--interactive]
```

**Examples:**
```
python -c "print('Hello, world!')"      # Execute Python code
python -f script.py                      # Execute Python file
python -i                                # Start interactive Python shell
```

---

## Pipeline, Redirection, and Background Execution

These features allow for powerful command chaining and control.

### Command Pipelines

Chain commands together by piping the output of one command to the input of another.

**Usage:**
```
command1 | command2 | command3
```

**Examples:**
```
cat file.txt | grep "error"              # Find lines containing "error"
ls -l | grep "\.py$"                     # List Python files
```

### Output Redirection

Redirect command output to a file.

**Usage:**
```
command > file              # Overwrite file
command >> file             # Append to file
```

**Examples:**
```
ls -l > directory_listing.txt           # Save directory listing to file
echo "Additional content" >> log.txt     # Append to file
```

### Background Execution

Run commands in the background.

**Usage:**
```
command &                   # Run command in background
```

**Examples:**
```
long_running_command &      # Run command in background
```

### bg

Manage background tasks.

**Usage:**
```
bg list                     # List all background tasks
bg status <task_id>         # Check status of a task
bg output <task_id>         # Display output of a task
bg kill <task_id>           # Terminate a task
bg kill --all               # Terminate all tasks
bg cleanup                  # Clean up completed tasks
```

**Examples:**
```
bg list                     # List all tasks
bg output 1                 # Show output of task 1
```

---

## Shell Scripting

The FixWurx shell provides a powerful scripting language.

### Script Execution

**Usage:**
```
shell_script run <file> [--debug]
```

**Examples:**
```
shell_script run script.fx                  # Run a script
shell_script run test.fx --debug            # Run with debug output
```

### Variables

Variables can be defined and referenced in scripts.

**Examples:**
```
# Variable assignment
name="FixWurx"
count=10

# Variable reference
echo $name
echo $count
```

### Conditionals

If-else statements for conditional execution.

**Examples:**
```
if $value == 10 then
    echo "Value is 10"
else
    echo "Value is not 10"
fi

if $status == "success" then
    echo "Operation succeeded"
elif $status == "warning" then
    echo "Operation completed with warnings"
else
    echo "Operation failed"
fi
```

### Loops

For and while loops for repeated execution.

**Examples:**
```
# For loop
for item in item1 item2 item3 do
    echo $item
done

# While loop
while $count > 0 do
    echo $count
    count=$count - 1
done
```

### Functions

Define and call functions.

**Examples:**
```
function greet(name)
    echo "Hello, $name!"
    return 0
end

greet("World")
```

---

## Component-Specific Commands

### Auditor Commands

Commands for interacting with the auditor subsystem.

#### audit

Perform system audits.

**Usage:**
```
audit [options] [target]
```

#### audit_report

Generate and view audit reports.

**Usage:**
```
audit_report [options]
```

#### audit_config

Configure the audit system.

**Usage:**
```
audit_config [options]
```

### Neural Matrix Commands

Commands for interacting with the neural matrix subsystem.

#### neural_matrix

Access the neural matrix.

**Usage:**
```
neural_matrix [subcommand] [options]
```

#### train

Train the neural matrix.

**Usage:**
```
train [model] [options]
```

#### predict

Make predictions using the neural matrix.

**Usage:**
```
predict [input] [options]
```

#### visualize

Visualize neural matrix data.

**Usage:**
```
visualize [data] [options]
```

### Triangulum Commands

Commands for interacting with the triangulation engine.

#### triangulate

Perform triangulation.

**Usage:**
```
triangulate [target] [options]
```

#### triangulum_status

Check triangulum status.

**Usage:**
```
triangulum_status [options]
```

#### triangulum_config

Configure the triangulum subsystem.

**Usage:**
```
triangulum_config [options]
```

### Agent Commands

Commands for interacting with the agent system.

#### agent

Manage agents.

**Usage:**
```
agent [subcommand] [options]
```

#### agent_task

Create and manage agent tasks.

**Usage:**
```
agent_task [subcommand] [options]
```

#### agent_status

Check agent status.

**Usage:**
```
agent_status [agent] [options]
```

### Bug Detection Commands

Commands for detecting and analyzing bugs.

#### detect_bugs

Detect bugs in the target.

**Usage:**
```
detect_bugs [target] [options]
```

#### analyze_bug

Analyze a specific bug.

**Usage:**
```
analyze_bug [bug_id] [options]
```

#### fix_bug

Generate a fix for a bug.

**Usage:**
```
fix_bug [bug_id] [options]
```

### Security Commands

Commands for managing security.

#### security

Configure security settings.

**Usage:**
```
security [subcommand] [options]
```

**Examples:**
```
security info                           # Show security information
security audit                          # View audit log
security encrypt -v "secret data"       # Encrypt data
security decrypt -v "encrypted_data"    # Decrypt data
security token                          # Generate secure token
```

---

## Script Examples

### Basic Script

```
# This is a comment
echo "Starting script"

# Variable assignment
count=5
name="FixWurx"

echo "Name: $name"
echo "Count: $count"

# Conditional statement
if $count > 3 then
    echo "Count is greater than 3"
else
    echo "Count is not greater than 3"
fi

echo "Script completed"
```

### Loop Example

```
# For loop example
echo "For loop example:"
for i in 1 2 3 4 5 do
    echo "Item: $i"
done

# While loop example
echo "While loop example:"
counter=3
while $counter > 0 do
    echo "Counter: $counter"
    counter=$counter - 1
done
```

### Function Example

```
# Function definition
function calculate_sum(a, b)
    result=$a + $b
    echo "The sum of $a and $b is $result"
    return $result
end

# Function call
sum=calculate_sum(5, 7)
echo "Returned sum: $sum"

# Another function
function check_value(value)
    if $value > 10 then
        echo "Value is greater than 10"
        return 1
    else
        echo "Value is not greater than 10"
        return 0
    fi
end

status=check_value($sum)
echo "Check status: $status"
```

### Complex Script Example

```
# Complex script example
echo "Starting complex script"

# Initialize variables
success_count=0
failure_count=0
total_tests=5

function run_test(test_name, expected_result)
    echo "Running test: $test_name"
    
    # Simulate test execution
    if $test_name == "test1" or $test_name == "test3" or $test_name == "test5" then
        result="pass"
    else
        result="fail"
    fi
    
    # Check result
    if $result == $expected_result then
        echo "Test $test_name succeeded"
        return 1  # Success
    else
        echo "Test $test_name failed"
        return 0  # Failure
    fi
end

# Run tests
for test in test1 test2 test3 test4 test5 do
    expected="pass"  # All tests are expected to pass
    
    test_result=run_test($test, $expected)
    
    if $test_result == 1 then
        success_count=$success_count + 1
    else
        failure_count=$failure_count + 1
    fi
end

# Report results
echo "Test results:"
echo "Total tests: $total_tests"
echo "Successful: $success_count"
echo "Failed: $failure_count"

# Calculate success rate
success_rate=($success_count * 100) / $total_tests
echo "Success rate: $success_rate%"

if $success_rate >= 80 then
    echo "OVERALL STATUS: PASS"
else
    echo "OVERALL STATUS: FAIL"
fi

echo "Complex script completed"
```

---

## Advanced Usage Tips

### Combining Pipeline, Redirection, and Background Execution

Commands can be combined in powerful ways:

```
# Pipeline with redirection
ls -l | grep "\.py$" > python_files.txt

# Background execution with redirection
long_running_command > output.log &

# Complex combination
cat input.txt | grep "error" | sort > sorted_errors.txt &
```

### Effective Script Organization

For complex tasks, organize your scripts logically:

1. Begin with initialization code
2. Define functions next
3. Execute main script logic last
4. Use comments to document your code

### Error Handling in Scripts

Implement error handling in your scripts:

```
# Simple error handling
if $result != "success" then
    echo "Error: Operation failed"
    exit 1
fi

# Function with error handling
function safe_operation(param)
    if $param == "" then
        echo "Error: Empty parameter"
        return -1
    fi
    
    # Perform operation
    # ...
    
    return 0
end

# Check return value
status=safe_operation($value)
if $status < 0 then
    echo "Operation failed with status: $status"
    exit 1
fi
```

### Integration Between Components

Different components can be combined in scripts:

```
# Example of integrated workflow
echo "Starting integrated workflow"

# Detect bugs
detect_bugs $target_dir

# Analyze each bug
for bug_id in $detected_bugs do
    analyze_bug $bug_id
    
    # Generate fix
    fix_status=fix_bug $bug_id
    
    if $fix_status == 0 then
        echo "Bug $bug_id fixed successfully"
    else
        echo "Failed to fix bug $bug_id"
    fi
done

# Generate report
audit_report --bugs --fixes > workflow_report.txt

echo "Workflow completed"
