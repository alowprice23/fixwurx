# Enhanced Shell Features

This document explains the enhanced shell features implemented in the FixWurx shell environment.

## Pipeline Support

The shell now supports command pipelines, allowing you to chain multiple commands together using the pipe (`|`) symbol. The output of each command is used as the input for the next command in the pipeline.

### Example Usage

```
fx> ls | grep .py | sort
```

This command lists all files, filters for those containing ".py", and sorts the results.

```
fx> cat file.txt | uppercase
```

This reads a file and converts its contents to uppercase.

## Command Output Redirection

You can redirect command output to files using the redirection operators:

- `>` - Redirect output to a file (overwrite if exists)
- `>>` - Redirect output to a file (append if exists)

### Example Usage

```
fx> ls > file_list.txt
fx> echo "Additional files:" >> file_list.txt
```

The first command writes the directory listing to a file, and the second command appends a line to the file.

## Background Task Execution

You can run commands in the background by appending an ampersand (`&`) to the end of the command. This allows you to continue using the shell while the command executes.

### Example Usage

```
fx> sleep 10 &
[1] Started in background: sleep 10
fx> 
```

The command runs in the background, and you can continue using the shell.

## Background Task Management

The `bg` command allows you to manage background tasks:

```
fx> bg list                # List all background tasks
fx> bg status 1            # Show status of task with ID 1
fx> bg output 1            # Show output from task with ID 1
fx> bg kill 1              # Terminate task with ID 1
fx> bg cleanup             # Remove completed tasks
```

## Combined Features

You can combine these features for more complex operations:

```
fx> ls -la | grep .py | sort > python_files.txt &
[1] Started in background: ls -la | grep .py | sort > python_files.txt
```

This command lists all files, filters for Python files, sorts them, saves the result to a file, and does all of this in the background.

## Implementation Details

The enhanced shell features are implemented in:

- `shell_environment_enhanced.py`: Contains the core implementation of the CommandPipeline class
- `shell_environment.py`: Integration with the shell environment
- `test_enhanced_shell.py`: Tests for the enhanced features

These features complete the shell environment implementation as outlined in the FixWurx blueprint.
