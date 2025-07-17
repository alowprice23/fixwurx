# FixWurx Scripting Guide

This guide provides an overview of FixWurx scripting conventions and best practices.

## Scripting Guidelines

1. **Start with Metadata Comments**: Begin every script with metadata comments including description, version, and author.
   ```fx
   # Script Name
   # Description: What the script does
   # Version: 1.0
   # Author: Your Name
   ```

2. **Include Input Validation**: Always validate input parameters before using them.
   ```fx
   if [ -z "$1" ]; then
       echo "Error: Missing required parameter"
       exit 1
   fi
   ```

3. **Use Logging**: Implement proper logging for important operations.
   ```fx
   log_message() {
       echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1"
   }
   
   log_message "Starting process"
   ```

4. **Handle Errors**: Implement proper error handling and return appropriate exit codes.
   ```fx
   if ! command; then
       log_message "Error: Command failed"
       exit 1
   fi
   ```

5. **Use Variables for Configuration**: Define variables at the beginning of the script for easy configuration.
   ```fx
   OUTPUT_DIR="./output"
   RETENTION_DAYS=30
   VERBOSE=true
   ```

6. **Create Functions for Reusable Code**: Organize code into functions for better readability and reuse.
   ```fx
   process_file() {
       local file="$1"
       # Process the file
   }
   
   for file in *.txt; do
       process_file "$file"
   done
   ```

7. **Include Help Information**: Add help/usage information that can be displayed with `-h` or `--help`.
   ```fx
   if [[ "$1" == "-h" || "$1" == "--help" ]]; then
       echo "Usage: ./script.fx [options] <arguments>"
       echo "Options:"
       echo "  -v, --verbose     Enable verbose output"
       echo "  -h, --help        Display this help message"
       exit 0
   fi
   ```

8. **Use Comments**: Add comments to explain complex logic or important decisions.
   ```fx
   # This is necessary because the API returns data in a specific format
   response=$(curl -s "$API_URL" | jq '.results')
   ```

9. **Clean Up Temporary Resources**: Always clean up temporary files or resources.
   ```fx
   cleanup() {
       rm -f "$TMP_FILE"
       log_message "Cleanup completed"
   }
   
   trap cleanup EXIT
   ```

10. **Follow Consistent Formatting**: Use consistent indentation and formatting.
    ```fx
    if condition; then
        command1
        command2
    else
        command3
    fi
    ```

## Script Structure

A well-structured FixWurx script typically follows this pattern:

1. **Metadata Comments**: Description, version, author
2. **Configuration Variables**: Settings and parameters
3. **Function Definitions**: Helper functions and main logic
4. **Argument Parsing**: Process command-line arguments
5. **Input Validation**: Validate parameters and environment
6. **Main Execution**: Perform the primary tasks
7. **Cleanup**: Clean up resources and exit properly

## Example Script

```fx
#!/usr/bin/env bash
# File Backup Script
# Description: Creates a backup of specified files or directories
# Version: 1.0
# Author: FixWurx Team

# Configuration variables
SOURCE_DIR="${1:-./data}"
BACKUP_DIR="${2:-./backups}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"
LOG_FILE="$BACKUP_DIR/backup_log.txt"

# Function definitions
log_message() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

create_backup() {
    local source="$1"
    local destination="$2"
    
    log_message "Creating backup of $source to $destination"
    tar -czf "$destination" -C "$(dirname "$source")" "$(basename "$source")"
    return $?
}

check_directory() {
    if [ ! -d "$1" ]; then
        log_message "Error: Directory $1 does not exist"
        return 1
    fi
    return 0
}

cleanup() {
    log_message "Backup process completed"
}

# Set up trap for cleanup
trap cleanup EXIT

# Main execution
log_message "Starting backup process"

# Create backup directory if it doesn't exist
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
    log_message "Created backup directory: $BACKUP_DIR"
fi

# Validate source directory
if ! check_directory "$SOURCE_DIR"; then
    log_message "Error: Source directory validation failed"
    exit 1
fi

# Create the backup
if create_backup "$SOURCE_DIR" "$BACKUP_FILE"; then
    log_message "Backup created successfully: $BACKUP_FILE"
else
    log_message "Error: Backup creation failed with status $?"
    exit 1
fi

# Display backup information
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
log_message "Backup size: $BACKUP_SIZE"

exit 0
```

## UI Interaction Examples

You can use the following commands to interact with the UI from your scripts.

### Updating the Dashboard

To update a component on the dashboard, use the `dashboard:update` command.

```fx
# fx
# Description: Updates the dashboard with the latest analysis results.
# Version: 1.0
# Author: FixWurx Team

# Run analysis
analyze --type patterns --dir ./src --out analysis.json

# Update dashboard
dashboard:update --component analysis_results --data @analysis.json
```

### Sending a Dashboard Alert

To send an alert to the dashboard, use the `dashboard:alert` command.

```fx
# fx
# Description: Sends an alert if the backup fails.
# Version: 1.0
# Author: FixWurx Team

run --script backup.fx
if [ $? -ne 0 ]; then
    dashboard:alert --severity critical --message "Backup failed. Please check the logs."
fi
```

### Generating a Visualization

To generate and display a visualization, use the `viz:generate` command.

```fx
# fx
# Description: Generates a visualization of the latest performance metrics.
# Version: 1.0
# Author: FixWurx Team

# Get performance data
run --script get_metrics.fx --out metrics.json

# Generate visualization
viz:generate --type line_chart --data @metrics.json --title "Performance Over Time"
```

## Command Reference

For a complete list of available commands, see the [FixWurx Shell Commands](fixwurx_shell_commands.md) document.
