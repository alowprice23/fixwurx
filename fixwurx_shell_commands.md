# FixWurx Shell Commands

This document serves as the command lexicon for the FixWurx shell environment.

## `analyze`
**Description:** Analyze code or data in the specified directory.
**Arguments:**
- `--type` - Type of analysis to perform (syntax, imports, patterns)
- `--dir` - Directory to analyze
- `--out` - Output file for the analysis report
**Examples:**
- `analyze --type syntax --dir ./src --out report.txt`
- `analyze --type imports --dir ./lib`

## `backup`
**Description:** Create a backup of specified data.
**Arguments:**
- `--type` - Type of data to backup (files, database)
- `--source` - Source to backup
- `--dest` - Destination for the backup
**Examples:**
- `backup --type files --source ./data --dest ./backups`
- `backup --type database --source mydatabase --dest ./db_backups`

## `configure`
**Description:** Configure system settings or services.
**Arguments:**
- `--service` - Service to configure
- `--options` - Configuration options as JSON string
**Examples:**
- `configure --service web-server --options '{"port": 8080}'`
- `configure --service database --options '{"max_connections": 100}'`

## `deploy`
**Description:** Deploy an application or service.
**Arguments:**
- `--app` - Application to deploy
- `--env` - Environment to deploy to
- `--version` - Version to deploy
**Examples:**
- `deploy --app myapp --env production --version 1.0.0`
- `deploy --app backend --env staging`

## `list`
**Description:** List resources or items.
**Arguments:**
- `--type` - Type of items to list
- `--filter` - Filter string
**Examples:**
- `list --type scripts`
- `list --type backups --filter 'date > 2023-01-01'`

## `run`
**Description:** Execute a script or task.
**Arguments:**
- `--script` - Script to run
- `--args` - Arguments to pass to the script
**Examples:**
- `run --script backup.fx --args 'daily'`
- `run --script deploy.fx --args 'production v1.2.3'`

## `dashboard:update`
**Description:** Updates a dashboard component with new data.
**Arguments:**
- (No arguments currently, future implementation could include component name and data)
**Examples:**
- `dashboard:update`

## `dashboard:alert`
**Description:** Displays an alert on the dashboard.
**Arguments:**
- (No arguments currently, future implementation could include alert message and severity)
**Examples:**
- `dashboard:alert`

## `viz:generate`
**Description:** Generates a new visualization.
**Arguments:**
- (No arguments currently, future implementation could include visualization type and data source)
**Examples:**
- `viz:generate`

---
## Common Shell Commands

These are standard shell commands that are also available within the FixWurx environment.

## `echo`
**Description:** Display a line of text.
**Arguments:**
- `string` - The text to display.
**Examples:**
- `echo "Hello, World!"`

## `mkdir`
**Description:** Create a new directory.
**Arguments:**
- `directory_name` - The name of the directory to create.
**Examples:**
- `mkdir new_directory`

## `ls`
**Description:** List files and directories.
**Arguments:**
- `path` - The path to list.
**Examples:**
- `ls -la`
- `ls src/`

## `cp`
**Description:** Copy files or directories.
**Arguments:**
- `source` - The source file or directory.
- `destination` - The destination path.
**Examples:**
- `cp file.txt file.bak`
- `cp -r src/ build/`

## `mv`
**Description:** Move or rename files or directories.
**Arguments:**
- `source` - The source file or directory.
- `destination` - The destination path or new name.
**Examples:**
- `mv old_name.txt new_name.txt`
- `mv file.txt /tmp/`

## `rm`
**Description:** Remove files or directories.
**Arguments:**
- `file` - The file or directory to remove.
**Examples:**
- `rm file.txt`
- `rm -r old_directory`
