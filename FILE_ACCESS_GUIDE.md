# FixWurx File Access Guide

This guide explains how to use the File Access commands to access files and directories outside the FixWurx working directory.

## Available Commands

### Read a File
Read the contents of a file from any location.

```
read:file path="/path/to/file.txt"
```

Example:
```
read:file path="C:/Users/Yusuf/Downloads/Carewurx V1/config.txt"
```

### Write to a File
Write content to a file at any location.

```
write:file path="/path/to/file.txt" content="Your content here"
```

Example:
```
write:file path="C:/Users/Yusuf/Downloads/Carewurx V1/new_file.txt" content="This is a test file"
```

### List Directory Contents
List the contents of a directory.

```
list:directory path="/path/to/directory" recursive=true|false
```

Example:
```
list:directory path="C:/Users/Yusuf/Downloads/Carewurx V1"
```

To list recursively (include all subdirectories):
```
list:directory path="C:/Users/Yusuf/Downloads/Carewurx V1" recursive=true
```

### Search Files
Search for a pattern in files within a directory.

```
search:files path="/path/to/directory" pattern="search term" recursive=true|false
```

Example:
```
search:files path="C:/Users/Yusuf/Downloads/Carewurx V1" pattern="groq model"
```

### Copy File
Copy a file from one location to another.

```
copy:file source="/path/to/source.txt" destination="/path/to/destination.txt"
```

Example:
```
copy:file source="C:/Users/Yusuf/Downloads/Carewurx V1/config.txt" destination="C:/Users/Yusuf/Downloads/FixWurx/backup_config.txt"
```

## Working with Carewurx Files

To find information about the Groq model used in Carewurx V1, you can:

1. List the directory contents:
   ```
   list:directory path="C:/Users/Yusuf/Downloads/Carewurx V1"
   ```

2. Search for files containing "groq" or "model":
   ```
   search:files path="C:/Users/Yusuf/Downloads/Carewurx V1" pattern="groq"
   ```

3. Read the configuration or model files:
   ```
   read:file path="C:/Users/Yusuf/Downloads/Carewurx V1/config.txt"
   ```

## Tips for Using File Access Commands

- Use double quotes around paths that contain spaces
- Use forward slashes (/) or escaped backslashes (\\\\) in paths
- The recursive parameter is optional and defaults to false for list:directory and true for search:files
- Files are read as text by default, but binary files will be detected automatically
