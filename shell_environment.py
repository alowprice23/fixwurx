#!/usr/bin/env python3
"""
Shell Environment

A dynamic and extensible shell environment that integrates various modules
and provides a unified interface for interacting with the system.
"""

import os
import sys
import cmd
import json
import yaml
import time
import logging
import readline
import importlib
import traceback
import io
import contextlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("shell.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ShellEnvironment")

# Import command pipeline
from shell_environment_enhanced import CommandPipeline, bg_command

class ComponentRegistry:
    """
    Registry for components, command handlers, and event handlers.
    Enables dynamic registration and lookup of functionality.
    """
    
    def __init__(self):
        self.components = {}
        self.command_handlers = {}
        self.event_handlers = {}
        self.aliases = {}
        
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component with the registry.
        
        Args:
            name: Name of the component
            component: Component instance
        """
        self.components[name] = component
        logger.info(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a component by name.
        
        Args:
            name: Name of the component
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
    
    def register_command_handler(self, command: str, handler: Callable, component_name: str) -> None:
        """
        Register a command handler.
        
        Args:
            command: Command string
            handler: Command handler function
            component_name: Name of the component that owns this handler
        """
        self.command_handlers[command] = {
            "handler": handler,
            "component": component_name
        }
        logger.info(f"Registered command handler: {command} -> {component_name}")
    
    def get_command_handler(self, command: str) -> Optional[Dict]:
        """
        Get a command handler by command string.
        
        Args:
            command: Command string
            
        Returns:
            Command handler dict or None if not found
        """
        if command in self.command_handlers:
            return self.command_handlers[command]
        
        # Check aliases
        if command in self.aliases:
            aliased_command = self.aliases[command]
            return self.command_handlers.get(aliased_command)
        
        return None
    
    def register_event_handler(self, event_type: str, handler: Callable, component_name: str) -> None:
        """
        Register an event handler.
        
        Args:
            event_type: Type of event
            handler: Event handler function
            component_name: Name of the component that owns this handler
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append({
            "handler": handler,
            "component": component_name
        })
        logger.info(f"Registered event handler: {event_type} -> {component_name}")
    
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Trigger an event of the specified type.
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            List of results from event handlers
        """
        results = []
        
        if event_type in self.event_handlers:
            for handler_info in self.event_handlers[event_type]:
                try:
                    handler = handler_info["handler"]
                    result = handler(event_data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
                    results.append({"success": False, "error": str(e)})
        
        return results
    
    def register_alias(self, alias: str, command: str) -> None:
        """
        Register a command alias.
        
        Args:
            alias: Alias string
            command: Command string
        """
        self.aliases[alias] = command
        logger.info(f"Registered alias: {alias} -> {command}")

class ShellEnvironment(cmd.Cmd):
    """
    Interactive shell environment with command processing and tab completion.
    """
    
    intro = """
  ███████╗██╗██╗  ██╗██╗    ██╗██╗   ██╗██████╗ ██╗  ██╗
  ██╔════╝██║╚██╗██╔╝██║    ██║██║   ██║██╔══██╗╚██╗██╔╝
  █████╗  ██║ ╚███╔╝ ██║ █╗ ██║██║   ██║██████╔╝ ╚███╔╝ 
  ██╔══╝  ██║ ██╔██╗ ██║███╗██║██║   ██║██╔══██╗ ██╔██╗ 
  ██║     ██║██╔╝ ██╗╚███╔███╔╝╚██████╔╝██║  ██║██╔╝ ██╗
  ╚═╝     ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
                                                        
 Dynamic Shell Environment v1.0.0
 Type 'help' for a list of commands.
"""
    prompt = "fx> "
    
    def __init__(self, registry: ComponentRegistry):
        super().__init__()
        self.registry = registry
        self.history_file = os.path.expanduser("~/.fx_history")
        self.load_history()
        
        # Create command pipeline
        self.command_pipeline = CommandPipeline(registry)
        registry.register_component("command_pipeline", self.command_pipeline)
        
        # Register built-in commands
        self.registry.register_command_handler("help", self._help_command, "shell")
        self.registry.register_command_handler("exit", self._exit_command, "shell")
        self.registry.register_command_handler("quit", self._exit_command, "shell")
        self.registry.register_command_handler("alias", self._alias_command, "shell")
        self.registry.register_command_handler("history", self._history_command, "shell")
        self.registry.register_command_handler("version", self._version_command, "shell")
        self.registry.register_command_handler("echo", self._echo_command, "shell")
        self.registry.register_command_handler("event", self._event_command, "shell")
        self.registry.register_command_handler("clear", self._clear_command, "shell")
        self.registry.register_command_handler("reload", self._reload_command, "shell")
        self.registry.register_command_handler("bg", bg_command, "shell")
        
    def load_history(self) -> None:
        """Load command history from file."""
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
                readline.set_history_length(1000)
        except Exception as e:
            logger.error(f"Error loading history: {e}")
    
    def save_history(self) -> None:
        """Save command history to file."""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def completedefault(self, text, line, begidx, endidx):
        """Provide tab completion for commands."""
        # Get the command being typed
        parts = line.split()
        if not parts:
            return []
        
        # If we're completing the command itself
        if len(parts) == 1 and not line.endswith(' '):
            commands = list(self.registry.command_handlers.keys()) + list(self.registry.aliases.keys())
            return [cmd for cmd in commands if cmd.startswith(text)]
        
        # TODO: Add completion for command arguments based on each command's specific needs
        return []
    
    def default(self, line: str) -> None:
        """Process commands not recognized as built-in."""
        try:
            # Skip empty lines
            if not line.strip():
                return
            
            # Parse the command line to detect pipelines, redirection, and background execution
            parsed_command = self.command_pipeline.parse_command_line(line)
            
            # Handle background execution
            if parsed_command["background"]:
                task_id = self.command_pipeline.execute_in_background(parsed_command)
                print(f"[{task_id}] Started in background: {parsed_command['original_line']}")
                return
            
            # Execute the command pipeline
            exit_code, output = self.command_pipeline.execute_pipeline(parsed_command)
            
            # Display output if not redirected and not empty
            if not parsed_command["redirection"] and output:
                print(output, end='')
            
            # Log execution
            logger.info(f"Executed command pipeline: {line} (exit_code: {exit_code})")
        except Exception as e:
            print(f"Error executing command: {e}")
            logger.error(f"Error executing command: {e}\n{traceback.format_exc()}")
    
    def emptyline(self) -> None:
        """Do nothing on empty line."""
        pass
    
    def do_EOF(self, arg: str) -> bool:
        """Handle EOF (Ctrl+D) to exit the shell."""
        print("\nExiting shell...")
        self.save_history()
        return True
    
    def _help_command(self, args: str) -> int:
        """
        Display help information.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        if args:
            # Help for a specific command
            command = args.strip()
            handler_info = self.registry.get_command_handler(command)
            
            if handler_info:
                handler = handler_info["handler"]
                component = handler_info.get("component", "unknown")
                
                print(f"\nCommand: {command} (Component: {component})")
                print("-" * 60)
                
                if handler.__doc__:
                    print(handler.__doc__)
                else:
                    print("No help available for this command.")
                
                # Try to get the command's own help
                try:
                    if command != "help":  # Avoid infinite recursion
                        handler("--help")
                except Exception:
                    pass
            else:
                print(f"Unknown command: {command}")
        else:
            # List all commands by component
            commands_by_component = {}
            
            for cmd, info in self.registry.command_handlers.items():
                component = info.get("component", "unknown")
                if component not in commands_by_component:
                    commands_by_component[component] = []
                commands_by_component[component].append(cmd)
            
            print("\nAvailable Commands:")
            print("-" * 60)
            
            for component, commands in sorted(commands_by_component.items()):
                print(f"\n{component.capitalize()} Commands:")
                # Sort and format commands in columns
                commands.sort()
                for i in range(0, len(commands), 3):
                    row = commands[i:i+3]
                    print("  " + "  ".join(f"{cmd:<20}" for cmd in row))
            
            print("\nType 'help <command>' for more information about a command.")
        
        return 0
    
    def _exit_command(self, args: str) -> int:
        """
        Exit the shell.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        print("Exiting shell...")
        self.save_history()
        sys.exit(0)
        return 0  # Not reached
    
    def _alias_command(self, args: str) -> int:
        """
        Manage command aliases.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        import shlex
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Manage command aliases")
        parser.add_argument("action", nargs="?", choices=["list", "add", "remove"], default="list", 
                           help="Action to perform")
        parser.add_argument("alias", nargs="?", help="Alias to add or remove")
        parser.add_argument("command", nargs="?", help="Command to alias")
        
        try:
            cmd_args = parser.parse_args(shlex.split(args))
        except SystemExit:
            return 1
        
        action = cmd_args.action
        alias_name = cmd_args.alias
        command = cmd_args.command
        
        # List aliases
        if action == "list":
            print("\nCommand Aliases:")
            print("-" * 60)
            
            if not self.registry.aliases:
                print("No aliases defined")
                return 0
            
            for alias, cmd in sorted(self.registry.aliases.items()):
                print(f"  {alias:<15} -> {cmd}")
        
        # Add alias
        elif action == "add" and alias_name and command:
            # Check if command exists
            if self.registry.get_command_handler(command) is None:
                print(f"Error: Command '{command}' does not exist")
                return 1
            
            self.registry.register_alias(alias_name, command)
            print(f"Alias '{alias_name}' created for command '{command}'")
        
        # Remove alias
        elif action == "remove" and alias_name:
            if alias_name in self.registry.aliases:
                del self.registry.aliases[alias_name]
                print(f"Alias '{alias_name}' removed")
            else:
                print(f"Error: Alias '{alias_name}' does not exist")
                return 1
        
        else:
            print("Error: Invalid action or missing required arguments")
            print("Usage examples:")
            print("  alias list")
            print("  alias add <alias> <command>")
            print("  alias remove <alias>")
            return 1
        
        return 0
    
    def _history_command(self, args: str) -> int:
        """
        Display command history.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        import shlex
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Display command history")
        parser.add_argument("--count", "-n", type=int, default=10, help="Number of history entries to show")
        parser.add_argument("--clear", action="store_true", help="Clear history")
        
        try:
            cmd_args = parser.parse_args(shlex.split(args))
        except SystemExit:
            return 1
        
        count = cmd_args.count
        clear_history = cmd_args.clear
        
        if clear_history:
            # Clear history
            readline.clear_history()
            print("Command history cleared")
            self.save_history()
            return 0
        
        # Display history
        history_length = readline.get_current_history_length()
        start = max(1, history_length - count + 1)
        
        print("\nCommand History:")
        print("-" * 60)
        
        for i in range(start, history_length + 1):
            try:
                cmd = readline.get_history_item(i)
                print(f"  {i}: {cmd}")
            except Exception:
                pass
        
        return 0
    
    def _version_command(self, args: str) -> int:
        """
        Display version information.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        print("\nFixWurx Shell Environment v1.0.0")
        print("Copyright (c) 2025 FixWurx Team")
        print("\nComponents:")
        
        for name, component in self.registry.components.items():
            version = getattr(component, "version", "Unknown")
            print(f"  {name}: {version}")
        
        return 0
    
    def _echo_command(self, args: str) -> int:
        """
        Echo the given arguments.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        print(args)
        return 0
    
    def _event_command(self, args: str) -> int:
        """
        Trigger an event.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        import shlex
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Trigger an event")
        parser.add_argument("event_type", help="Type of event to trigger")
        parser.add_argument("--data", help="Event data (JSON string)")
        parser.add_argument("--file", help="Path to JSON or YAML file containing event data")
        
        try:
            cmd_args = parser.parse_args(shlex.split(args))
        except SystemExit:
            return 1
        
        event_type = cmd_args.event_type
        event_data_str = cmd_args.data
        event_data_file = cmd_args.file
        
        # Get event data
        event_data = {}
        
        if event_data_str:
            try:
                event_data = json.loads(event_data_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing event data: {e}")
                return 1
        elif event_data_file:
            try:
                with open(event_data_file, 'r') as f:
                    if event_data_file.endswith('.json'):
                        event_data = json.load(f)
                    elif event_data_file.endswith('.yaml') or event_data_file.endswith('.yml'):
                        event_data = yaml.safe_load(f)
                    else:
                        print("Error: Event data file must be JSON or YAML")
                        return 1
            except Exception as e:
                print(f"Error loading event data file: {e}")
                return 1
        
        # Trigger the event
        print(f"Triggering event: {event_type}")
        results = self.registry.trigger_event(event_type, event_data)
        
        # Display results
        print("\nEvent Handler Results:")
        print("-" * 60)
        
        if not results:
            print("No handlers for this event type")
            return 0
        
        for i, result in enumerate(results, 1):
            success = result.get("success", False)
            print(f"  Handler {i}: {'Success' if success else 'Failure'}")
            
            if not success and "error" in result:
                print(f"    Error: {result['error']}")
            
            # Print other result data
            for key, value in result.items():
                if key not in ("success", "error"):
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, indent=2)
                        print(f"    {key}:")
                        for line in value_str.split("\n"):
                            print(f"      {line}")
                    else:
                        print(f"    {key}: {value}")
        
        return 0
    
    def _clear_command(self, args: str) -> int:
        """
        Clear the terminal screen.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        return 0
    
    def _reload_command(self, args: str) -> int:
        """
        Reload command modules.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        import shlex
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Reload command modules")
        parser.add_argument("module", nargs="?", help="Module to reload (e.g., 'auditor', 'neural_matrix', 'triangulum')")
        parser.add_argument("--all", action="store_true", help="Reload all modules")
        
        try:
            cmd_args = parser.parse_args(shlex.split(args))
        except SystemExit:
            return 1
        
        module_name = cmd_args.module
        reload_all = cmd_args.all or not module_name
        
        if reload_all:
            # Reload all modules
            print("Reloading all command modules...")
            load_command_modules(self.registry)
            print("All command modules reloaded")
        else:
            # Reload specific module
            module_file = f"{module_name}_commands.py"
            if not Path(module_file).exists():
                print(f"Error: Module file '{module_file}' not found")
                return 1
            
            print(f"Reloading module: {module_name}")
            
            try:
                module = importlib.import_module(f"{module_name}_commands")
                importlib.reload(module)
                
                # Call the registration function
                register_func = getattr(module, f"register_{module_name}_commands", None)
                if register_func:
                    register_func(self.registry)
                    print(f"Module '{module_name}' reloaded successfully")
                else:
                    print(f"Error: Module '{module_name}' does not have a register function")
                    return 1
            except Exception as e:
                print(f"Error reloading module: {e}")
                logger.error(f"Error reloading module: {e}\n{traceback.format_exc()}")
                return 1
        
        return 0


def load_command_modules(registry: ComponentRegistry) -> None:
    """
    Load all command modules.
    
    Args:
        registry: Component registry
    """
    # List of command modules to load
    modules = [
        "auditor_commands",
        "neural_matrix_commands",
        "triangulum_commands",
        "fixwurx_commands",
        "launchpad_commands",
        "auditor_shell_access",  
        "auditor_report_query",
        "conversational_interface",
        "conversation_history_manager",  # Add conversation history manager
        "agent_shell_integration",
        "test_commands",
        "bug_detection_commands",
        "collaborative_improvement_commands"  # Add collaborative improvement commands
    ]
    
    # Load each module
    for module_name in modules:
        module_file = f"{module_name}.py"
        
        if Path(module_file).exists():
            try:
                logger.info(f"Loading module: {module_name}")
                module = importlib.import_module(module_name)
                
                # Call the registration function
                register_func = getattr(module, f"register_{module_name.split('_')[0]}_commands", None)
                if register_func:
                    register_func(registry)
                else:
                    logger.warning(f"Module {module_name} does not have a register function")
            except Exception as e:
                logger.error(f"Error loading module {module_name}: {e}\n{traceback.format_exc()}")
        else:
            logger.warning(f"Module file not found: {module_file}")


def create_base_commands(registry: ComponentRegistry) -> None:
    """
    Create base command handlers.
    
    Args:
        registry: Component registry
    """
    # Register basic commands
    registry.register_command_handler("ls", ls_command, "shell")
    registry.register_command_handler("cd", cd_command, "shell")
    registry.register_command_handler("pwd", pwd_command, "shell")
    registry.register_command_handler("cat", cat_command, "shell")
    registry.register_command_handler("exec", exec_command, "shell")
    registry.register_command_handler("python", python_command, "shell")


def ls_command(args: str) -> int:
    """
    List directory contents.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="List directory contents")
    parser.add_argument("path", nargs="?", default=".", help="Directory path to list")
    parser.add_argument("-l", "--long", action="store_true", help="Use long listing format")
    parser.add_argument("-a", "--all", action="store_true", help="Show all files (including hidden)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    path = cmd_args.path
    long_format = cmd_args.long
    show_all = cmd_args.all
    
    try:
        # Get directory listing
        path_obj = Path(path).resolve()
        if not path_obj.exists():
            print(f"Error: Path '{path}' does not exist")
            return 1
        
        if not path_obj.is_dir():
            print(f"Error: '{path}' is not a directory")
            return 1
        
        # Get files and directories
        entries = list(path_obj.iterdir())
        
        # Filter hidden files if not showing all
        if not show_all:
            entries = [entry for entry in entries if not entry.name.startswith('.')]
        
        # Sort entries (directories first, then files)
        entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        
        if long_format:
            # Long format listing
            print(f"total {len(entries)}")
            
            for entry in entries:
                # Get file stats
                stat = entry.stat()
                
                # Format mode (permissions)
                mode = "d" if entry.is_dir() else "-"
                mode += "rwx" if (stat.st_mode & 0o400) else "---"
                mode += "rwx" if (stat.st_mode & 0o040) else "---"
                mode += "rwx" if (stat.st_mode & 0o004) else "---"
                
                # Format size
                size = stat.st_size
                
                # Format time
                mtime = time.strftime("%b %d %H:%M", time.localtime(stat.st_mtime))
                
                # Format name (add trailing slash for directories)
                name = entry.name
                if entry.is_dir():
                    name += "/"
                
                print(f"{mode} {stat.st_nlink:4} {stat.st_uid:8} {stat.st_gid:8} {size:10} {mtime} {name}")
        else:
            # Simple listing
            col_width = max(len(entry.name) for entry in entries) + 4 if entries else 20
            num_cols = max(1, 80 // col_width)
            
            for i in range(0, len(entries), num_cols):
                row = entries[i:i+num_cols]
                print("".join(f"{entry.name + ('/' if entry.is_dir() else ''):<{col_width}}" for entry in row))
        
        return 0
    except Exception as e:
        print(f"Error listing directory: {e}")
        return 1


def cd_command(args: str) -> int:
    """
    Change current working directory.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    try:
        cmd_args = shlex.split(args)
        path = cmd_args[0] if cmd_args else os.path.expanduser("~")
        
        # Handle special cases
        if path == "-":
            path = os.environ.get("OLDPWD", ".")
        
        # Expand home directory
        path = os.path.expanduser(path)
        
        # Resolve path
        abs_path = os.path.abspath(path)
        
        # Check if path exists
        if not os.path.exists(abs_path):
            print(f"Error: Path '{path}' does not exist")
            return 1
        
        # Check if path is a directory
        if not os.path.isdir(abs_path):
            print(f"Error: '{path}' is not a directory")
            return 1
        
        # Save current directory
        os.environ["OLDPWD"] = os.getcwd()
        
        # Change directory
        os.chdir(abs_path)
        print(f"Changed directory to: {os.getcwd()}")
        
        return 0
    except Exception as e:
        print(f"Error changing directory: {e}")
        return 1


def pwd_command(args: str) -> int:
    """
    Print current working directory.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    print(os.getcwd())
    return 0


def cat_command(args: str) -> int:
    """
    Display file contents.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Display file contents")
    parser.add_argument("files", nargs="+", help="Files to display")
    parser.add_argument("-n", "--number", action="store_true", help="Number all output lines")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    files = cmd_args.files
    number_lines = cmd_args.number
    
    exit_code = 0
    
    for file_path in files:
        try:
            # Check if file exists
            path_obj = Path(file_path).resolve()
            if not path_obj.exists():
                print(f"Error: File '{file_path}' does not exist")
                exit_code = 1
                continue
            
            # Check if file is a directory
            if path_obj.is_dir():
                print(f"Error: '{file_path}' is a directory")
                exit_code = 1
                continue
            
            # Print file contents
            if len(files) > 1:
                print(f"\n==> {file_path} <==")
            
            with open(path_obj, 'r', errors='replace') as f:
                if number_lines:
                    for i, line in enumerate(f, 1):
                        print(f"{i:6}  {line}", end="")
                else:
                    print(f.read(), end="")
        
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            exit_code = 1
    
    return exit_code


def exec_command(args: str) -> int:
    """
    Execute a shell command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import subprocess
    
    if not args:
        print("Error: No command specified")
        print("Usage: exec <command> [args...]")
        return 1
    
    try:
        # Execute the command
        process = subprocess.run(args, shell=True, text=True)
        return process.returncode
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1


def python_command(args: str) -> int:
    """
    Execute Python code.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Execute Python code")
    parser.add_argument("-c", "--code", help="Python code to execute")
    parser.add_argument("-f", "--file", help="Python file to execute")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive Python shell")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    code = cmd_args.code
    file_path = cmd_args.file
    interactive = cmd_args.interactive
    
    try:
        # Execute Python code
        if code:
            # Execute code string
            print("Executing Python code:")
            print("-" * 60)
            
            # Create a local namespace
            local_ns = {}
            
            # Execute the code
            exec(code, globals(), local_ns)
            
            # Print local variables
            if local_ns:
                print("\nLocal variables:")
                for name, value in local_ns.items():
                    if not name.startswith("__"):
                        print(f"  {name} = {value}")
        
        elif file_path:
            # Execute Python file
            path_obj = Path(file_path).resolve()
            
            if not path_obj.exists():
                print(f"Error: File '{file_path}' does not exist")
                return 1
            
            if not path_obj.is_file():
                print(f"Error: '{file_path}' is not a file")
                return 1
            
            print(f"Executing Python file: {file_path}")
            print("-" * 60)
            
            # Execute the file
            with open(path_obj, 'r') as f:
                code = f.read()
                exec(code, globals())
        
        elif interactive:
            # Start interactive Python shell
            import code as pycode
            
            console = pycode.InteractiveConsole(locals=globals())
            console.interact(banner="Python Interactive Shell\nType 'exit()' or Ctrl+D to exit")
        
        else:
            print("Error: No Python code, file, or interactive mode specified")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error executing Python: {e}")
        return 1


def main():
    """Main function to run the shell environment."""
    # Create registry
    registry = ComponentRegistry()
    
    # Create base commands
    create_base_commands(registry)
    
    # Load command modules
    load_command_modules(registry)
    
    # Create shell environment
    shell = ShellEnvironment(registry)
    
    # Make registry available to modules
    sys.modules["__main__"].registry = registry
    
    try:
        # Run shell
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting shell...")
    except Exception as e:
        logger.error(f"Error in shell: {e}\n{traceback.format_exc()}")
    finally:
        shell.save_history()


if __name__ == "__main__":
    main()
