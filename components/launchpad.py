#!/usr/bin/env python3
"""
Launchpad

The Launchpad serves as the system bootstrap and entry point for the FixWurx shell.
It initializes components, sets up the environment, and provides the conversational
interface for users to interact with the system.
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import importlib
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging - only log to file, not to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("launchpad.log")
    ]
)

# Ensure logs from other modules don't print to console
for logger_name in ['httpx', 'LLMClientReal', 'StateManager', 'ConversationalInterface']:
    module_logger = logging.getLogger(logger_name)
    module_logger.propagate = False
    # Make sure this logger only writes to file
    if not any(isinstance(h, logging.FileHandler) for h in module_logger.handlers):
        module_logger.addHandler(logging.FileHandler(f"{logger_name.lower()}.log"))

logger = logging.getLogger("Launchpad")

class ComponentRegistry:
    """
    Registry for system components. Manages component registration,
    retrieval, and initialization to ensure proper dependency resolution.
    """
    
    def __init__(self):
        """Initialize the component registry."""
        self.components = {}
        self.initialized = False
        logger.info("Component Registry initialized")
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component with the registry.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
        logger.info(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Any:
        """
        Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """
        List all registered components.
        
        Returns:
            List of component names
        """
        return list(self.components.keys())
    
    def initialize_components(self, config: Dict[str, Any]) -> bool:
        """
        Initialize all registered components in the correct order based on dependencies.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Components already initialized")
            return True

        # Define component initialization order
        component_order = [
            "state_manager",
            "llm_client",
            "neural_matrix",
            "command_executor",
            "intent_classification_system",
            "planning_engine",
            "decision_tree",
            "script_library",
            "conversation_logger",
            "agent_system",
            "conversational_interface"
        ]
        
        # Initialize components in order
        for component_name in component_order:
            self._initialize_component(component_name)
        
        # Verify all required components are available
        self._verify_required_components()
        
        self.initialized = True
        logger.info("All components initialized successfully")
        return True

    def _get_component_dependencies(self, component_name: str) -> List[str]:
        """Get dependencies for a component."""
        dependency_map = {
            "conversational_interface": ["state_manager", "planning_engine", "command_executor", "llm_client", "neural_matrix"],
            "planning_engine": ["llm_client", "decision_tree", "script_library"],
            "command_executor": ["permission_system"],
            "decision_tree": ["bug_identification_logic", "solution_path_generation", "patch_generation_logic", "verification_logic"],
            "agent_system": ["meta_agent"]
        }
        return dependency_map.get(component_name, [])

    def _initialize_component(self, component_name: str) -> bool:
        """Initialize a specific component and its dependencies."""
        component = self.get_component(component_name)
        if not component:
            logger.warning(f"Component {component_name} not available")
            return False
        
        # Check if already initialized
        if self.components.get(component_name) and hasattr(self.components[component_name], 'initialized') and self.components[component_name].initialized:
            return True
        
        # Get component dependencies
        dependencies = self._get_component_dependencies(component_name)
        
        # Initialize dependencies first
        for dependency in dependencies:
            if not self._initialize_component(dependency):
                logger.error(f"Failed to initialize dependency {dependency} for {component_name}")
                return False
        
        # Initialize the component
        if hasattr(component, 'initialize'):
            try:
                success = component.initialize()
                logger.info(f"Initialized component {component_name}: {success}")
                if hasattr(component, 'initialized'):
                    component.initialized = True
                return success
            except Exception as e:
                logger.error(f"Error initializing component {component_name}: {e}")
                return False
        else:
            logger.warning(f"Component {component_name} has no initialize method")
            return True

    def _verify_required_components(self) -> bool:
        """Verify that all required components are available."""
        required_components = [
            "state_manager",
            "llm_client",
            "planning_engine",
            "command_executor",
            "conversational_interface"
        ]
        
        missing_components = []
        for component_name in required_components:
            component = self.get_component(component_name)
            if not component:
                missing_components.append(component_name)
        
        if missing_components:
            logger.error(f"Missing required components: {', '.join(missing_components)}")
            
            # Try to diagnose why components are missing
            for component_name in missing_components:
                self._diagnose_missing_component(component_name)
            
            return False
        
        logger.info("All required components are available")
        return True

    def _diagnose_missing_component(self, component_name: str):
        """Diagnose why a component is missing."""
        # Check if the component file exists
        component_module = component_name.replace('_', '')
        potential_paths = [
            f"components/{component_module}.py",
            f"components/{component_name}.py",
            f"{component_module}.py",
            f"{component_name}.py"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Component file exists at {path}, but component was not registered")
                return
        
        logger.error(f"Component file not found for {component_name}")
    
    def shutdown_components(self) -> None:
        """Shutdown all registered components in reverse order."""
        if not self.initialized:
            return
        
        # Define shutdown order (reverse of initialization order)
        shutdown_order = [
            "conversational_interface",
            "command_executor",
            "planning_engine",
            "conversation_logger",
            "script_library",
            "neural_matrix"
        ]
        
        # Shutdown components in order
        for component_name in shutdown_order:
            component = self.get_component(component_name)
            if component:
                logger.info(f"Shutting down component: {component_name}")
                try:
                    if hasattr(component, "shutdown"):
                        component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down component {component_name}: {e}")
                    logger.error(traceback.format_exc())
        
        self.initialized = False
        logger.info("All components shut down")

class Launchpad:
    """
    The Launchpad is responsible for bootstrapping the system, initializing
    components, and providing the entry point for users to interact with the shell.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Launchpad.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or "config.json"
        self.config = self._load_config()
        self.registry = ComponentRegistry()
        self.initialized = False
        self.history_file = os.path.join(os.path.expanduser("~"), ".launchpad_history")
        
        # Create history file if it doesn't exist
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w") as f:
                pass
        
        # Create banner text
        self.banner = """
        █████████████████████████████████████████████████████████████████████████████
        █                                                                         █
        █                            FixWurx LLM Shell                           █
        █                                                                         █
        █████████████████████████████████████████████████████████████████████████████
        
        Welcome to the FixWurx LLM Shell. Type 'help' for a list of commands.
        """
        
        logger.info("Launchpad initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                return self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        config = {
            "neural_matrix": {
                "model_path": "neural_matrix/models/default",
                "embedding_dimension": 768,
                "context_window": 4096,
                "enable_visualization": True
            },
            "script_library": {
                "library_path": "script_library",
                "git_enabled": True
            },
            "conversation_logger": {
                "logs_path": "conversation_logs",
                "max_conversation_size": 100,
                "compression_enabled": True,
                "retention_days": 90
            },
            "planning_engine": {
                "command_lexicon_path": "fixwurx_shell_commands.md",
                "scripting_guide_path": "fixwurx_scripting_guide.md",
                "enable_goal_deconstruction": True
            },
            "command_executor": {
                "timeout_seconds": 60,
                "blacklist_path": "security/command_blacklist.json",
                "enable_confirmation": True
            },
            "conversational_interface": {
                "default_verbosity": "standard",
                "history_size": 50,
                "enable_streaming": True
            },
            "launchpad": {
                "shell_prompt": "fx> ",
                "enable_llm_startup": True,
                "startup_script_path": "startup.fx",
                "enable_agent_system": True,
                "enable_auditor": True
            }
        }
        
        # Create configuration file
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
        
        return config
    
    def initialize(self) -> bool:
        """
        Initialize the Launchpad and system components.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Launchpad already initialized")
            return True
        
        try:
            # Load components
            self._load_components()
            
            # Initialize components
            if not self.registry.initialize_components(self.config):
                logger.error("Failed to initialize components")
                return False
            
            # Run startup script if enabled
            launchpad_config = self.config.get("launchpad", {})
            if launchpad_config.get("enable_llm_startup", True):
                self._run_startup_script()
            
            # Start Auditor Agent daemon if enabled
            if launchpad_config.get("enable_auditor", True):
                self._start_auditor_daemon()
            
            self.initialized = True
            logger.info("Launchpad initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Launchpad: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_components(self) -> None:
        """Load system components."""
        component_loaders = {
            "state_manager": "components.state_manager",
            "permission_system": "components.permission_system",
            "llm_client": "components.llm_client_real",
            "neural_matrix": "neural_matrix.core.neural_matrix",
            "file_access_utility": "components.file_access_utility",
            "script_library": "components.script_library",
            "conversation_logger": "components.conversation_logger",
            "bug_identification_logic": "components.bug_identification_logic",
            "solution_path_generation": "components.solution_path_generation",
            "patch_generation_logic": "components.patch_generation_logic",
            "verification_logic": "components.verification_logic",
            "decision_tree": "components.decision_tree_integration",
            "intent_classification_system": "components.intent_classification_system",
            "planning_engine": "components.planning_engine",
            "command_executor": "components.command_executor",
            "meta_agent": "agents.core.meta_agent",
            "agent_system": "agents.core.agent_system",
            "auditor_agent": "agents.auditor.auditor_agent",
            "credential_manager": "components.credential_manager",
            "conversational_interface": "components.conversational_interface",
        }

        for name, path in component_loaders.items():
            try:
                module = importlib.import_module(path)
                # Handle cases where the class name is different from the module name
                if name == "neural_matrix":
                    instance = module.NeuralMatrix(self.config.get(name, {}))
                elif name == "decision_tree":
                    instance = module.DecisionTreeIntegration(self.config.get(name, {}))
                elif name in ["llm_client", "agent_system", "auditor_agent"]:
                    instance = module.get_instance(self.config.get(name, {}))
                elif name == "intent_classification_system":
                    instance = module.IntentClassificationSystem(self.registry)
                else:
                    instance = module.get_instance(self.registry, self.config.get(name, {}))
                self.registry.register_component(name, instance)
            except ImportError:
                logger.warning(f"{name.replace('_', ' ').title()} not available at {path}")
            except Exception as e:
                logger.error(f"Error loading component {name}: {e}")
                logger.error(traceback.format_exc())

        # Load UI and File Access commands separately as they are not components
        try:
            from components import ui_commands
            ui_cmds = ui_commands.get_commands()
            executor = self.registry.get_component("command_executor")
            if executor:
                for name, func in ui_cmds.items():
                    executor.register_internal_command(name, func)
        except ImportError:
            logger.warning("UI Commands not available")

        try:
            from components import file_access_commands
            file_cmds = file_access_commands.get_commands(self.registry)
            executor = self.registry.get_component("command_executor")
            if executor and file_cmds:
                for name, func in file_cmds.items():
                    executor.register_internal_command(name, func)
                logger.info(f"Registered {len(file_cmds)} file access commands")
        except ImportError:
            logger.warning("File Access Commands not available")
            
        logger.info("Components loaded successfully")
    
    def _run_startup_script(self) -> None:
        """Run the startup script."""
        launchpad_config = self.config.get("launchpad", {})
        startup_script_path = launchpad_config.get("startup_script_path", "startup.fx")
        
        if not os.path.exists(startup_script_path):
            logger.info(f"Startup script not found: {startup_script_path}")
            return
        
        try:
            logger.info(f"Running startup script: {startup_script_path}")
            # Get command executor
            executor = self.registry.get_component("command_executor")
            if not executor:
                logger.warning("Command Executor not available, skipping startup script")
                return
            
            # Read startup script
            with open(startup_script_path, "r") as f:
                script_content = f.read()
            
            # Execute startup script
            result = executor.execute_script(script_content, "system", timeout=60)
            if not result.get("success", False):
                logger.error(f"Error running startup script: {result.get('error', 'Unknown error')}")
            else:
                logger.info("Startup script executed successfully")
        except Exception as e:
            logger.error(f"Error running startup script: {e}")
            logger.error(traceback.format_exc())
    
    def start_cli(self) -> None:
        """Start the command-line interface."""
        if not self.initialized:
            if not self.initialize():
                logger.error("Failed to initialize Launchpad, exiting")
                return
        
        # Get conversational interface
        interface = self.registry.get_component("conversational_interface")
        if not interface:
            logger.error("Conversational Interface not available, exiting")
            return
        
        # Start the CLI
        try:
            # Print banner
            print(self.banner)
            
            # Start conversation
            conv_logger = self.registry.get_component("conversation_logger")
            conversation_id = None
            if conv_logger:
                result = conv_logger.start_conversation("user")
                if result.get("success", False):
                    conversation_id = result.get("conversation_id")
            
            # Main interaction loop
            launchpad_config = self.config.get("launchpad", {})
            shell_prompt = launchpad_config.get("shell_prompt", "fx> ")
            
            while True:
                try:
                    # Get user input
                    user_input = input(shell_prompt)
                    
                    # Save to history
                    with open(self.history_file, "a") as f:
                        f.write(f"{user_input}\n")
                    
                    # Exit if requested
                    if user_input.lower() in ["exit", "quit"]:
                        break
                    
                    # Process user input
                    if conversation_id:
                        conv_logger.add_message(conversation_id, "user", user_input)
                    
                    # Process through conversational interface
                    response = interface.process_input(user_input)
                    
                    # Log response
                    if conversation_id and response:
                        conv_logger.add_message(conversation_id, "system", response)
                    
                    # Print response
                    if response:
                        # Make sure there's a clear line break before the response
                        print(response)
                        # Add an extra newline after the response for better readability
                        print()
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except EOFError:
                    print("\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Error processing input: {e}")
                    logger.error(traceback.format_exc())
                    print(f"Error: {e}")
            
            # End conversation
            if conversation_id:
                conv_logger.end_conversation(conversation_id)
                
        except Exception as e:
            logger.error(f"Error in CLI: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Shutdown components
            self.shutdown()
    
    def start_interactive(self) -> None:
        """Start the interactive shell mode."""
        self.start_cli()
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a single command.
        
        Args:
            command: Command to execute
            
        Returns:
            Result dictionary
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Failed to initialize Launchpad")
                return {"success": False, "error": "Failed to initialize Launchpad"}
        
        # Get command executor
        executor = self.registry.get_component("command_executor")
        if not executor:
            logger.error("Command Executor not available")
            return {"success": False, "error": "Command Executor not available"}
        
        # Execute command
        try:
            result = executor.execute(command, "user")
            return result
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def execute_script(self, script_path: str) -> Dict[str, Any]:
        """
        Execute a script file.
        
        Args:
            script_path: Path to the script file
            
        Returns:
            Result dictionary
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Failed to initialize Launchpad")
                return {"success": False, "error": "Failed to initialize Launchpad"}
        
        # Get command executor
        executor = self.registry.get_component("command_executor")
        if not executor:
            logger.error("Command Executor not available")
            return {"success": False, "error": "Command Executor not available"}
        
        # Read script file
        try:
            with open(script_path, "r") as f:
                script_content = f.read()
        except Exception as e:
            logger.error(f"Error reading script file: {e}")
            return {"success": False, "error": f"Error reading script file: {e}"}
        
        # Execute script
        try:
            result = executor.execute_script(script_content, "user")
            return result
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _start_auditor_daemon(self) -> None:
        """Start the Auditor Agent as a daemon process."""
        auditor_agent = self.registry.get_component("auditor_agent")
        if not auditor_agent:
            logger.warning("Auditor Agent not available, cannot start daemon")
            return
        
        try:
            logger.info("Starting Auditor Agent daemon...")
            result = auditor_agent.start_auditing()
            if result:
                logger.info("Auditor Agent daemon started successfully")
            else:
                logger.warning("Failed to start Auditor Agent daemon")
        except Exception as e:
            logger.error(f"Error starting Auditor Agent daemon: {e}")
            logger.error(traceback.format_exc())
    
    def shutdown(self) -> None:
        """Shutdown the Launchpad and all components."""
        if not self.initialized:
            return
        
        logger.info("Shutting down Launchpad")
        
        # Stop Auditor Agent daemon if running
        auditor_agent = self.registry.get_component("auditor_agent")
        if auditor_agent:
            try:
                logger.info("Stopping Auditor Agent daemon...")
                auditor_agent.stop_auditing()
            except Exception as e:
                logger.error(f"Error stopping Auditor Agent daemon: {e}")
        
        # Shutdown components
        self.registry.shutdown_components()
        
        self.initialized = False
        logger.info("Launchpad shutdown complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FixWurx LLM Shell")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive shell")
    parser.add_argument("-e", "--execute", help="Execute a command")
    parser.add_argument("-s", "--script", help="Execute a script file")
    args = parser.parse_args()
    
    # Initialize launchpad
    launchpad = Launchpad(args.config)
    
    # Execute command if provided
    if args.execute:
        result = launchpad.execute_command(args.execute)
        if result.get("success", False):
            print(result.get("output", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(0 if result.get("success", False) else 1)
    
    # Execute script if provided
    if args.script:
        result = launchpad.execute_script(args.script)
        if result.get("success", False):
            print(result.get("output", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(0 if result.get("success", False) else 1)
    
    # Start interactive shell if requested
    if args.interactive or (not args.execute and not args.script):
        launchpad.start_interactive()

if __name__ == "__main__":
    main()
