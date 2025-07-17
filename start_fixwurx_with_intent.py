#!/usr/bin/env python3
"""
Start FixWurx with Advanced Intent Classification

This script launches the FixWurx system with the advanced intent classification system
fully integrated with all components including the shell, agent system, neural matrix,
and triangulum.
"""

import os
import sys
import time
import logging
import argparse
import importlib
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fixwurx_startup.log"), logging.StreamHandler()]
)
logger = logging.getLogger("FixWurxStartup")

def print_banner():
    """Print the FixWurx startup banner."""
    banner = """
    ███████╗██╗██╗  ██╗██╗    ██╗██╗   ██╗██████╗ ██╗  ██╗
    ██╔════╝██║╚██╗██╔╝██║    ██║██║   ██║██╔══██╗╚██╗██╔╝
    █████╗  ██║ ╚███╔╝ ██║ █╗ ██║██║   ██║██████╔╝ ╚███╔╝ 
    ██╔══╝  ██║ ██╔██╗ ██║███╗██║██║   ██║██╔══██╗ ██╔██╗ 
    ██║     ██║██╔╝ ██╗╚███╔███╔╝╚██████╔╝██║  ██║██╔╝ ██╗
    ╚═╝     ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
                                                          
    =============== WITH ADVANCED INTENT CLASSIFICATION ===============
    """
    print(banner)

def load_fixwurx_components():
    """Load and initialize the core FixWurx components."""
    components = {}
    
    # Try to import the registry
    try:
        from components.component_registry import ComponentRegistry
        registry = ComponentRegistry()
        components["registry"] = registry
        logger.info("Loaded ComponentRegistry")
    except ImportError:
        # Create a simple registry if the real one isn't available
        class SimpleRegistry:
            def __init__(self):
                self.components = {}
                self.command_handlers = {}
                self.command_pipelines = {}
            
            def register_component(self, name, component):
                self.components[name] = component
                logger.info(f"Registered component: {name}")
            
            def get_component(self, name):
                return self.components.get(name)
                
            def register_command_handler(self, command, handler, category="default"):
                """Register a command handler for the shell."""
                if category not in self.command_handlers:
                    self.command_handlers[category] = {}
                self.command_handlers[category][command] = handler
                logger.info(f"Registered command handler: {command} in category {category}")
                return True
            
            def get_command_handler(self, command, category="default"):
                """Get a command handler by name and category."""
                if category in self.command_handlers and command in self.command_handlers[category]:
                    return self.command_handlers[category][command]
                return None
                
            def register_pipeline(self, name, pipeline):
                """Register a command pipeline."""
                self.command_pipelines[name] = pipeline
                logger.info(f"Registered command pipeline: {name}")
                return True
                
            def get_pipeline(self, name):
                """Get a command pipeline by name."""
                return self.command_pipelines.get(name)
        
        registry = SimpleRegistry()
        components["registry"] = registry
        logger.warning("Using SimpleRegistry as fallback")
    
    # Try to import shell environment
    try:
        from shell_environment import ShellEnvironment
        shell_env = ShellEnvironment(registry=registry)
        components["shell_environment"] = shell_env
        logger.info("Loaded ShellEnvironment")
    except ImportError:
        logger.warning("ShellEnvironment not available")
    
    # Try to import launchpad
    try:
        from launchpad import Launchpad
        try:
            # Try to initialize with registry parameter
            launchpad = Launchpad(registry=registry)
        except TypeError:
            # If that fails, try without registry parameter
            launchpad = Launchpad()
            # Try to set registry after initialization if possible
            if hasattr(launchpad, 'set_registry'):
                launchpad.set_registry(registry)
        components["launchpad"] = launchpad
        logger.info("Loaded Launchpad")
    except ImportError:
        logger.warning("Launchpad not available")
    
    # Try to import neural matrix
    try:
        from neural_matrix.core.neural_matrix import NeuralMatrix
        neural_matrix = NeuralMatrix()
        registry.register_component("neural_matrix", neural_matrix)
        components["neural_matrix"] = neural_matrix
        logger.info("Loaded NeuralMatrix")
    except ImportError:
        logger.warning("NeuralMatrix not available")
    
    # Try to import triangulum client
    try:
        from triangulum.client import TriangulumClient
        triangulum_client = TriangulumClient()
        registry.register_component("triangulum_client", triangulum_client)
        components["triangulum_client"] = triangulum_client
        logger.info("Loaded TriangulumClient")
    except ImportError:
        logger.warning("TriangulumClient not available")
    
    # Try to import agent system
    try:
        from agents.core.agent_system import AgentSystem
        try:
            # Try to initialize with registry parameter
            agent_system = AgentSystem(registry=registry)
        except TypeError:
            # If that fails, try without registry parameter
            agent_system = AgentSystem()
            # Try to set registry after initialization if possible
            if hasattr(agent_system, 'set_registry'):
                agent_system.set_registry(registry)
        registry.register_component("agent_system", agent_system)
        components["agent_system"] = agent_system
        logger.info("Loaded AgentSystem")
    except ImportError:
        logger.warning("AgentSystem not available")
    
    # Try to import file access utility
    try:
        from components.file_access_utility import FileAccessUtility
        try:
            # Try to initialize with registry parameter
            file_access = FileAccessUtility(registry=registry)
        except TypeError:
            # If that fails, try with just the registry as positional arg
            try:
                file_access = FileAccessUtility(registry)
            except:
                # If that also fails, try without parameters
                file_access = FileAccessUtility()
                
        registry.register_component("file_access_utility", file_access)
        components["file_access_utility"] = file_access
        logger.info("Loaded FileAccessUtility")
    except ImportError:
        logger.warning("FileAccessUtility not available")
    
    # Try to import command executor
    try:
        from components.command_executor import CommandExecutor
        try:
            # Try different initialization patterns
            try:
                command_executor = CommandExecutor(registry=registry)
            except TypeError:
                try:
                    command_executor = CommandExecutor(registry)
                except:
                    command_executor = CommandExecutor()
        except Exception as e:
            logger.warning(f"CommandExecutor initialization error: {e}")
            command_executor = None
            
        if command_executor:
            registry.register_component("command_executor", command_executor)
            components["command_executor"] = command_executor
            logger.info("Loaded CommandExecutor")
    except ImportError:
        logger.warning("CommandExecutor not available")
    
    # Try to import planning engine
    try:
        from components.planning_engine import PlanningEngine
        try:
            # Try different initialization patterns
            try:
                planning_engine = PlanningEngine(registry=registry)
            except TypeError:
                try:
                    planning_engine = PlanningEngine(registry)
                except:
                    planning_engine = PlanningEngine()
        except Exception as e:
            logger.warning(f"PlanningEngine initialization error: {e}")
            planning_engine = None
            
        if planning_engine:
            registry.register_component("planning_engine", planning_engine)
            components["planning_engine"] = planning_engine
            logger.info("Loaded PlanningEngine")
    except ImportError:
        logger.warning("PlanningEngine not available")
    
    return components

def integrate_intent_system(components):
    """Integrate the intent classification system with FixWurx components."""
    try:
        # Import intent system modules
        from intent_classification_system import initialize_system
        from components.intent_caching_system import IntentOptimizationSystem
        from fixwurx_intent_integration import integrate_with_fixwurx
        
        # Get registry
        registry = components.get("registry")
        if not registry:
            logger.error("Registry not available, intent system integration failed")
            return False
        
        # Initialize intent classification system
        intent_system = initialize_system(registry)
        logger.info("Initialized intent classification system")
        
        # Initialize intent optimization system
        optimization_system = IntentOptimizationSystem(
            cache_capacity=100,
            history_size=50,
            window_size=10
        )
        registry.register_component("intent_optimization_system", optimization_system)
        logger.info("Initialized intent optimization system")
        
        # Integrate with FixWurx
        shell_env = components.get("shell_environment")
        launchpad = components.get("launchpad")
        
        integration = integrate_with_fixwurx(shell_env, launchpad)
        logger.info("Integrated intent classification system with FixWurx")
        
        # Log integration stats
        if hasattr(integration, "get_stats"):
            stats = integration.get_stats()
            logger.info(f"Integration stats: {stats}")
        
        return True
    except ImportError as e:
        logger.error(f"Intent system integration failed: {e}")
        logger.error("Make sure intent_classification_system.py and related components are available")
        return False
    except Exception as e:
        logger.error(f"Error during intent system integration: {e}")
        return False

def launch_system(components, with_intent):
    """Launch the FixWurx system."""
    # Get shell environment
    shell_env = components.get("shell_environment")
    if not shell_env:
        logger.error("Shell environment not available, cannot launch system")
        return False
    
    # Get launchpad
    launchpad = components.get("launchpad")
    if not launchpad:
        logger.warning("Launchpad not available, some features may be limited")
    
    # Launch the system
    try:
        # Print system status
        print("\nSystem components:")
        for name, component in components.items():
            status = "LOADED" if component else "NOT AVAILABLE"
            print(f"  - {name}: {status}")
        
        print("\nIntent classification system:", "ENABLED" if with_intent else "DISABLED")
        
        # Start the shell
        if hasattr(shell_env, "start"):
            print("\nStarting FixWurx shell...")
            shell_env.start()
            return True
        elif hasattr(shell_env, "run"):
            print("\nStarting FixWurx shell...")
            shell_env.run()
            return True
        else:
            # Fallback to a simple interactive shell if native methods aren't available
            print("\nFallback to simple interactive shell (native shell methods not available)")
            return run_simple_interactive_shell(components, with_intent)
    except Exception as e:
        logger.error(f"Error launching system: {e}")
        return False

def run_simple_interactive_shell(components, with_intent):
    """Run a simple interactive shell when native shell methods aren't available."""
    registry = components.get("registry")
    intent_system = registry.get_component("intent_classification_system") if with_intent else None
    file_access = registry.get_component("file_access_utility")
    command_executor = registry.get_component("command_executor")
    
    print("\nSimple Interactive Shell")
    print("Type 'exit' or 'quit' to exit")
    print("Type 'help' for available commands")
    
    while True:
        try:
            user_input = input("\n> ")
            
            # Handle exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            # Handle help command
            elif user_input.lower() == "help":
                print("Available commands:")
                print("  exit, quit - Exit the shell")
                print("  help - Show this help message")
                print("  list <directory> - List files in directory")
                print("  read <file> - Read a file")
                print("  exec <command> - Execute a system command")
                if with_intent:
                    print("  Any other input will be processed by the intent classification system")
            
            # Handle list command
            elif user_input.lower().startswith("list "):
                path = user_input[5:].strip()
                if file_access:
                    result = file_access.list_directory(path)
                    if result.get("success"):
                        for item in result.get("items", []):
                            print(f"{'[DIR]' if item.get('is_dir') else '[FILE]'} {item.get('name')}")
                    else:
                        print(f"Error: {result.get('error')}")
                else:
                    print("File access utility not available")
            
            # Handle read command
            elif user_input.lower().startswith("read "):
                path = user_input[5:].strip()
                if file_access:
                    result = file_access.read_file(path)
                    if result.get("success"):
                        print(result.get("content"))
                    else:
                        print(f"Error: {result.get('error')}")
                else:
                    print("File access utility not available")
            
            # Handle exec command
            elif user_input.lower().startswith("exec "):
                cmd = user_input[5:].strip()
                if command_executor:
                    result = command_executor.execute(cmd)
                    if result.get("success"):
                        print(result.get("output"))
                    else:
                        print(f"Error: {result.get('error')}")
                else:
                    print("Command executor not available")
            
            # Process through intent system if available
            elif with_intent and intent_system:
                context = {"state": {"current_folder": os.getcwd()}}
                intent = intent_system.classify_intent(user_input, context)
                
                if intent:
                    print(f"Intent: {intent.type if hasattr(intent, 'type') else 'Unknown'}")
                    print(f"Parameters: {intent.parameters if hasattr(intent, 'parameters') else '{}'}")
                    print(f"Confidence: {intent.confidence if hasattr(intent, 'confidence') else 0}")
                    
                    # Here we would process the intent, but for this simple shell
                    # we just show the classification result
                    print("Intent processed (demonstration only)")
                else:
                    print("Could not classify intent")
            
            # Fallback
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start FixWurx with intent classification")
    parser.add_argument("--no-intent", action="store_true", help="Disable intent classification integration")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Load components
    print("Loading FixWurx components...")
    components = load_fixwurx_components()
    
    # Integrate intent system if enabled
    with_intent = not args.no_intent
    if with_intent:
        print("Integrating intent classification system...")
        success = integrate_intent_system(components)
        if not success:
            logger.warning("Intent system integration failed, continuing without intent classification")
            with_intent = False
    
    # Launch the system
    if args.demo:
        # Run a simple demo
        print("Running in demo mode...")
        
        # If intent system is available, show a demo
        if with_intent:
            try:
                from intent_system_bootstrap import run_demo
                run_demo()
            except ImportError:
                logger.error("Cannot run demo: intent_system_bootstrap.py not available")
        else:
            print("Intent classification system not available for demo")
    else:
        # Launch the full system
        print("Launching FixWurx system...")
        launch_system(components, with_intent)

if __name__ == "__main__":
    main()
