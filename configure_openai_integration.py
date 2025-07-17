#!/usr/bin/env python3
"""
OpenAI Integration Configuration

This script configures the FixWurx system to use the OpenAI API for LLM services.
It installs necessary dependencies, updates configuration files, and performs
a validation test to ensure the integration is working properly.
"""

import os
import sys
import json
import time
import logging
import subprocess
import argparse
import dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("openai_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("OpenAIIntegration")

def install_dependencies():
    """Install required Python packages."""
    logger.info("Installing required dependencies...")
    
    requirements = [
        "openai>=1.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    for req in requirements:
        logger.info(f"Installing {req}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            logger.info(f"Successfully installed {req}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {req}: {e}")
            return False
    
    return True

def update_config_file():
    """Update configuration files to use the real LLM client."""
    logger.info("Updating system configuration...")
    
    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Update or create config.json
    config_file = config_dir / "config.json"
    
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update LLM client configuration
        config["llm_client"] = {
            "module": "llm_client_real",
            "class": "LLMClientReal",
            "model": os.getenv("DEFAULT_MODEL", "gpt-4o"),
            "max_tokens": int(os.getenv("MAX_TOKENS", 4096)),
            "temperature": float(os.getenv("TEMPERATURE", 0.2))
        }
        
        # Write updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated configuration file: {config_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to update configuration file: {e}")
        return False

def validate_integration():
    """Validate the OpenAI integration by running a simple test."""
    logger.info("Validating OpenAI integration...")
    
    try:
        import openai
        from llm_client_real import LLMClientReal
        
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            return False
        
        # Create a test client
        client = LLMClientReal()
        
        # Test a simple prompt
        prompt = "Hello, this is a test of the OpenAI integration. Please respond with 'Integration successful'."
        response = client.generate(prompt)
        
        logger.info(f"OpenAI API Response: {response}")
        
        if "integration successful" in response.lower():
            logger.info("Integration validation successful!")
            return True
        else:
            logger.info("Integration validation completed, but response did not contain expected text.")
            logger.info("This is not necessarily an error, as the AI might respond differently.")
            return True
    except Exception as e:
        logger.error(f"Integration validation failed: {e}")
        return False

def main():
    """Main function to configure OpenAI integration."""
    parser = argparse.ArgumentParser(description="Configure OpenAI integration for FixWurx")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip integration validation")
    args = parser.parse_args()
    
    # Load environment variables
    dotenv.load_dotenv()
    
    logger.info("Starting OpenAI integration configuration...")
    
    # Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            logger.error("Failed to install dependencies. Configuration aborted.")
            return 1
    else:
        logger.info("Skipping dependency installation as requested.")
    
    # Update configuration
    if not update_config_file():
        logger.error("Failed to update configuration. Configuration aborted.")
        return 1
    
    # Validate integration
    if not args.skip_validation:
        if not validate_integration():
            logger.warning("Integration validation failed. You may need to check your API key or network connection.")
            logger.warning("Configuration completed with warnings.")
            return 0
    else:
        logger.info("Skipping integration validation as requested.")
    
    logger.info("OpenAI integration configuration completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
