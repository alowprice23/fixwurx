#!/usr/bin/env python3
"""
Fix Auditor Components

This script leverages FixWurx's built-in repair capabilities to fix all
the auditor components through the FixWurx shell environment.
"""

import os
import sys
import subprocess
import logging
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [FixAuditor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('fix_auditor')

# List of auditor components to fix
AUDITOR_COMPONENTS = [
    "auditor.py",
    "auditor_agent.py",
    "system_auditor.py",
    "functionality_verification.py",
    "advanced_error_analysis.py",
    "benchmarking_system.py",
    "graph_database.py",
    "time_series_database.py",
    "document_store.py"
]

def run_fixwurx_command(command):
    """Run a FixWurx command through the shell environment"""
    try:
        # Use the appropriate command based on the OS
        if os.name == 'nt':  # Windows
            cmd = ["fx.bat"] + command.split()
        else:  # Unix/Linux/Mac
            cmd = ["./fx"] + command.split()
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Command failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
        
        logger.info(f"Command executed successfully")
        return True, result.stdout
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return False, str(e)

def run_comprehensive_fix(components):
    """Run a comprehensive fix on the specified components"""
    logger.info(f"Starting comprehensive fix for {len(components)} components...")
    
    # Use the comprehensive_fix module to fix all components at once
    success, output = run_fixwurx_command(f"fix --comprehensive --components {','.join(components)}")
    
    if not success:
        logger.error("Comprehensive fix failed")
        return False
    
    logger.info("Comprehensive fix completed successfully")
    return True

def repair_specific_component(component):
    """Repair a specific component using FixWurx repair functionality"""
    logger.info(f"Repairing component: {component}")
    
    # Use repair module to fix this specific component
    success, output = run_fixwurx_command(f"repair --component {component} --verify")
    
    if not success:
        logger.error(f"Failed to repair {component}")
        return False
    
    logger.info(f"Successfully repaired {component}")
    return True

def verify_fixes():
    """Verify that all fixes have been applied correctly"""
    logger.info("Verifying fixes...")
    
    # Run the test suite to verify the fixes
    success, output = run_fixwurx_command("test --components auditor --verbose")
    
    if not success:
        logger.error("Fix verification failed")
        return False
    
    if "FAILED" in output:
        logger.error("Some tests are still failing after fixes")
        return False
    
    logger.info("All fixes verified successfully")
    return True

def update_agentic_integration():
    """Update the integration with the LLM and agentic system"""
    logger.info("Updating agentic system integration...")
    
    # Run the command to update the agentic integration
    success, output = run_fixwurx_command("integrate --llm --agent-system --component auditor")
    
    if not success:
        logger.error("Agentic integration update failed")
        return False
    
    logger.info("Agentic integration updated successfully")
    return True

def main():
    """Main function"""
    logger.info("Starting auditor fix process through FixWurx shell environment...")
    
    # Step 1: Run a comprehensive fix on all auditor components
    if not run_comprehensive_fix(AUDITOR_COMPONENTS):
        logger.error("Comprehensive fix failed, trying individual component repair...")
        
        # If comprehensive fix fails, try repairing each component individually
        for component in AUDITOR_COMPONENTS:
            repair_specific_component(component)
    
    # Step 2: Specifically focus on functionality_verification.py and advanced_error_analysis.py
    # which were causing issues in previous attempts
    logger.info("Focusing on problematic components...")
    repair_specific_component("functionality_verification.py")
    repair_specific_component("advanced_error_analysis.py")
    
    # Step 3: Update the agentic system integration
    update_agentic_integration()
    
    # Step 4: Verify all fixes have been applied correctly
    if verify_fixes():
        logger.info("All auditor components have been successfully fixed!")
        return 0
    else:
        logger.error("Some fixes could not be verified. Please check the logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
