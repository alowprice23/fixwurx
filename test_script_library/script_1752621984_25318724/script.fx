#!/usr/bin/env bash
# Automated Grep Error Log Script
#
# This script was automatically generated from a detected command pattern.
#

# Error handling
set -e

# Define variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Command sequence
echo "Step 1: ls -la"
ls -la

echo "Step 2: grep 'error' log.txt"
grep 'error' log.txt

echo "Script completed successfully"