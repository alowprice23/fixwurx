#!/usr/bin/env python3
"""
LLM Client

This module provides a client for interacting with Large Language Models.
In production, this would connect to a real LLM service like OpenAI's GPT,
but for testing purposes, this mock implementation returns predefined responses.
"""

import os
import sys
import json
import time
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LLMClient")

class LLMClient:
    """
    Client for interacting with Large Language Models.
    
    This class provides methods for:
    - Generating text completions
    - Managing conversation context
    - Handling API communication
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.api_key = self.config.get("api_key", os.getenv("OPENAI_API_KEY", ""))
        self.model = self.config.get("model", "gpt-4")
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.initialized = False
        
        # Initialize predefined responses for mock mode
        self._init_mock_responses()
        
        logger.info(f"LLM Client initialized with model {self.model}")
        self.initialized = True
    
    def _init_mock_responses(self) -> None:
        """
        Initialize predefined responses for mock mode.
        """
        self.mock_responses = {
            # Goal deconstruction responses
            "goal_deconstruction": {
                "analyze_python_code": """
                1. Identify all Python files in the project
                2. For each file, check for syntax errors using Python's built-in parser
                3. Analyze import statements for unused or missing dependencies
                4. Check for common code smells like duplicated code, long functions, and deeply nested logic
                5. Generate a report of all identified issues
                6. Suggest fixes for the most critical issues
                """,
                "setup_web_server": """
                1. Check system requirements for the web server
                2. Install necessary dependencies
                3. Configure server settings
                4. Set up virtual hosts and domain routing
                5. Configure SSL certificates for secure connections
                6. Set up proper logging and monitoring
                7. Test the server configuration
                8. Start the web server service
                """,
                "backup_database": """
                1. Check available disk space for the backup
                2. Stop or lock the database to ensure consistent backup
                3. Export the database to a dump file
                4. Compress the dump file to save space
                5. Move the backup file to a secure location
                6. Unlock or restart the database
                7. Verify the integrity of the backup
                8. Update backup logs
                """,
                "default": """
                1. Analyze the current state of the system
                2. Identify the requirements to achieve the goal
                3. Gather necessary data and resources
                4. Execute the required steps in sequence
                5. Verify the outcome against expected results
                6. Clean up any temporary resources
                7. Report on the success or failure of the operation
                """
            },
            
            # Script generation responses
            "script_generation": {
                "analyze_python_code": """```fx
                # Python Code Analysis Script
                # Description: Analyzes Python code for syntax errors, unused imports, and code smells
                # Version: 1.0
                
                # Define output directory for reports
                output_dir="./analysis_reports"
                
                echo "Starting Python code analysis..."
                
                # Create output directory if it doesn't exist
                if [ ! -d "$output_dir" ]; then
                    mkdir -p "$output_dir"
                    echo "Created output directory: $output_dir"
                fi
                
                # Find all Python files
                echo "Finding Python files..."
                python_files=$(find . -name "*.py" | grep -v "__pycache__" | sort)
                
                if [ -z "$python_files" ]; then
                    echo "No Python files found!"
                    exit 1
                fi
                
                echo "Found $(echo "$python_files" | wc -l) Python files"
                
                # Check for syntax errors
                echo "Checking for syntax errors..."
                syntax_report="$output_dir/syntax_errors.txt"
                echo "Syntax Error Report" > "$syntax_report"
                echo "===================" >> "$syntax_report"
                echo "" >> "$syntax_report"
                
                syntax_errors=0
                for file in $python_files; do
                    python -m py_compile "$file" 2> /dev/null
                    if [ $? -ne 0 ]; then
                        echo "Error in $file" >> "$syntax_report"
                        python -m py_compile "$file" 2>> "$syntax_report"
                        echo "" >> "$syntax_report"
                        syntax_errors=$((syntax_errors + 1))
                    fi
                done
                
                echo "Found $syntax_errors files with syntax errors"
                
                # Check for unused imports
                echo "Checking for unused imports..."
                import_report="$output_dir/unused_imports.txt"
                echo "Unused Imports Report" > "$import_report"
                echo "====================" >> "$import_report"
                echo "" >> "$import_report"
                
                for file in $python_files; do
                    echo "Analyzing $file..."
                    python -m pyflakes "$file" | grep -i "imported but unused" >> "$import_report"
                done
                
                # Check for code smells
                echo "Checking for code smells..."
                smell_report="$output_dir/code_smells.txt"
                echo "Code Smells Report" > "$smell_report"
                echo "=================" >> "$smell_report"
                echo "" >> "$smell_report"
                
                for file in $python_files; do
                    echo "Analyzing $file for code smells..." >> "$smell_report"
                    
                    # Look for long functions (more than 50 lines)
                    long_functions=$(grep -n "def " "$file" | while read -r line; do
                        lineno=$(echo "$line" | cut -d: -f1)
                        function_name=$(echo "$line" | cut -d: -f2- | sed -E 's/.*def ([a-zA-Z0-9_]+).*/\1/')
                        end_lineno=$(tail -n +$lineno "$file" | grep -n "^[^ ]" | grep -v "^1:" | head -1 | cut -d: -f1)
                        if [ -z "$end_lineno" ]; then
                            end_lineno=$(wc -l < "$file")
                        else
                            end_lineno=$((lineno + end_lineno - 1))
                        fi
                        function_length=$((end_lineno - lineno))
                        if [ $function_length -gt 50 ]; then
                            echo "  - Long function: $function_name (lines $lineno-$end_lineno, $function_length lines)"
                        fi
                    done)
                    
                    if [ ! -z "$long_functions" ]; then
                        echo "$long_functions" >> "$smell_report"
                    fi
                    
                    # Look for deeply nested code (more than 4 levels of indentation)
                    deep_nesting=$(grep -n "^        " "$file" | head -5)
                    if [ ! -z "$deep_nesting" ]; then
                        echo "  - Deep nesting detected (more than 4 levels):" >> "$smell_report"
                        echo "$deep_nesting" | sed 's/^/    line /' >> "$smell_report"
                    fi
                    
                    echo "" >> "$smell_report"
                done
                
                # Generate summary report
                echo "Generating summary report..."
                summary_report="$output_dir/summary.txt"
                echo "Python Code Analysis Summary" > "$summary_report"
                echo "==========================" >> "$summary_report"
                echo "" >> "$summary_report"
                echo "Total Python files analyzed: $(echo "$python_files" | wc -l)" >> "$summary_report"
                echo "Files with syntax errors: $syntax_errors" >> "$summary_report"
                echo "Files with unused imports: $(grep -c "imported but unused" "$import_report")" >> "$summary_report"
                echo "" >> "$summary_report"
                echo "See detailed reports in the $output_dir directory:" >> "$summary_report"
                echo "  - syntax_errors.txt: Details on syntax errors" >> "$summary_report"
                echo "  - unused_imports.txt: Details on unused imports" >> "$summary_report"
                echo "  - code_smells.txt: Details on code smells" >> "$summary_report"
                
                echo "Analysis complete! See $output_dir for reports."
                ```""",
                
                "setup_web_server": """```fx
                # Web Server Setup Script
                # Description: Sets up and configures a web server with SSL
                # Version: 1.0
                
                # Check if running as root
                if [ "$(id -u)" -ne 0 ]; then
                    echo "This script must be run as root"
                    exit 1
                fi
                
                # Configuration variables
                SERVER_TYPE=${1:-"nginx"}  # Default to nginx if not specified
                DOMAIN=${2:-"example.com"}
                EMAIL=${3:-"admin@example.com"}
                WEB_ROOT=${4:-"/var/www/$DOMAIN"}
                
                echo "Starting web server setup for $DOMAIN using $SERVER_TYPE..."
                
                # Check system requirements
                echo "Checking system requirements..."
                
                # Check available memory
                TOTAL_MEM=$(free -m | grep "Mem:" | awk '{print $2}')
                echo "Total memory: $TOTAL_MEM MB"
                
                if [ $TOTAL_MEM -lt 512 ]; then
                    echo "Warning: Less than 512MB of RAM available. Server may perform poorly."
                fi
                
                # Check disk space
                DISK_SPACE=$(df -h / | tail -1 | awk '{print $4}')
                echo "Available disk space: $DISK_SPACE"
                
                # Install necessary packages
                echo "Installing necessary packages..."
                
                case $SERVER_TYPE in
                    nginx)
                        apt-get update
                        apt-get install -y nginx certbot python3-certbot-nginx
                        ;;
                    apache)
                        apt-get update
                        apt-get install -y apache2 certbot python3-certbot-apache
                        ;;
                    *)
                        echo "Unsupported server type: $SERVER_TYPE"
                        echo "Supported types: nginx, apache"
                        exit 1
                        ;;
                esac
                
                # Create web root directory
                echo "Creating web root directory: $WEB_ROOT"
                mkdir -p $WEB_ROOT
                
                # Create a basic index.html file
                echo "Creating basic index.html file..."
                cat > $WEB_ROOT/index.html << EOF
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Welcome to $DOMAIN</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            margin: 40px;
                            text-align: center;
                        }
                        h1 {
                            color: #333;
                        }
                    </style>
                </head>
                <body>
                    <h1>Welcome to $DOMAIN</h1>
                    <p>Your web server is successfully configured!</p>
                </body>
                </html>
                EOF
                
                # Set appropriate permissions
                echo "Setting appropriate permissions..."
                chown -R www-data:www-data $WEB_ROOT
                chmod -R 755 $WEB_ROOT
                
                # Configure server
                echo "Configuring $SERVER_TYPE for $DOMAIN..."
                
                case $SERVER_TYPE in
                    nginx)
                        # Create Nginx server block
                        cat > /etc/nginx/sites-available/$DOMAIN << EOF
                server {
                    listen 80;
                    listen [::]:80;
                
                    server_name $DOMAIN www.$DOMAIN;
                    root $WEB_ROOT;
                
                    index index.html index.htm index.nginx-debian.html;
                
                    location / {
                        try_files \$uri \$uri/ =404;
                    }
                }
                EOF
                        
                        # Enable the site
                        ln -s /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
                        
                        # Test Nginx configuration
                        nginx -t
                        
                        # If test passed, reload Nginx
                        if [ $? -eq 0 ]; then
                            systemctl reload nginx
                        else
                            echo "Nginx configuration test failed. Please check the configuration."
                            exit 1
                        fi
                        ;;
                        
                    apache)
                        # Create Apache virtual host
                        cat > /etc/apache2/sites-available/$DOMAIN.conf << EOF
                <VirtualHost *:80>
                    ServerName $DOMAIN
                    ServerAlias www.$DOMAIN
                    DocumentRoot $WEB_ROOT
                    
                    <Directory $WEB_ROOT>
                        Options -Indexes +FollowSymLinks
                        AllowOverride All
                        Require all granted
                    </Directory>
                    
                    ErrorLog \${APACHE_LOG_DIR}/$DOMAIN-error.log
                    CustomLog \${APACHE_LOG_DIR}/$DOMAIN-access.log combined
                </VirtualHost>
                EOF
                        
                        # Enable the site
                        a2ensite $DOMAIN.conf
                        
                        # Enable necessary modules
                        a2enmod rewrite
                        
                        # Test Apache configuration
                        apache2ctl configtest
                        
                        # If test passed, reload Apache
                        if [ $? -eq 0 ]; then
                            systemctl reload apache2
                        else
                            echo "Apache configuration test failed. Please check the configuration."
                            exit 1
                        fi
                        ;;
                esac
                
                # Set up SSL with Certbot
                echo "Setting up SSL certificates for $DOMAIN..."
                
                case $SERVER_TYPE in
                    nginx)
                        certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos -m $EMAIL
                        ;;
                    apache)
                        certbot --apache -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos -m $EMAIL
                        ;;
                esac
                
                # Set up logging and monitoring
                echo "Setting up logging and monitoring..."
                
                # Create a log rotation configuration
                cat > /etc/logrotate.d/$DOMAIN << EOF
                $WEB_ROOT/logs/*.log {
                    daily
                    missingok
                    rotate 14
                    compress
                    delaycompress
                    notifempty
                    create 0640 www-data www-data
                    sharedscripts
                    postrotate
                        [ ! -f /var/run/$SERVER_TYPE.pid ] || kill -USR1 \`cat /var/run/$SERVER_TYPE.pid\`
                    endscript
                }
                EOF
                
                # Test the server
                echo "Testing the server configuration..."
                
                # Check if the server is running
                systemctl is-active $SERVER_TYPE
                if [ $? -eq 0 ]; then
                    echo "$SERVER_TYPE is running."
                else
                    echo "Warning: $SERVER_TYPE is not running."
                    systemctl start $SERVER_TYPE
                fi
                
                # Print summary
                echo ""
                echo "========================================"
                echo "Web Server Setup Complete!"
                echo "========================================"
                echo "Domain: $DOMAIN"
                echo "Server Type: $SERVER_TYPE"
                echo "Web Root: $WEB_ROOT"
                echo "SSL: Enabled (via Let's Encrypt)"
                echo ""
                echo "Next steps:"
                echo "1. Upload your website files to $WEB_ROOT"
                echo "2. Set up database if needed"
                echo "3. Configure firewall to allow HTTP and HTTPS"
                echo "4. Set up regular backups"
                echo "========================================"
                ```""",
                
                "backup_database": """```fx
                # Database Backup Script
                # Description: Creates a compressed backup of a database and stores it securely
                # Version: 1.0
                
                # Configuration variables
                DB_TYPE=${1:-"mysql"}        # Database type (mysql, postgresql, mongodb)
                DB_NAME=${2:-"mydatabase"}   # Database name
                DB_USER=${3:-"dbuser"}       # Database user
                DB_PASS=${4:-"dbpassword"}   # Database password
                DB_HOST=${5:-"localhost"}    # Database host
                BACKUP_DIR=${6:-"./backups"} # Backup directory
                KEEP_DAYS=${7:-30}           # Number of days to keep backups
                
                # Generate timestamp for backup file
                TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
                BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql"
                COMPRESSED_FILE="${BACKUP_FILE}.gz"
                LOG_FILE="$BACKUP_DIR/backup_log.txt"
                
                # Create backup directory if it doesn't exist
                if [ ! -d "$BACKUP_DIR" ]; then
                    mkdir -p "$BACKUP_DIR"
                    echo "Created backup directory: $BACKUP_DIR"
                fi
                
                # Log function
                log_message() {
                    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
                }
                
                # Check available disk space
                log_message "Checking available disk space..."
                AVAILABLE_SPACE=$(df -h "$BACKUP_DIR" | awk 'NR==2 {print $4}')
                log_message "Available space in backup directory: $AVAILABLE_SPACE"
                
                # Get database size
                case $DB_TYPE in
                    mysql)
                        DB_SIZE=$(mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" -e "SELECT table_schema AS 'Database', ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'Size (MB)' FROM information_schema.TABLES WHERE table_schema = '$DB_NAME' GROUP BY table_schema;" | tail -1 | awk '{print $2}')
                        if [ -z "$DB_SIZE" ]; then
                            log_message "Error: Could not determine database size. Check credentials and database name."
                            DB_SIZE="unknown"
                        else
                            log_message "Database size: $DB_SIZE MB"
                        fi
                        ;;
                    postgresql)
                        DB_SIZE=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" | grep -v -e "----" -e "row" | tr -d '([:space:])' | sed 's/[^0-9.]//g')
                        if [ -z "$DB_SIZE" ]; then
                            log_message "Error: Could not determine database size. Check credentials and database name."
                            DB_SIZE="unknown"
                        else
                            log_message "Database size: $DB_SIZE MB"
                        fi
                        ;;
                    mongodb)
                        DB_SIZE=$(mongo --host "$DB_HOST" --username "$DB_USER" --password "$DB_PASS" --eval "db.stats().dataSize / (1024*1024)" "$DB_NAME" | grep -v -e "MongoDB shell" -e "connecting" | tail -1 | tr -d '[:space:]')
                        if [ -z "$DB_SIZE" ]; then
                            log_message "Error: Could not determine database size. Check credentials and database name."
                            DB_SIZE="unknown"
                        else
                            log_message "Database size: $DB_SIZE MB"
                        fi
                        ;;
                    *)
                        log_message "Error: Unsupported database type: $DB_TYPE"
                        log_message "Supported types: mysql, postgresql, mongodb"
                        exit 1
                        ;;
                esac
                
                # Perform database backup
                log_message "Starting backup of $DB_NAME database..."
                
                case $DB_TYPE in
                    mysql)
                        # Lock tables to ensure consistent backup
                        log_message "Locking tables..."
                        mysqladmin -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" flush-tables
                        
                        # Perform the backup
                        log_message "Dumping database to $BACKUP_FILE..."
                        mysqldump -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" --single-transaction --routines --triggers --events "$DB_NAME" > "$BACKUP_FILE"
                        BACKUP_STATUS=$?
                        
                        # Unlock tables
                        log_message "Unlocking tables..."
                        mysqladmin -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" flush-tables
                        ;;
                    postgresql)
                        # Perform the backup
                        log_message "Dumping database to $BACKUP_FILE..."
                        PGPASSWORD="$DB_PASS" pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f "$BACKUP_FILE"
                        BACKUP_STATUS=$?
                        ;;
                    mongodb)
                        # Perform the backup
                        log_message "Dumping database to $BACKUP_FILE..."
                        mongodump --host "$DB_HOST" --username "$DB_USER" --password "$DB_PASS" --db "$DB_NAME" --out "$BACKUP_DIR/temp_mongo_dump"
                        BACKUP_STATUS=$?
                        
                        # Convert to a single archive file
                        tar -czf "$BACKUP_FILE" -C "$BACKUP_DIR/temp_mongo_dump" .
                        rm -rf "$BACKUP_DIR/temp_mongo_dump"
                        ;;
                esac
                
                # Check if backup was successful
                if [ $BACKUP_STATUS -ne 0 ]; then
                    log_message "Error: Database backup failed with status $BACKUP_STATUS"
                    exit 1
                fi
                
                # Compress the backup
                log_message "Compressing backup file..."
                gzip -f "$BACKUP_FILE"
                COMPRESSION_STATUS=$?
                
                if [ $COMPRESSION_STATUS -ne 0 ]; then
                    log_message "Error: Compression failed with status $COMPRESSION_STATUS"
                    exit 1
                fi
                
                # Verify the integrity of the backup
                log_message "Verifying backup integrity..."
                
                if [ -f "$COMPRESSED_FILE" ]; then
                    BACKUP_SIZE=$(du -h "$COMPRESSED_FILE" | awk '{print $1}')
                    log_message "Backup file size: $BACKUP_SIZE"
                    
                    # Test the compressed file
                    gzip -t "$COMPRESSED_FILE"
                    if [ $? -eq 0 ]; then
                        log_message "Backup integrity verified."
                    else
                        log_message "Error: Backup file is corrupted."
                        exit 1
                    fi
                else
                    log_message "Error: Compressed backup file not found."
                    exit 1
                fi
                
                # Calculate MD5 checksum
                if command -v md5sum > /dev/null; then
                    MD5_CHECKSUM=$(md5sum "$COMPRESSED_FILE" | awk '{print $1}')
                    log_message "MD5 Checksum: $MD5_CHECKSUM"
                    echo "$MD5_CHECKSUM" > "${COMPRESSED_FILE}.md5"
                fi
                
                # Remove old backups
                log_message "Removing backups older than $KEEP_DAYS days..."
                find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -type f -mtime +$KEEP_DAYS -delete
                find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz.md5" -type f -mtime +$KEEP_DAYS -delete
                
                # Final summary
                log_message "Backup completed successfully!"
                log_message "Database: $DB_NAME"
                log_message "Backup file: $COMPRESSED_FILE"
                log_message "File size: $BACKUP_SIZE"
                log_message "=============================================="
                
                echo ""
                echo "Database backup completed successfully!"
                echo "Backup saved to: $COMPRESSED_FILE"
                echo "Backup size: $BACKUP_SIZE"
                echo ""
                ```""",
                
                "default": """```fx
                # Generic Task Script
                # Description: Executes a generic task with input validation and error handling
                # Version: 1.0
                
                # Configuration variables
                TASK_NAME="${1:-Generic Task}"
                RESOURCE_DIR="${2:-./resources}"
                OUTPUT_DIR="${3:-./output}"
                
                # Log file setup
                LOG_DIR="./logs"
                LOG_FILE="$LOG_DIR/task_execution_$(date +%Y%m%d_%H%M%S).log"
                
                # Create directories if they don't exist
                for DIR in "$RESOURCE_DIR" "$OUTPUT_DIR" "$LOG_DIR"; do
                    if [ ! -d "$DIR" ]; then
                        mkdir -p "$DIR"
                        echo "Created directory: $DIR"
                    fi
                done
                
                # Logging function
                log() {
                    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
                    echo "[$timestamp] $1" | tee -a "$LOG_FILE"
                }
                
                # Error handling function
                handle_error() {
                    log "ERROR: $1"
                    log "Task failed with exit code $2"
                    exit $2
                }
                
                # Validate inputs and environment
                validate_environment() {
                    log "Validating environment..."
                    
                    # Check resource directory
                    if [ ! -d "$RESOURCE_DIR" ]; then
                        handle_error "Resource directory '$RESOURCE_DIR' does not exist" 1
                    fi
                    
                    # Check if output directory is writable
                    if [ ! -w "$OUTPUT_DIR" ]; then
                        handle_error "Output directory '$OUTPUT_DIR' is not writable" 2
                    fi
                    
                    # Check for required tools
                    for TOOL in grep awk sed; do
                        if ! command -v $TOOL &> /dev/null; then
                            handle_error "Required tool '$TOOL' is not installed" 3
                        fi
                    done
                    
                    log "Environment validation completed successfully"
                }
                
                # Main task execution function
                execute_task() {
                    log "Starting task: $TASK_NAME"
                    
                    # Step 1: Gather system information
                    log "Step 1: Gathering system information..."
                    SYSTEM_INFO="$OUTPUT_DIR/system_info.txt"
                    
                    {
                        echo "Task: $TASK_NAME"
                        echo "Date: $(date)"
                        echo "System: $(uname -a)"
                        echo "User: $(whoami)"
                        echo "Directory: $(pwd)"
                        echo "Resource Dir: $RESOURCE_DIR"
                        echo "Output Dir: $OUTPUT_DIR"
                        echo "------------------------"
                    } > "$SYSTEM_INFO"
                    
                    log "System information saved to $SYSTEM_INFO"
                    
                    # Step 2: Process resources
                    log "Step 2: Processing resources..."
                    RESOURCE_COUNT=$(ls -1 "$RESOURCE_DIR" | wc -l)
                    log "Found $RESOURCE_COUNT resources to process"
                    
                    if [ $RESOURCE_COUNT -eq 0 ]; then
                        log "Warning: No resources found in $RESOURCE_DIR"
                    else
                        PROCESSED_COUNT=0
                        for RESOURCE in "$RESOURCE_DIR"/*; do
                            if [ -f "$RESOURCE" ]; then
                                RESOURCE_NAME=$(basename "$RESOURCE")
                                log "Processing $RESOURCE_NAME..."
                                
                                # Simple processing example (copy with timestamp)
                                OUTPUT_FILE="$OUTPUT_DIR/processed_${RESOURCE_NAME}_$(date +%s)"
                                cp "$RESOURCE" "$OUTPUT_FILE"
                                
                                if [ $? -eq 0 ]; then
                                    log "Successfully processed $RESOURCE_NAME"
                                    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
                                else
                                    log "Warning: Failed to process $RESOURCE_NAME"
                                fi
                            fi
                        done
                        
                        log "Processed $PROCESSED_COUNT out of $RESOURCE_COUNT resources"
                    fi
                    
                    # Step 3: Generate report
                    log "Step 3: Generating report..."
                    REPORT_FILE="$OUTPUT_DIR/task_report.txt"
                    
                    {
                        echo "Task Report: $TASK_NAME"
                        echo "Generated: $(date)"
                        echo "------------------------"
                        echo "Resources found: $RESOURCE_COUNT"
                        echo "Resources processed: $PROCESSED_COUNT"
                        echo "Success rate: $(( RESOURCE_COUNT > 0 ? PROCESSED_COUNT * 100 / RESOURCE_COUNT : 100 ))%"
                        echo "------------------------"
                        echo "Output files:"
                        ls -la "$OUTPUT_DIR" | grep -v "task_report.txt" | tail -n +2
                    } > "$REPORT_FILE"
                    
                    log "Report generated at $REPORT_FILE"
                }
                
                # Cleanup function
                cleanup() {
                    log "Performing cleanup..."
                    
                    # Remove temporary files
                    find "$OUTPUT_DIR" -name "tmp_*" -type f -delete
                    
                    log "Cleanup completed"
                }
                
                # Main execution flow
                main() {
                    log "Starting execution of $TASK_NAME"
                    
                    # Trap errors
                    trap 'handle_error "Received signal to terminate" $?' TERM INT
                    
                    # Execute steps
                    validate_environment || exit $?
                    execute_task || exit $?
                    cleanup || exit $?
                    
                    log "Task $TASK_NAME completed successfully"
                    echo ""
                    echo "Task completed successfully. See log at $LOG_FILE"
                    echo "Report available at $OUTPUT_DIR/task_report.txt"
                    echo ""
                }
                
                # Run the main function
                main
                ```"""
            },
            
            # Script matching responses
            "script_matching": {
                "match": "MATCH: backup_script",
                "no_match": "NO_MATCH"
            }
        }
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """
        Generate text based on a prompt.
        
        In production, this would call an external LLM API, but for testing,
        it returns predefined responses based on pattern matching.
        
        Args:
            prompt: Prompt string
            temperature: Temperature parameter for generation randomness
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text string
        """
        max_tokens = max_tokens or self.max_tokens
        logger.info(f"Generating text with temperature={temperature}, max_tokens={max_tokens}")
        
        # For testing purposes, use mock responses based on pattern matching
        
        # Check for goal deconstruction prompts
        if "Goal Deconstruction" in prompt:
            logger.info("Detected goal deconstruction prompt")
            
            if "analyze" in prompt.lower() and "python" in prompt.lower():
                return self.mock_responses["goal_deconstruction"]["analyze_python_code"]
            elif "setup" in prompt.lower() and "web server" in prompt.lower():
                return self.mock_responses["goal_deconstruction"]["setup_web_server"]
            elif "backup" in prompt.lower() and "database" in prompt.lower():
                return self.mock_responses["goal_deconstruction"]["backup_database"]
            else:
                return self.mock_responses["goal_deconstruction"]["default"]
        
        # Check for script generation prompts
        elif "Script Generation" in prompt:
            logger.info("Detected script generation prompt")
            
            if "analyze" in prompt.lower() and "python" in prompt.lower():
                return self.mock_responses["script_generation"]["analyze_python_code"]
            elif "setup" in prompt.lower() and "web server" in prompt.lower():
                return self.mock_responses["script_generation"]["setup_web_server"]
            elif "backup" in prompt.lower() and "database" in prompt.lower():
                return self.mock_responses["script_generation"]["backup_database"]
            else:
                return self.mock_responses["script_generation"]["default"]
        
        # Check for script matching prompts
        elif "Script Matching" in prompt:
            logger.info("Detected script matching prompt")
            
            if "backup" in prompt.lower() and "script_backup" in prompt:
                return self.mock_responses["script_matching"]["match"]
            else:
                return self.mock_responses["script_matching"]["no_match"]
        
        # Check for script fixing prompts
        elif "Script Fixing" in prompt:
            logger.info("Detected script fixing prompt")
            return self.mock_responses["script_generation"]["default"]
        
        # Default response for unrecognized prompts
        logger.warning("No matching mock response found for prompt")
        return "I'm sorry, I don't have a specific response for that prompt."

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a chat completion based on a series of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Temperature parameter for generation randomness
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with response information
        """
        # Extract the last user message as the prompt
        last_user_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user_message = message.get("content", "")
                break
        
        if not last_user_message:
            logger.warning("No user message found in the conversation")
            response_text = "I'm sorry, I couldn't find your question or request."
        else:
            # Generate a response based on the last user message
            response_text = self.generate(last_user_message, temperature, max_tokens)
        
        # Simulate API response format
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(last_user_message) // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(last_user_message) + len(response_text)) // 4
            }
        }

# Helper function to get the singleton instance
_instance = None

def get_instance(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """
    Get the singleton instance of the LLM client.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LLMClient instance
    """
    global _instance
    if _instance is None:
        _instance = LLMClient(config)
    return _instance


# Example usage
if __name__ == "__main__":
    client = get_instance()
    
    # Example goal deconstruction
    goal = "Analyze Python code in the project for syntax errors and other issues"
    prompt = f"""
    # Goal Deconstruction
    
    ## User Goal
    {goal}
    
    ## Available Commands and Tools
    - command1: Description
    - command2: Description
    
    ## Task
    Break down the user's goal into a logical sequence of steps.
    
    ## Steps for: {goal}
    """
    
    response = client.generate(prompt)
    print("\nGoal Deconstruction Response:")
    print(response)
    
    # Example script generation
    prompt = f"""
    # FixWurx Script Generation
    
    ## User Goal
    {goal}
    
    ## Steps to Accomplish
    1. Find all Python files
    2. Check syntax
    
    ## Available Commands
    - command1: Description
    - command2: Description
    
    ## Task
    Generate a complete, well-commented .fx script.
    """
    
    response = client.generate(prompt)
    print("\nScript Generation Response:")
    print(response[:100] + "...")  # Print just the start
