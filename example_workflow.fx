#!/usr/bin/env fx
#
# Example FixWurx Shell Script - Advanced Bug Detection and Resolution Workflow
# This script demonstrates a complete workflow for detecting, analyzing, and fixing bugs
#

# ---------- Initialize variables ----------
target_dir="."
log_file="workflow_results.log"
report_file="bug_report.json"
success_count=0
failure_count=0
total_bugs=0
max_fix_attempts=3
verbose=true

# ---------- Define utility functions ----------

function log(message)
    if $verbose == true then
        echo $message
    fi
    
    # Append to log file
    echo "$message" >> $log_file
    return 0
end

function check_directory(dir_path)
    # Verify the directory exists
    if exec test -d $dir_path then
        return 1
    else
        log "Error: Directory '$dir_path' does not exist"
        return 0
    fi
end

function initialize_workflow()
    log "===== Starting Bug Detection and Resolution Workflow ====="
    log "Target directory: $target_dir"
    log "Timestamp: $(date)"
    
    # Clear previous log file if it exists
    echo "# Bug Detection and Resolution Workflow Log" > $log_file
    echo "# Started at: $(date)" >> $log_file
    
    return 0
end

function analyze_and_fix_bug(bug_id)
    log "Processing bug ID: $bug_id"
    
    # Analyze the bug
    log "  Analyzing bug..."
    analyze_bug $bug_id --detailed > "bug_${bug_id}_analysis.txt"
    
    # Attempt to fix the bug
    attempts=0
    fixed=false
    
    while $attempts < $max_fix_attempts and $fixed == false do
        attempts=$attempts + 1
        log "  Fix attempt $attempts of $max_fix_attempts"
        
        # Try to fix the bug
        fix_status=fix_bug $bug_id --auto --force
        
        if $fix_status == 0 then
            log "  ✅ Bug fixed successfully on attempt $attempts"
            fixed=true
            success_count=$success_count + 1
        else
            log "  ❌ Fix attempt $attempts failed"
            
            # On last attempt, try with different strategy
            if $attempts == $max_fix_attempts - 1 then
                log "  Trying alternative fix strategy..."
                fix_status=fix_bug $bug_id --auto --force --strategy=alternative
            fi
        fi
    done
    
    if $fixed == false then
        log "  ❌ Failed to fix bug after $max_fix_attempts attempts"
        failure_count=$failure_count + 1
    fi
    
    return $fixed
end

function generate_report()
    log "Generating comprehensive report..."
    
    # Create report content
    report="{\n"
    report="${report}  \"workflow\": \"Bug Detection and Resolution\",\n"
    report="${report}  \"timestamp\": \"$(date)\",\n"
    report="${report}  \"target_directory\": \"$target_dir\",\n"
    report="${report}  \"total_bugs\": $total_bugs,\n"
    report="${report}  \"fixed_bugs\": $success_count,\n"
    report="${report}  \"failed_fixes\": $failure_count,\n"
    
    # Calculate success rate
    if $total_bugs > 0 then
        success_rate=($success_count * 100) / $total_bugs
    else
        success_rate=100
    fi
    
    report="${report}  \"success_rate\": $success_rate,\n"
    report="${report}  \"status\": "
    
    if $success_rate >= 80 then
        report="${report}\"SUCCESS\"\n"
    else
        report="${report}\"FAILURE\"\n"
    fi
    
    report="${report}}"
    
    # Save report to file
    echo $report > $report_file
    
    log "Report saved to $report_file"
    return 0
end

# ---------- Main script execution ----------

# Process command line args if provided
if $1 != "" then
    target_dir=$1
fi

if $2 == "--quiet" then
    verbose=false
fi

# Initialize workflow
initialize_workflow()

# Verify target directory
dir_check=check_directory($target_dir)
if $dir_check == 0 then
    log "Exiting due to invalid target directory"
    exit 1
fi

# Step 1: Detect bugs
log "Step 1: Detecting bugs in $target_dir"
detect_bugs $target_dir --save-results="detected_bugs.json"

# Get the list of detected bugs
bug_list=cat detected_bugs.json | grep "\"id\":" | grep -o "BUG-[0-9]*"
total_bugs=$(cat detected_bugs.json | grep "\"id\":" | wc -l)

log "Detected $total_bugs potential bugs"

# Step 2: Process each bug
log "Step 2: Analyzing and fixing bugs"

if $total_bugs > 0 then
    for bug_id in $bug_list do
        analyze_and_fix_bug($bug_id)
    done
else
    log "No bugs to fix - codebase is clean!"
fi

# Step 3: Generate verification tests
log "Step 3: Generating verification tests"
exec mkdir -p tests
verification --generate-tests --output="tests/" --bugs="detected_bugs.json"

# Step 4: Run verification tests
log "Step 4: Running verification tests"
test_result=exec python -m pytest tests/ -v

if $test_result == 0 then
    log "All verification tests passed!"
else
    log "Some verification tests failed"
fi

# Step 5: Generate final report
log "Step 5: Generating comprehensive report"
generate_report()

log "===== Workflow Complete ====="
log "Results summary:"
log "  Total bugs detected: $total_bugs"
log "  Successfully fixed: $success_count"
log "  Failed to fix: $failure_count"
log "  Success rate: $success_rate%"

# Print final status based on success rate
if $success_rate >= 80 then
    log "OVERALL STATUS: ✅ SUCCESS"
    exit 0
else
    log "OVERALL STATUS: ❌ FAILURE"
    exit 1
fi
