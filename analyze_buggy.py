#!/usr/bin/env python3
"""
Standalone script to analyze sample_buggy.py for bugs.
"""

import os
import sys
import json
from pathlib import Path

# Ensure the bug_detection module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from bug_detection import analyze_code, apply_suggested_fixes
except ImportError:
    print("Error: Could not import bug_detection module")
    sys.exit(1)

def main():
    """Main function."""
    file_path = "sample_buggy.py"
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Analyzing file: {file_path}")
    result = analyze_code(file_path)
    
    if result.get("success", False):
        analysis = result["analysis"]
        print(f"\nAnalysis results for {file_path}:")
        print(f"Language: {analysis.get('language', 'unknown')}")
        print(f"Issues found: {analysis.get('total_issues', 0)}")
        print(f"Highest severity: {analysis.get('highest_severity', 'none')}")
        print(f"Analysis time: {analysis.get('analysis_time', 0):.2f} seconds")
        
        if analysis.get("issues"):
            print("\nIssues:")
            for i, issue in enumerate(analysis.get("issues", []), 1):
                print(f"\nIssue {i}:")
                print(f"  Description: {issue.get('description', 'Unknown')}")
                print(f"  Location: {issue.get('location', 'Unknown')}")
                print(f"  Severity: {issue.get('severity', 'Unknown')}")
                print(f"  Confidence: {issue.get('confidence', 0):.2f}")
                print(f"  Fix: {issue.get('fix', 'Unknown')}")
            
            # Ask if user wants to apply fixes
            if analysis.get("total_issues", 0) > 0:
                answer = input("\nWould you like to apply the suggested fixes? (y/n): ").lower()
                if answer == 'y':
                    fix_result = apply_suggested_fixes(result, auto_apply=False)
                    
                    if fix_result.get("success", False):
                        print(f"\nSuccessfully applied fixes to {file_path}")
                        print(f"Fixed {fix_result.get('issues_fixed', 0)} issues")
                        print(f"Backup saved to {file_path}.bak")
                    else:
                        print(f"\nFailed to apply fixes: {fix_result.get('error', 'Unknown error')}")
                else:
                    print("No fixes applied")
        else:
            print("No issues found")
        
        # Output to JSON file
        output_file = "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
        
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
