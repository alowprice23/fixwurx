#!/usr/bin/env python
"""
FixWurx - An AI-powered bug detection and fixing tool
Uses OpenAI's o3 model for code analysis and repair.
"""
import os
import sys
import argparse
import time
import json
import logging
import openai
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fixwurx_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FixWurx")

# Configure OpenAI API
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use environment variable for API key
openai.api_key = os.getenv("API_KEY")
MODEL = "o3"  # Using o3 model as specified

class FixWurx:
    def __init__(self, files, focus=None, comprehensive=False, auto_apply=True):
        """
        Initialize FixWurx with target files and options.
        
        Args:
            files (list): List of file paths to analyze
            focus (str, optional): Specific function to focus on
            comprehensive (bool): Whether to perform comprehensive analysis
            auto_apply (bool): Whether to automatically apply fixes without prompting (default: True)
        """
        self.files = files
        self.focus = focus
        self.comprehensive = comprehensive
        self.auto_apply = auto_apply
        self.error_reports = []
        self.success_reports = []
        self.start_time = datetime.now()
        
        # Create results directory
        self.results_dir = Path("fixwurx_results") / self.start_time.strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FixWurx initialized with files: {files}")
        if focus:
            logger.info(f"Focus set to: {focus}")
        if comprehensive:
            logger.info("Comprehensive analysis enabled")
        logger.info(f"Auto-apply mode: {'enabled' if auto_apply else 'disabled'}")
    
    def read_file(self, file_path):
        """Read the content of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            self.error_reports.append({
                "file": file_path,
                "error": f"File reading error: {str(e)}",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return None
    
    def write_file(self, file_path, content):
        """Write content to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            self.error_reports.append({
                "file": file_path,
                "error": f"File writing error: {str(e)}",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return False
    
    def analyze_file(self, file_path):
        """
        Analyze a single file for bugs using OpenAI's model.
        
        Args:
            file_path (str): Path to the file to analyze
            
        Returns:
            dict: Analysis results including identified bugs and fixes
        """
        logger.info(f"Analyzing file: {file_path}")
        content = self.read_file(file_path)
        if content is None:
            return None
        
        # Prepare analysis prompt
        focus_prompt = f" Focus on the function '{self.focus}'." if self.focus else ""
        
        try:
            # Get file extension for language detection
            file_ext = Path(file_path).suffix.lower()
            language = "python" if file_ext == ".py" else "unknown"
            
            prompt = f"""
            You are a highly skilled software engineer tasked with finding and fixing bugs in {language} code.
            
            Here is the code to analyze:{focus_prompt}
            
            ```{language}
            {content}
            ```
            
            Analyze this code for bugs, issues, or improvements. Focus on:
            1. Logical errors
            2. Input validation issues
            3. Error handling problems
            4. Performance issues
            5. Code structure and best practices
            
            For each issue found:
            1. Describe the bug or issue
            2. Explain why it's problematic
            3. Provide a detailed fix
            
            Format your response as JSON with:
            - "issues": array of found issues with "description", "reason", "fix"
            - "fixed_code": the complete corrected code
            """
            
            logger.info("Sending request to OpenAI API")
            start_time = time.time()
            
            # Make API call to OpenAI
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a code analysis and repair assistant that identifies bugs and provides fixes in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"OpenAI API response received in {elapsed_time:.2f} seconds")
            
            # Parse and return the analysis results
            try:
                analysis = json.loads(response.choices[0].message.content)
                return {
                    "file": file_path,
                    "analysis": analysis,
                    "time_taken": elapsed_time
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw response: {response.choices[0].message.content}")
                self.error_reports.append({
                    "file": file_path,
                    "error": f"JSON parsing error: {str(e)}",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            self.error_reports.append({
                "file": file_path,
                "error": f"Analysis error: {str(e)}",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return None
    
    def apply_fixes(self, file_path, fixed_code, issues):
        """
        Apply the suggested fixes to the file after user confirmation (if auto_apply is False).
        
        Args:
            file_path (str): Path to the file to fix
            fixed_code (str): The corrected code
            issues (list): List of identified issues
            
        Returns:
            bool: Whether the fixes were successfully applied
        """
        if not issues:
            logger.info(f"No issues found in {file_path}")
            return False
        
        logger.info(f"Found {len(issues)} issues in {file_path}")
        for i, issue in enumerate(issues, 1):
            print(f"\nIssue {i}: {issue['description']}")
            print(f"Reason: {issue['reason']}")
            print(f"Fix: {issue['fix']}")
        
        # In auto-apply mode, always apply fixes without prompting
        if self.auto_apply:
            confirm = 'y'
            logger.info("Auto-applying fixes")
        else:
            confirm = input("\nApply suggested fixes? (y/n): ").lower()
            
        if confirm != 'y':
            logger.info("User declined to apply fixes")
            return False
        
        # Create backup
        backup_path = str(file_path) + ".bak"
        try:
            with open(file_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            logger.info(f"Created backup at {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return False
        
        # Write fixed code
        success = self.write_file(file_path, fixed_code)
        if success:
            logger.info(f"Successfully applied fixes to {file_path}")
            self.success_reports.append({
                "file": file_path,
                "issues_fixed": len(issues),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        return success
    
    def run_tests(self, test_path=None):
        """
        Run the test suite to verify fixes.
        
        Args:
            test_path (str, optional): Specific test path to run
            
        Returns:
            bool: Whether all tests passed
        """
        import subprocess
        
        logger.info(f"Running tests: {test_path or 'all tests'}")
        
        cmd = ["python", "-m", "unittest"]
        if test_path:
            # Convert file path format to module format if needed
            if '/' in test_path:
                test_path = test_path.replace('/', '.').replace('.py', '')
            cmd.append(test_path)
        else:
            cmd.append("discover")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"Test results: Exit code {result.returncode}")
            
            # Write test output to results directory
            test_output_path = self.results_dir / "test_results.txt"
            with open(test_output_path, 'w', encoding='utf-8') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            
            print("\nTest Results:")
            print(result.stdout)
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)
            
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return False
    
    def analyze(self):
        """
        Main method to analyze all files and apply fixes.
        """
        logger.info("Starting analysis")
        total_issues = 0
        
        for file_path in self.files:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue
                
            analysis_result = self.analyze_file(file_path)
            
            if analysis_result:
                analysis = analysis_result["analysis"]
                
                if "issues" in analysis and "fixed_code" in analysis:
                    issues = analysis["issues"]
                    fixed_code = analysis["fixed_code"]
                    
                    # Save analysis to results directory
                    result_file = self.results_dir / f"{Path(file_path).name}_analysis.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_result, f, indent=2)
                    
                    if self.apply_fixes(file_path, fixed_code, issues):
                        total_issues += len(issues)
                else:
                    logger.error(f"Invalid analysis format for {file_path}")
        
        # Generate summary report
        self.generate_report(total_issues)
        
        return total_issues > 0
    
    def generate_report(self, total_issues):
        """
        Generate a summary report of the analysis.
        
        Args:
            total_issues (int): Total number of issues fixed
        """
        end_time = datetime.now()
        elapsed = (end_time - self.start_time).total_seconds()
        
        report = {
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "files_analyzed": len(self.files),
            "total_issues_fixed": total_issues,
            "success_reports": self.success_reports,
            "error_reports": self.error_reports
        }
        
        # Save report to results directory
        report_path = self.results_dir / "summary_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis completed in {elapsed:.2f} seconds")
        logger.info(f"Fixed {total_issues} issues across {len(self.files)} files")
        logger.info(f"Summary report saved to {report_path}")
        
        # Print summary to console
        print("\n========== FixWurx Analysis Summary ==========")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Files analyzed: {len(self.files)}")
        print(f"Total issues fixed: {total_issues}")
        print(f"Success reports: {len(self.success_reports)}")
        print(f"Error reports: {len(self.error_reports)}")
        print(f"Detailed report saved to: {report_path}")
        print("=============================================")


def main():
    """Main entry point for the FixWurx command line tool."""
    parser = argparse.ArgumentParser(description="FixWurx - AI-powered bug detection and fixing tool")
    parser.add_argument("--analyze", nargs="+", help="File(s) or directory to analyze")
    parser.add_argument("--focus", help="Specific function to focus on")
    parser.add_argument("--comprehensive", action="store_true", help="Perform comprehensive analysis")
    parser.add_argument("--run-tests", help="Run tests after fixing")
    parser.add_argument("--no-auto", action="store_true", help="Disable auto-apply mode (prompt for confirmation)")
    
    args = parser.parse_args()
    
    if args.analyze:
        auto_apply = not args.no_auto  # Default to auto-apply unless --no-auto is specified
        fixwurx = FixWurx(args.analyze, args.focus, args.comprehensive, auto_apply)
        fixwurx.analyze()
        
        if args.run_tests:
            fixwurx.run_tests(args.run_tests)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
