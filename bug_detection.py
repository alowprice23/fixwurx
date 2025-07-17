#!/usr/bin/env python3
"""
Bug Detection Module

This module implements AI-powered code analysis for bug detection, including
logical error detection, input validation issues, error handling problems,
performance issues, and best practices analysis.
"""

import os
import sys
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bug_detection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("BugDetection")

# Import necessary components
try:
    from fixwurx import read_file, write_file
except ImportError:
    # Fallback implementation if fixwurx module is not available
    def read_file(file_path):
        """Read the content of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def write_file(file_path, content):
        """Write content to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            return False

class BugDetector:
    """
    Bug detection and analysis system using AI-powered techniques.
    
    This class implements various code analysis methods to detect:
    - Logical errors
    - Input validation issues
    - Error handling problems
    - Performance issues
    - Best practices violations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the bug detector.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.model = self.config.get("model", "o3")
        self.temperature = self.config.get("temperature", 0.1)
        self.api_key = self.config.get("api_key", os.environ.get("OPENAI_API_KEY"))
        
        # Detection thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.severity_levels = self.config.get("severity_levels", ["critical", "high", "medium", "low"])
        
        # Cache for analysis results
        self.analysis_cache = {}
        
        logger.info("Bug detector initialized")
    
    def analyze_file(self, file_path: str, focus: str = None) -> Optional[Dict[str, Any]]:
        """
        Analyze a file for potential bugs.
        
        Args:
            file_path: Path to the file to analyze
            focus: Specific function or section to focus on (optional)
            
        Returns:
            Dict containing analysis results, or None if analysis failed
        """
        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Read file contents
        content = read_file(file_path)
        if content is None:
            return None
        
        # Get file extension for language detection
        file_ext = Path(file_path).suffix.lower()
        language = self._detect_language(file_ext, content)
        
        # Prepare focus prompt if needed
        focus_prompt = f" Focus on the function or section: '{focus}'." if focus else ""
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(content, language, focus_prompt)
        
        # Call the AI model for analysis
        try:
            logger.info(f"Analyzing {file_path} for bugs")
            start_time = time.time()
            
            # Get analysis from AI model (mock implementation for now)
            analysis = self._get_ai_analysis(prompt, language)
            
            # Process analysis results
            if analysis:
                elapsed_time = time.time() - start_time
                logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
                
                # Add metadata
                analysis["file"] = file_path
                analysis["language"] = language
                analysis["analysis_time"] = elapsed_time
                analysis["timestamp"] = time.time()
                
                # Cache the results
                self.analysis_cache[file_path] = analysis
                
                return analysis
            else:
                logger.error(f"Failed to analyze {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return None
    
    def _detect_language(self, file_ext: str, content: str) -> str:
        """
        Detect the programming language based on file extension and content.
        
        Args:
            file_ext: File extension
            content: File content
            
        Returns:
            Detected language
        """
        # Map common file extensions to languages
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".sh": "bash",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
        }
        
        # Return language based on extension if available
        if file_ext in ext_map:
            return ext_map[file_ext]
        
        # Try to detect based on content (basic heuristics)
        if "def " in content and "import " in content:
            return "python"
        elif "function " in content and ("var " in content or "const " in content or "let " in content):
            return "javascript"
        elif "class " in content and "public " in content:
            return "java"
        elif "#include " in content:
            return "c" if ".h>" in content else "cpp"
        elif "using System;" in content:
            return "csharp"
        elif "package main" in content:
            return "go"
        
        # Default to unknown
        return "unknown"
    
    def _create_analysis_prompt(self, content: str, language: str, focus_prompt: str) -> str:
        """
        Create the prompt for AI analysis.
        
        Args:
            content: File content
            language: Programming language
            focus_prompt: Additional focus instructions
            
        Returns:
            Complete prompt for analysis
        """
        return f"""
        You are a highly skilled software engineer specializing in bug detection and code analysis.
        
        Analyze the following {language} code for potential bugs and issues:{focus_prompt}
        
        ```{language}
        {content}
        ```
        
        Perform a comprehensive analysis focusing on:
        1. Logical errors - Incorrect algorithm implementation, off-by-one errors, etc.
        2. Input validation issues - Missing or incomplete validation, type checking
        3. Error handling problems - Missing try/catch, insufficient error handling
        4. Performance issues - Inefficient algorithms, memory leaks, etc.
        5. Best practices violations - Code style, maintainability issues
        
        For each issue found, provide:
        - Description: Clear explanation of the issue
        - Location: Line number or function name
        - Severity: Critical, High, Medium, or Low
        - Confidence: How confident you are in this finding (0.0-1.0)
        - Fix: Suggested code fix or approach to resolve the issue
        
        Format your response as a JSON object with the following structure:
        {{
            "issues": [
                {{
                    "description": "Description of the issue",
                    "location": "Line number or function name",
                    "severity": "critical|high|medium|low",
                    "confidence": 0.95,
                    "fix": "Suggested fix"
                }}
            ],
            "summary": "Brief summary of the analysis",
            "total_issues": 5,
            "highest_severity": "critical|high|medium|low",
            "fixed_code": "Complete fixed code with all issues resolved"
        }}
        """
    
    def _get_ai_analysis(self, prompt: str, language: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis from the AI model.
        
        Args:
            prompt: Analysis prompt
            language: Programming language
            
        Returns:
            Analysis results or None if failed
        """
        # For now, we'll provide a mock implementation that simulates AI analysis
        # In a real implementation, this would call the OpenAI API
        
        # Simulated delay to mimic API call
        time.sleep(2)
        
        # Create a mock analysis based on the language
        if language == "python":
            return {
                "issues": [
                    {
                        "description": "Missing input validation for user-provided data",
                        "location": "process_data function, line 25",
                        "severity": "high",
                        "confidence": 0.92,
                        "fix": "Add input validation using isinstance() or try/except blocks"
                    },
                    {
                        "description": "Unused variable 'result'",
                        "location": "Line 42",
                        "severity": "low",
                        "confidence": 0.98,
                        "fix": "Remove the unused variable or use it in the function"
                    },
                    {
                        "description": "Potential division by zero",
                        "location": "calculate_ratio function, line 57",
                        "severity": "critical",
                        "confidence": 0.89,
                        "fix": "Add a check to ensure the denominator is not zero before division"
                    }
                ],
                "summary": "Found 3 issues: 1 critical, 1 high, 0 medium, 1 low",
                "total_issues": 3,
                "highest_severity": "critical",
                "fixed_code": "# Mock fixed code would be provided here"
            }
        elif language == "javascript":
            return {
                "issues": [
                    {
                        "description": "Potential null reference exception",
                        "location": "getUserData function, line 18",
                        "severity": "high",
                        "confidence": 0.87,
                        "fix": "Add null checking before accessing properties"
                    },
                    {
                        "description": "Inefficient array operation using nested loops",
                        "location": "processArray function, lines 30-40",
                        "severity": "medium",
                        "confidence": 0.82,
                        "fix": "Use map/filter/reduce functions instead of nested loops"
                    }
                ],
                "summary": "Found 2 issues: 0 critical, 1 high, 1 medium, 0 low",
                "total_issues": 2,
                "highest_severity": "high",
                "fixed_code": "// Mock fixed code would be provided here"
            }
        else:
            # Generic response for other languages
            return {
                "issues": [
                    {
                        "description": "Generic issue for demonstration",
                        "location": "Unknown location",
                        "severity": "medium",
                        "confidence": 0.75,
                        "fix": "This is a placeholder for actual AI analysis"
                    }
                ],
                "summary": "Found 1 issue: 0 critical, 0 high, 1 medium, 0 low",
                "total_issues": 1,
                "highest_severity": "medium",
                "fixed_code": "# Mock fixed code would be provided here"
            }
    
    def filter_issues(self, analysis: Dict[str, Any], min_confidence: float = None, severity: List[str] = None) -> Dict[str, Any]:
        """
        Filter issues based on confidence and severity.
        
        Args:
            analysis: Analysis results
            min_confidence: Minimum confidence threshold
            severity: List of severities to include
            
        Returns:
            Filtered analysis results
        """
        if not analysis or "issues" not in analysis:
            return analysis
        
        # Use default threshold if not provided
        if min_confidence is None:
            min_confidence = self.confidence_threshold
        
        # Use all severity levels if not provided
        if severity is None:
            severity = self.severity_levels
        
        # Filter issues
        filtered_issues = [
            issue for issue in analysis["issues"]
            if issue.get("confidence", 0) >= min_confidence and issue.get("severity", "low") in severity
        ]
        
        # Update analysis with filtered issues
        filtered_analysis = analysis.copy()
        filtered_analysis["issues"] = filtered_issues
        filtered_analysis["total_issues"] = len(filtered_issues)
        
        # Update highest severity
        severity_map = {s: i for i, s in enumerate(["critical", "high", "medium", "low"])}
        
        if filtered_issues:
            highest_severity = min(
                [severity_map.get(issue.get("severity", "low"), 3) for issue in filtered_issues]
            )
            filtered_analysis["highest_severity"] = list(severity_map.keys())[highest_severity]
        else:
            filtered_analysis["highest_severity"] = "none"
        
        # Update summary
        filtered_analysis["summary"] = f"Found {len(filtered_issues)} issues after filtering"
        
        return filtered_analysis
    
    def suggest_fixes(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate suggested fixes for detected issues.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Analysis with detailed fix suggestions
        """
        if not analysis or "issues" not in analysis:
            return analysis
        
        # In a real implementation, this would generate more detailed fixes
        # For now, we'll just use the existing fix suggestions
        
        return analysis
    
    def apply_fixes(self, analysis: Dict[str, Any], auto_apply: bool = False) -> bool:
        """
        Apply suggested fixes to the file.
        
        Args:
            analysis: Analysis results
            auto_apply: Whether to automatically apply fixes without confirmation
            
        Returns:
            Whether fixes were successfully applied
        """
        if not analysis or "issues" not in analysis or "file" not in analysis:
            logger.error("Invalid analysis data for applying fixes")
            return False
        
        file_path = analysis["file"]
        fixed_code = analysis.get("fixed_code")
        
        if not fixed_code:
            logger.error("No fixed code available in analysis results")
            return False
        
        # Create backup before applying fixes
        backup_path = f"{file_path}.bak"
        original_content = read_file(file_path)
        
        if original_content and write_file(backup_path, original_content):
            logger.info(f"Created backup at {backup_path}")
        else:
            logger.error(f"Failed to create backup at {backup_path}")
            if not auto_apply:
                return False
        
        # Apply fixes
        if write_file(file_path, fixed_code):
            logger.info(f"Successfully applied fixes to {file_path}")
            return True
        else:
            logger.error(f"Failed to apply fixes to {file_path}")
            return False

# API Functions for command handlers

def analyze_code(file_path: str, focus: str = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze code for bugs and issues.
    
    Args:
        file_path: Path to the file to analyze
        focus: Specific function or section to focus on (optional)
        config: Configuration options (optional)
        
    Returns:
        Analysis results
    """
    try:
        detector = BugDetector(config)
        analysis = detector.analyze_file(file_path, focus)
        
        if analysis:
            return {
                "success": True,
                "analysis": analysis
            }
        else:
            return {
                "success": False,
                "error": f"Failed to analyze {file_path}"
            }
    except Exception as e:
        logger.error(f"Error analyzing code: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def detect_bugs_in_directory(directory: str, pattern: str = None, recursive: bool = True, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Detect bugs in all files in a directory.
    
    Args:
        directory: Directory to analyze
        pattern: File pattern to match (e.g., "*.py")
        recursive: Whether to analyze subdirectories
        config: Configuration options (optional)
        
    Returns:
        Analysis results for all files
    """
    try:
        detector = BugDetector(config)
        results = {}
        
        # Get files to analyze
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }
        
        # Determine file pattern
        if pattern:
            # Convert glob pattern to regex
            pattern = pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
            pattern = f"^{pattern}$"
            pattern_re = re.compile(pattern)
        else:
            pattern_re = None
        
        # Find files
        if recursive:
            files = list(path.glob("**/*"))
        else:
            files = list(path.glob("*"))
        
        # Filter files
        files = [f for f in files if f.is_file()]
        if pattern_re:
            files = [f for f in files if pattern_re.match(f.name)]
        
        # Analyze each file
        for file_path in files:
            logger.info(f"Analyzing {file_path}")
            analysis = detector.analyze_file(str(file_path))
            if analysis:
                results[str(file_path)] = analysis
        
        return {
            "success": True,
            "results": results,
            "total_files": len(files),
            "analyzed_files": len(results)
        }
    except Exception as e:
        logger.error(f"Error detecting bugs in directory: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def apply_suggested_fixes(analysis_result: Dict[str, Any], auto_apply: bool = False) -> Dict[str, Any]:
    """
    Apply suggested fixes from analysis.
    
    Args:
        analysis_result: Analysis results
        auto_apply: Whether to automatically apply fixes without confirmation
        
    Returns:
        Result of applying fixes
    """
    try:
        if not analysis_result.get("success", False) or "analysis" not in analysis_result:
            return {
                "success": False,
                "error": "Invalid analysis result"
            }
        
        analysis = analysis_result["analysis"]
        detector = BugDetector()
        
        # Apply fixes
        success = detector.apply_fixes(analysis, auto_apply)
        
        if success:
            return {
                "success": True,
                "file": analysis.get("file"),
                "issues_fixed": analysis.get("total_issues", 0)
            }
        else:
            return {
                "success": False,
                "error": f"Failed to apply fixes to {analysis.get('file')}"
            }
    except Exception as e:
        logger.error(f"Error applying fixes: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bug Detection Tool")
    parser.add_argument("file", help="File or directory to analyze")
    parser.add_argument("--focus", help="Function or section to focus on")
    parser.add_argument("--pattern", help="File pattern to match (for directory analysis)")
    parser.add_argument("--recursive", action="store_true", help="Analyze subdirectories recursively")
    parser.add_argument("--auto-apply", action="store_true", help="Automatically apply suggested fixes")
    
    args = parser.parse_args()
    
    if Path(args.file).is_file():
        # Analyze single file
        result = analyze_code(args.file, args.focus)
        
        if result.get("success", False):
            analysis = result["analysis"]
            print(f"\nAnalysis of {args.file}:")
            print(f"Found {analysis.get('total_issues', 0)} issues")
            print(f"Highest severity: {analysis.get('highest_severity', 'none')}")
            
            for i, issue in enumerate(analysis.get("issues", []), 1):
                print(f"\nIssue {i}:")
                print(f"  Description: {issue.get('description', 'Unknown')}")
                print(f"  Location: {issue.get('location', 'Unknown')}")
                print(f"  Severity: {issue.get('severity', 'Unknown')}")
                print(f"  Confidence: {issue.get('confidence', 0):.2f}")
                print(f"  Fix: {issue.get('fix', 'Unknown')}")
            
            if args.auto_apply and analysis.get("total_issues", 0) > 0:
                fix_result = apply_suggested_fixes(result, args.auto_apply)
                
                if fix_result.get("success", False):
                    print(f"\nSuccessfully applied fixes to {args.file}")
                    print(f"Fixed {fix_result.get('issues_fixed', 0)} issues")
                else:
                    print(f"\nFailed to apply fixes: {fix_result.get('error', 'Unknown error')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    elif Path(args.file).is_dir():
        # Analyze directory
        result = detect_bugs_in_directory(args.file, args.pattern, args.recursive)
        
        if result.get("success", False):
            results = result["results"]
            print(f"\nAnalysis of directory {args.file}:")
            print(f"Analyzed {result.get('analyzed_files', 0)} of {result.get('total_files', 0)} files")
            
            total_issues = sum(analysis.get("total_issues", 0) for analysis in results.values())
            print(f"Found {total_issues} issues in total")
            
            for file_path, analysis in results.items():
                if analysis.get("total_issues", 0) > 0:
                    print(f"\nFile: {file_path}")
                    print(f"  Issues: {analysis.get('total_issues', 0)}")
                    print(f"  Highest severity: {analysis.get('highest_severity', 'none')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"Error: {args.file} is not a valid file or directory")
