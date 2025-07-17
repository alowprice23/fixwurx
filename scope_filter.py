#!/usr/bin/env python3
"""
Scope Filter Module

This module provides file and directory filtering capabilities for code analysis,
allowing users to focus on specific parts of the codebase and exclude others.
"""

import os
import re
import fnmatch
import logging
from pathlib import Path
from typing import List, Set, Dict, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scope_filter.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ScopeFilter")

class ScopeFilter:
    """
    A class that filters files and directories based on various criteria
    such as extensions, paths, patterns, and content.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the scope filter.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Extension filters
        self.include_extensions = self.config.get("include_extensions", [])
        self.exclude_extensions = self.config.get("exclude_extensions", [])
        
        # Directory filters
        self.include_dirs = self.config.get("include_dirs", [])
        self.exclude_dirs = self.config.get("exclude_dirs", [])
        
        # Pattern filters
        self.include_patterns = self.config.get("include_patterns", [])
        self.exclude_patterns = self.config.get("exclude_patterns", [])
        
        # Content filters
        self.content_patterns = self.config.get("content_patterns", [])
        
        # Normalize paths
        self._normalize_paths()
        
        # Compile patterns
        self._compile_patterns()
        
        logger.info("Scope filter initialized")
    
    def _normalize_paths(self) -> None:
        """
        Normalize directory paths.
        """
        # Convert to Path objects and resolve
        self.include_dirs = [os.path.normpath(Path(d).resolve()) for d in self.include_dirs]
        self.exclude_dirs = [os.path.normpath(Path(d).resolve()) for d in self.exclude_dirs]
    
    def _compile_patterns(self) -> None:
        """
        Compile file name patterns.
        """
        # Convert glob patterns to regex
        self.include_pattern_regex = []
        for pattern in self.include_patterns:
            regex_pattern = fnmatch.translate(pattern)
            self.include_pattern_regex.append(re.compile(regex_pattern))
        
        self.exclude_pattern_regex = []
        for pattern in self.exclude_patterns:
            regex_pattern = fnmatch.translate(pattern)
            self.exclude_pattern_regex.append(re.compile(regex_pattern))
        
        # Compile content patterns
        self.content_pattern_regex = []
        for pattern in self.content_patterns:
            self.content_pattern_regex.append(re.compile(pattern))
    
    def should_include_file(self, file_path: str, check_content: bool = False) -> bool:
        """
        Check if a file should be included in the analysis scope.
        
        Args:
            file_path: Path to the file
            check_content: Whether to check file content
            
        Returns:
            Whether the file should be included
        """
        path_obj = Path(file_path)
        file_name = path_obj.name
        
        # Check if file exists
        if not path_obj.exists() or not path_obj.is_file():
            logger.debug(f"File does not exist: {file_path}")
            return False
        
        # Check extension filters
        ext = path_obj.suffix.lower()
        if self.include_extensions and ext not in self.include_extensions:
            logger.debug(f"File extension not included: {ext}")
            return False
        
        if ext in self.exclude_extensions:
            logger.debug(f"File extension excluded: {ext}")
            return False
        
        # Check directory filters
        parent_path = os.path.normpath(path_obj.parent.resolve())
        
        if self.include_dirs and not any(parent_path.startswith(d) for d in self.include_dirs):
            logger.debug(f"File directory not included: {parent_path}")
            return False
        
        if any(parent_path.startswith(d) for d in self.exclude_dirs):
            logger.debug(f"File directory excluded: {parent_path}")
            return False
        
        # Check file name pattern filters
        if self.include_pattern_regex and not any(pattern.match(file_name) for pattern in self.include_pattern_regex):
            logger.debug(f"File name does not match include patterns: {file_name}")
            return False
        
        if any(pattern.match(file_name) for pattern in self.exclude_pattern_regex):
            logger.debug(f"File name matches exclude patterns: {file_name}")
            return False
        
        # Check content filters
        if check_content and self.content_pattern_regex:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                if not any(pattern.search(content) for pattern in self.content_pattern_regex):
                    logger.debug(f"File content does not match content patterns: {file_path}")
                    return False
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return False
        
        return True
    
    def filter_files(self, files: List[str], check_content: bool = False) -> List[str]:
        """
        Filter a list of files based on the scope filters.
        
        Args:
            files: List of file paths
            check_content: Whether to check file content
            
        Returns:
            Filtered list of file paths
        """
        return [file for file in files if self.should_include_file(file, check_content)]
    
    def find_files(self, directory: str, recursive: bool = True, check_content: bool = False) -> List[str]:
        """
        Find files in a directory that match the scope filters.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            check_content: Whether to check file content
            
        Returns:
            List of matching file paths
        """
        directory_path = Path(directory)
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        matching_files = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.should_include_file(file_path, check_content):
                        matching_files.append(file_path)
        else:
            for item in directory_path.iterdir():
                if item.is_file() and self.should_include_file(str(item), check_content):
                    matching_files.append(str(item))
        
        return matching_files
    
    def calculate_file_entropy(self, file_path: str) -> float:
        """
        Calculate entropy of a file to determine complexity.
        Higher entropy indicates more complex or diverse content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Entropy value (0.0 to 1.0)
        """
        import math
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Skip empty files
            if not content:
                return 0.0
            
            # Calculate character frequency
            char_freq = {}
            for char in content:
                if char in char_freq:
                    char_freq[char] += 1
                else:
                    char_freq[char] = 1
            
            # Calculate entropy
            content_length = len(content)
            entropy = 0.0
            
            for freq in char_freq.values():
                probability = freq / content_length
                entropy -= probability * math.log2(probability)
            
            # Normalize entropy to 0.0-1.0
            # Maximum entropy for ASCII is log2(256) = 8
            normalized_entropy = min(1.0, entropy / 8.0)
            
            return normalized_entropy
        except Exception as e:
            logger.error(f"Error calculating file entropy for {file_path}: {e}")
            return 0.0
    
    def filter_by_entropy(self, files: List[str], min_entropy: float = 0.0, max_entropy: float = 1.0) -> List[str]:
        """
        Filter files based on their entropy values.
        
        Args:
            files: List of file paths
            min_entropy: Minimum entropy threshold
            max_entropy: Maximum entropy threshold
            
        Returns:
            Filtered list of file paths
        """
        result = []
        
        for file_path in files:
            entropy = self.calculate_file_entropy(file_path)
            if min_entropy <= entropy <= max_entropy:
                result.append(file_path)
        
        return result
    
    def detect_bug_patterns(self, file_path: str) -> Dict[str, Any]:
        """
        Detect common bug patterns in a file without full analysis.
        This is a lightweight check to determine if a file is likely to contain bugs.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with pattern detection results
        """
        result = {
            "file": file_path,
            "patterns_detected": 0,
            "pattern_types": [],
            "severity_estimate": "low"
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Common bug pattern signatures
            patterns = {
                "division_by_zero": r'\/\s*0',
                "null_reference": r'null\s*\.\w+|\.\w+\s*=\s*null',
                "array_index": r'\[\s*-1\s*\]|\[\s*len\s*\(|\.length\s*\]',
                "exception_swallow": r'catch\s*\([^\)]+\)\s*{',
                "hardcoded_credentials": r'password\s*=\s*["\'][^"\']+["\']',
                "sql_injection": r'SELECT\s+.*\+\s*\w+',
                "insecure_random": r'random\s*\(',
                "resource_leak": r'open\s*\(|new\s+FileInputStream',
                "infinite_loop": r'while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\)',
                "deprecated_api": r'@Deprecated|@deprecated'
            }
            
            # Check each pattern
            for pattern_name, pattern_regex in patterns.items():
                regex = re.compile(pattern_regex, re.IGNORECASE)
                if regex.search(content):
                    result["patterns_detected"] += 1
                    result["pattern_types"].append(pattern_name)
            
            # Check for TODO/FIXME comments
            todo_count = 0
            for line in lines:
                if re.search(r'TODO|FIXME|XXX|HACK', line, re.IGNORECASE):
                    todo_count += 1
            
            result["todo_comments"] = todo_count
            
            # Estimate severity
            if result["patterns_detected"] > 3 or todo_count > 5:
                result["severity_estimate"] = "high"
            elif result["patterns_detected"] > 1 or todo_count > 2:
                result["severity_estimate"] = "medium"
            
            return result
        except Exception as e:
            logger.error(f"Error detecting bug patterns in {file_path}: {e}")
            return result
    
    def content_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Perform content analysis on a file to determine relevant metrics.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with content analysis results
        """
        result = {
            "file": file_path,
            "lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "complexity_estimate": "low",
            "entropy": 0.0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Count lines
            result["lines"] = len(lines)
            
            # Analyze line types
            in_multiline_comment = False
            for line in lines:
                line = line.strip()
                
                if not line:
                    result["blank_lines"] += 1
                    continue
                
                # Check for multiline comment start/end
                if '/*' in line and '*/' not in line:
                    in_multiline_comment = True
                    result["comment_lines"] += 1
                    continue
                
                if '*/' in line and '/*' not in line:
                    in_multiline_comment = False
                    result["comment_lines"] += 1
                    continue
                
                if in_multiline_comment:
                    result["comment_lines"] += 1
                    continue
                
                # Check for single line comments
                if line.startswith('//') or line.startswith('#') or line.startswith('--'):
                    result["comment_lines"] += 1
                    continue
                
                # Count as code line
                result["code_lines"] += 1
            
            # Calculate complexity estimate
            result["entropy"] = self.calculate_file_entropy(file_path)
            
            # Nesting level
            max_nesting = 0
            current_nesting = 0
            for line in lines:
                # Count opening and closing braces
                opening = line.count('{')
                closing = line.count('}')
                current_nesting += opening - closing
                max_nesting = max(max_nesting, current_nesting)
            
            # Estimate complexity
            if result["code_lines"] > 500 or max_nesting > 5 or result["entropy"] > 0.7:
                result["complexity_estimate"] = "high"
            elif result["code_lines"] > 200 or max_nesting > 3 or result["entropy"] > 0.5:
                result["complexity_estimate"] = "medium"
            
            return result
        except Exception as e:
            logger.error(f"Error performing content analysis on {file_path}: {e}")
            return result

# API Functions for command handlers

def filter_directory(directory: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Filter files in a directory based on scope filters.
    
    Args:
        directory: Directory to filter
        config: Configuration options
        
    Returns:
        Dictionary with filtering results
    """
    try:
        filter = ScopeFilter(config)
        matching_files = filter.find_files(directory, recursive=True)
        
        return {
            "success": True,
            "directory": directory,
            "matching_files": matching_files,
            "file_count": len(matching_files)
        }
    except Exception as e:
        logger.error(f"Error filtering directory: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def detect_patterns(directory: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Detect bug patterns in files within a directory.
    
    Args:
        directory: Directory to analyze
        config: Configuration options
        
    Returns:
        Dictionary with pattern detection results
    """
    try:
        filter = ScopeFilter(config)
        matching_files = filter.find_files(directory, recursive=True)
        
        results = {}
        high_severity_files = []
        
        for file_path in matching_files:
            pattern_result = filter.detect_bug_patterns(file_path)
            results[file_path] = pattern_result
            
            if pattern_result["severity_estimate"] == "high":
                high_severity_files.append(file_path)
        
        return {
            "success": True,
            "directory": directory,
            "file_count": len(matching_files),
            "high_severity_count": len(high_severity_files),
            "high_severity_files": high_severity_files,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def analyze_content(directory: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze content of files within a directory.
    
    Args:
        directory: Directory to analyze
        config: Configuration options
        
    Returns:
        Dictionary with content analysis results
    """
    try:
        filter = ScopeFilter(config)
        matching_files = filter.find_files(directory, recursive=True)
        
        results = {}
        high_complexity_files = []
        
        for file_path in matching_files:
            content_result = filter.content_analysis(file_path)
            results[file_path] = content_result
            
            if content_result["complexity_estimate"] == "high":
                high_complexity_files.append(file_path)
        
        return {
            "success": True,
            "directory": directory,
            "file_count": len(matching_files),
            "high_complexity_count": len(high_complexity_files),
            "high_complexity_files": high_complexity_files,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error analyzing content: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def entropy_analysis(directory: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Perform entropy analysis on files within a directory.
    
    Args:
        directory: Directory to analyze
        config: Configuration options
        
    Returns:
        Dictionary with entropy analysis results
    """
    try:
        filter = ScopeFilter(config)
        matching_files = filter.find_files(directory, recursive=True)
        
        results = {}
        entropy_values = []
        
        for file_path in matching_files:
            entropy = filter.calculate_file_entropy(file_path)
            results[file_path] = entropy
            entropy_values.append(entropy)
        
        # Calculate statistics
        avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0
        max_entropy = max(entropy_values) if entropy_values else 0
        min_entropy = min(entropy_values) if entropy_values else 0
        
        # Find high entropy files
        high_entropy_files = []
        for file_path, entropy in results.items():
            if entropy > 0.7:  # Threshold for high entropy
                high_entropy_files.append(file_path)
        
        return {
            "success": True,
            "directory": directory,
            "file_count": len(matching_files),
            "average_entropy": avg_entropy,
            "max_entropy": max_entropy,
            "min_entropy": min_entropy,
            "high_entropy_count": len(high_entropy_files),
            "high_entropy_files": high_entropy_files,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error performing entropy analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scope Filter Tool")
    parser.add_argument("directory", help="Directory to filter")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to include")
    parser.add_argument("--exclude-extensions", "-x", nargs="+", help="File extensions to exclude")
    parser.add_argument("--patterns", "-p", nargs="+", help="File name patterns to include")
    parser.add_argument("--exclude-patterns", "-xp", nargs="+", help="File name patterns to exclude")
    parser.add_argument("--content", "-c", nargs="+", help="Content patterns to match")
    parser.add_argument("--analyze", "-a", action="store_true", help="Perform content analysis")
    parser.add_argument("--entropy", "-en", action="store_true", help="Perform entropy analysis")
    parser.add_argument("--detect", "-d", action="store_true", help="Detect bug patterns")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {}
    
    if args.extensions:
        config["include_extensions"] = [f".{ext.lstrip('.')}" for ext in args.extensions]
    
    if args.exclude_extensions:
        config["exclude_extensions"] = [f".{ext.lstrip('.')}" for ext in args.exclude_extensions]
    
    if args.patterns:
        config["include_patterns"] = args.patterns
    
    if args.exclude_patterns:
        config["exclude_patterns"] = args.exclude_patterns
    
    if args.content:
        config["content_patterns"] = args.content
    
    # Create filter
    scope_filter = ScopeFilter(config)
    
    # Find files
    matching_files = scope_filter.find_files(args.directory, recursive=True)
    
    print(f"\nFound {len(matching_files)} matching files in {args.directory}:")
    for file in matching_files[:10]:  # Show first 10 files
        print(f"  {file}")
    
    if len(matching_files) > 10:
        print(f"  ... and {len(matching_files) - 10} more")
    
    # Perform additional analysis if requested
    if args.analyze:
        print("\nPerforming content analysis...")
        for file in matching_files[:5]:  # Analyze first 5 files
            result = scope_filter.content_analysis(file)
            print(f"\nFile: {file}")
            print(f"  Lines: {result['lines']}")
            print(f"  Code lines: {result['code_lines']}")
            print(f"  Comment lines: {result['comment_lines']}")
            print(f"  Blank lines: {result['blank_lines']}")
            print(f"  Complexity: {result['complexity_estimate']}")
            print(f"  Entropy: {result['entropy']:.2f}")
    
    if args.entropy:
        print("\nPerforming entropy analysis...")
        entropy_values = []
        
        for file in matching_files:
            entropy = scope_filter.calculate_file_entropy(file)
            entropy_values.append((file, entropy))
        
        # Sort by entropy (highest first)
        entropy_values.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 files by entropy:")
        for file, entropy in entropy_values[:5]:
            print(f"  {file}: {entropy:.2f}")
    
    if args.detect:
        print("\nDetecting bug patterns...")
        for file in matching_files[:5]:  # Detect patterns in first 5 files
            result = scope_filter.detect_bug_patterns(file)
            print(f"\nFile: {file}")
            print(f"  Patterns detected: {result['patterns_detected']}")
            print(f"  Pattern types: {', '.join(result['pattern_types'])}")
            print(f"  TODO comments: {result['todo_comments']}")
            print(f"  Severity estimate: {result['severity_estimate']}")
