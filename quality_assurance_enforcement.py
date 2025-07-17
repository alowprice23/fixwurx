#!/usr/bin/env python3
"""
Quality Assurance Enforcement Module

This module provides quality assurance enforcement capabilities for the auditor agent,
ensuring that code, tests, documentation, and other artifacts meet defined standards.
"""

import os
import sys
import json
import logging
import time
import threading
import subprocess
import re
import ast
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, Pattern

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("quality_assurance.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("QualityAssurance")

class QualityStandard:
    """
    Base class for quality standards.
    """
    
    def __init__(self, name: str, description: str = None, severity: str = "medium",
                enabled: bool = True, tags: List[str] = None):
        """
        Initialize quality standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity (low, medium, high, critical)
            enabled: Whether the standard is enabled
            tags: Standard tags
        """
        self.name = name
        self.description = description or ""
        self.severity = severity
        self.enabled = enabled
        self.tags = tags or []
    
    def check(self, target: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if the target meets the standard.
        
        Args:
            target: Target to check
            context: Additional context information
            
        Returns:
            Check result
        """
        raise NotImplementedError("Subclass must implement check")
    
    def fix(self, target: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fix the target to meet the standard.
        
        Args:
            target: Target to fix
            context: Additional context information
            
        Returns:
            Fix result
        """
        raise NotImplementedError("Subclass must implement fix")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity,
            "enabled": self.enabled,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityStandard':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Quality standard
        """
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            severity=data.get("severity", "medium"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", [])
        )

class CodeQualityStandard(QualityStandard):
    """
    Base class for code quality standards.
    """
    
    def __init__(self, name: str, description: str = None, severity: str = "medium",
                enabled: bool = True, tags: List[str] = None, 
                file_patterns: List[str] = None):
        """
        Initialize code quality standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity (low, medium, high, critical)
            enabled: Whether the standard is enabled
            tags: Standard tags
            file_patterns: File patterns to check (e.g., "*.py")
        """
        super().__init__(name, description, severity, enabled, tags)
        self.file_patterns = file_patterns or ["*.py"]
    
    def applies_to_file(self, file_path: str) -> bool:
        """
        Check if the standard applies to a file.
        
        Args:
            file_path: File path
            
        Returns:
            Whether the standard applies to the file
        """
        import fnmatch
        
        for pattern in self.file_patterns:
            if fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        data["file_patterns"] = self.file_patterns
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeQualityStandard':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Code quality standard
        """
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            severity=data.get("severity", "medium"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
            file_patterns=data.get("file_patterns", ["*.py"])
        )

class PythonDocstringStandard(CodeQualityStandard):
    """
    Standard for Python docstrings.
    """
    
    def __init__(self, name: str = "Python Docstring Standard", 
                description: str = "Checks if Python modules, classes, methods, and functions have docstrings",
                severity: str = "medium", enabled: bool = True, 
                tags: List[str] = None, file_patterns: List[str] = None,
                require_module_docstring: bool = True,
                require_class_docstring: bool = True,
                require_method_docstring: bool = True,
                require_function_docstring: bool = True,
                min_docstring_length: int = 10):
        """
        Initialize Python docstring standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity
            enabled: Whether the standard is enabled
            tags: Standard tags
            file_patterns: File patterns to check
            require_module_docstring: Whether to require module docstrings
            require_class_docstring: Whether to require class docstrings
            require_method_docstring: Whether to require method docstrings
            require_function_docstring: Whether to require function docstrings
            min_docstring_length: Minimum docstring length
        """
        super().__init__(
            name=name,
            description=description,
            severity=severity,
            enabled=enabled,
            tags=tags or ["python", "docstring", "documentation"],
            file_patterns=file_patterns or ["*.py"]
        )
        self.require_module_docstring = require_module_docstring
        self.require_class_docstring = require_class_docstring
        self.require_method_docstring = require_method_docstring
        self.require_function_docstring = require_function_docstring
        self.min_docstring_length = min_docstring_length
    
    def check(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if a Python file has proper docstrings.
        
        Args:
            target: Python file path
            context: Additional context information
            
        Returns:
            Check result
        """
        if not self.applies_to_file(target):
            return {
                "success": True,
                "message": f"Standard does not apply to {target}",
                "violations": []
            }
        
        try:
            with open(target, "r") as f:
                code = f.read()
            
            tree = ast.parse(code)
            violations = []
            
            # Check module docstring
            if self.require_module_docstring and not ast.get_docstring(tree):
                violations.append({
                    "type": "module",
                    "name": os.path.basename(target),
                    "message": "Module is missing a docstring",
                    "line": 1
                })
            
            # Check class, method, and function docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and self.require_class_docstring:
                    docstring = ast.get_docstring(node)
                    if not docstring:
                        violations.append({
                            "type": "class",
                            "name": node.name,
                            "message": f"Class '{node.name}' is missing a docstring",
                            "line": node.lineno
                        })
                    elif len(docstring.strip()) < self.min_docstring_length:
                        violations.append({
                            "type": "class",
                            "name": node.name,
                            "message": f"Class '{node.name}' has a docstring shorter than {self.min_docstring_length} characters",
                            "line": node.lineno
                        })
                
                elif isinstance(node, ast.FunctionDef):
                    is_method = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef) and node in parent.body:
                            is_method = True
                            break
                    
                    if (is_method and self.require_method_docstring) or (not is_method and self.require_function_docstring):
                        docstring = ast.get_docstring(node)
                        if not docstring:
                            violations.append({
                                "type": "method" if is_method else "function",
                                "name": node.name,
                                "message": f"{'Method' if is_method else 'Function'} '{node.name}' is missing a docstring",
                                "line": node.lineno
                            })
                        elif len(docstring.strip()) < self.min_docstring_length:
                            violations.append({
                                "type": "method" if is_method else "function",
                                "name": node.name,
                                "message": f"{'Method' if is_method else 'Function'} '{node.name}' has a docstring shorter than {self.min_docstring_length} characters",
                                "line": node.lineno
                            })
            
            return {
                "success": len(violations) == 0,
                "message": f"Found {len(violations)} docstring violations" if violations else "No docstring violations found",
                "violations": violations
            }
        except Exception as e:
            logger.error(f"Error checking docstrings in {target}: {e}")
            return {
                "success": False,
                "message": f"Error checking docstrings: {str(e)}",
                "violations": []
            }
    
    def fix(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fix missing docstrings in a Python file.
        
        Args:
            target: Python file path
            context: Additional context information
            
        Returns:
            Fix result
        """
        if not self.applies_to_file(target):
            return {
                "success": True,
                "message": f"Standard does not apply to {target}",
                "fixes": []
            }
        
        # Check first to get violations
        check_result = self.check(target, context)
        
        if check_result["success"]:
            return {
                "success": True,
                "message": "No fixes needed",
                "fixes": []
            }
        
        try:
            with open(target, "r") as f:
                lines = f.readlines()
            
            fixes = []
            
            # For now, we'll just create a log of fixes that would be needed
            # A real implementation would modify the file
            for violation in check_result["violations"]:
                fixes.append({
                    "type": violation["type"],
                    "name": violation["name"],
                    "line": violation["line"],
                    "message": f"Would add docstring for {violation['type']} '{violation['name']}'"
                })
            
            return {
                "success": True,
                "message": f"Identified {len(fixes)} docstring fixes",
                "fixes": fixes
            }
        except Exception as e:
            logger.error(f"Error fixing docstrings in {target}: {e}")
            return {
                "success": False,
                "message": f"Error fixing docstrings: {str(e)}",
                "fixes": []
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        data.update({
            "require_module_docstring": self.require_module_docstring,
            "require_class_docstring": self.require_class_docstring,
            "require_method_docstring": self.require_method_docstring,
            "require_function_docstring": self.require_function_docstring,
            "min_docstring_length": self.min_docstring_length
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PythonDocstringStandard':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Python docstring standard
        """
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            severity=data.get("severity", "medium"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", ["python", "docstring", "documentation"]),
            file_patterns=data.get("file_patterns", ["*.py"]),
            require_module_docstring=data.get("require_module_docstring", True),
            require_class_docstring=data.get("require_class_docstring", True),
            require_method_docstring=data.get("require_method_docstring", True),
            require_function_docstring=data.get("require_function_docstring", True),
            min_docstring_length=data.get("min_docstring_length", 10)
        )

class PythonLintingStandard(CodeQualityStandard):
    """
    Standard for Python linting using flake8.
    """
    
    def __init__(self, name: str = "Python Linting Standard", 
                description: str = "Checks if Python code meets PEP8 and other style guidelines",
                severity: str = "medium", enabled: bool = True, 
                tags: List[str] = None, file_patterns: List[str] = None,
                max_line_length: int = 100, ignore_rules: List[str] = None):
        """
        Initialize Python linting standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity
            enabled: Whether the standard is enabled
            tags: Standard tags
            file_patterns: File patterns to check
            max_line_length: Maximum line length
            ignore_rules: Flake8 rules to ignore
        """
        super().__init__(
            name=name,
            description=description,
            severity=severity,
            enabled=enabled,
            tags=tags or ["python", "linting", "style"],
            file_patterns=file_patterns or ["*.py"]
        )
        self.max_line_length = max_line_length
        self.ignore_rules = ignore_rules or []
    
    def check(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if a Python file meets linting standards.
        
        Args:
            target: Python file path
            context: Additional context information
            
        Returns:
            Check result
        """
        if not self.applies_to_file(target):
            return {
                "success": True,
                "message": f"Standard does not apply to {target}",
                "violations": []
            }
        
        try:
            # Try to import flake8
            try:
                import flake8
                has_flake8 = True
            except ImportError:
                has_flake8 = False
            
            if not has_flake8:
                # Fallback to using subprocess
                cmd = ["flake8", target, f"--max-line-length={self.max_line_length}"]
                
                if self.ignore_rules:
                    cmd.append(f"--ignore={','.join(self.ignore_rules)}")
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "message": "No linting violations found",
                            "violations": []
                        }
                    
                    violations = []
                    
                    for line in result.stdout.strip().split("\n"):
                        if not line:
                            continue
                        
                        # Parse flake8 output (file:line:col: code message)
                        match = re.match(r"([^:]+):(\d+):(\d+): ([A-Z]\d+) (.*)", line)
                        
                        if match:
                            file_path, line_num, col, code, message = match.groups()
                            violations.append({
                                "file": file_path,
                                "line": int(line_num),
                                "column": int(col),
                                "code": code,
                                "message": message
                            })
                    
                    return {
                        "success": False,
                        "message": f"Found {len(violations)} linting violations",
                        "violations": violations
                    }
                except FileNotFoundError:
                    return {
                        "success": False,
                        "message": "Flake8 not found. Please install flake8 to use this standard.",
                        "violations": []
                    }
            else:
                # Direct flake8 API usage would go here
                # For now, we'll use the subprocess approach
                return {
                    "success": False,
                    "message": "Direct flake8 API usage not implemented yet",
                    "violations": []
                }
        except Exception as e:
            logger.error(f"Error checking linting in {target}: {e}")
            return {
                "success": False,
                "message": f"Error checking linting: {str(e)}",
                "violations": []
            }
    
    def fix(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fix linting issues in a Python file.
        
        Args:
            target: Python file path
            context: Additional context information
            
        Returns:
            Fix result
        """
        if not self.applies_to_file(target):
            return {
                "success": True,
                "message": f"Standard does not apply to {target}",
                "fixes": []
            }
        
        try:
            # Try to use autopep8 to fix issues
            try:
                import autopep8
                has_autopep8 = True
            except ImportError:
                has_autopep8 = False
            
            if not has_autopep8:
                # Fallback to using subprocess
                cmd = ["autopep8", "--in-place", target, f"--max-line-length={self.max_line_length}"]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "message": "Fixed linting issues",
                            "fixes": [{
                                "file": target,
                                "message": "Applied autopep8 fixes"
                            }]
                        }
                    else:
                        return {
                            "success": False,
                            "message": f"Error fixing linting issues: {result.stderr}",
                            "fixes": []
                        }
                except FileNotFoundError:
                    return {
                        "success": False,
                        "message": "Autopep8 not found. Please install autopep8 to use this standard.",
                        "fixes": []
                    }
            else:
                # Direct autopep8 API usage
                with open(target, "r") as f:
                    content = f.read()
                
                fixed_content = autopep8.fix_code(
                    content, 
                    options={"max_line_length": self.max_line_length}
                )
                
                if fixed_content != content:
                    with open(target, "w") as f:
                        f.write(fixed_content)
                    
                    return {
                        "success": True,
                        "message": "Fixed linting issues",
                        "fixes": [{
                            "file": target,
                            "message": "Applied autopep8 fixes"
                        }]
                    }
                else:
                    return {
                        "success": True,
                        "message": "No fixes needed",
                        "fixes": []
                    }
        except Exception as e:
            logger.error(f"Error fixing linting in {target}: {e}")
            return {
                "success": False,
                "message": f"Error fixing linting: {str(e)}",
                "fixes": []
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        data.update({
            "max_line_length": self.max_line_length,
            "ignore_rules": self.ignore_rules
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PythonLintingStandard':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Python linting standard
        """
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            severity=data.get("severity", "medium"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", ["python", "linting", "style"]),
            file_patterns=data.get("file_patterns", ["*.py"]),
            max_line_length=data.get("max_line_length", 100),
            ignore_rules=data.get("ignore_rules", [])
        )

class TestingStandard(QualityStandard):
    """
    Base class for testing standards.
    """
    
    def __init__(self, name: str, description: str = None, severity: str = "medium",
                enabled: bool = True, tags: List[str] = None):
        """
        Initialize testing standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity (low, medium, high, critical)
            enabled: Whether the standard is enabled
            tags: Standard tags
        """
        super().__init__(name, description, severity, enabled, tags or ["testing"])

class CoverageStandard(TestingStandard):
    """
    Standard for test coverage.
    """
    
    def __init__(self, name: str = "Test Coverage Standard", 
                description: str = "Checks if code meets minimum test coverage requirements",
                severity: str = "high", enabled: bool = True, 
                tags: List[str] = None, min_coverage: float = 80.0,
                coverage_tool: str = "coverage"):
        """
        Initialize coverage standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity
            enabled: Whether the standard is enabled
            tags: Standard tags
            min_coverage: Minimum required coverage percentage
            coverage_tool: Coverage tool to use
        """
        super().__init__(
            name=name,
            description=description,
            severity=severity,
            enabled=enabled,
            tags=tags or ["testing", "coverage"]
        )
        self.min_coverage = min_coverage
        self.coverage_tool = coverage_tool
    
    def check(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if code meets coverage requirements.
        
        Args:
            target: Target to check (module or directory)
            context: Additional context information
            
        Returns:
            Check result
        """
        try:
            if self.coverage_tool == "coverage":
                # Try to use Python coverage module
                try:
                    import coverage
                    has_coverage = True
                except ImportError:
                    has_coverage = False
                
                if not has_coverage:
                    # Fallback to using subprocess
                    cmd = ["coverage", "report", "--include=" + target]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            return {
                                "success": False,
                                "message": f"Error running coverage: {result.stderr}",
                                "coverage": 0.0
                            }
                        
                        # Parse coverage output
                        last_line = result.stdout.strip().split("\n")[-1]
                        
                        if "TOTAL" in last_line:
                            # Extract coverage percentage
                            match = re.search(r"(\d+)%", last_line)
                            
                            if match:
                                coverage_pct = float(match.group(1))
                                
                                return {
                                    "success": coverage_pct >= self.min_coverage,
                                    "message": f"Coverage: {coverage_pct}% (minimum: {self.min_coverage}%)",
                                    "coverage": coverage_pct
                                }
                        
                        return {
                            "success": False,
                            "message": "Could not parse coverage output",
                            "coverage": 0.0
                        }
                    except FileNotFoundError:
                        return {
                            "success": False,
                            "message": "Coverage tool not found. Please install coverage.",
                            "coverage": 0.0
                        }
                else:
                    # Direct coverage API usage
                    cov = coverage.Coverage()
                    cov.load()
                    
                    # Get coverage for target
                    total = cov.report(include=target)
                    
                    return {
                        "success": total >= self.min_coverage,
                        "message": f"Coverage: {total:.1f}% (minimum: {self.min_coverage}%)",
                        "coverage": total
                    }
            else:
                return {
                    "success": False,
                    "message": f"Unsupported coverage tool: {self.coverage_tool}",
                    "coverage": 0.0
                }
        except Exception as e:
            logger.error(f"Error checking coverage for {target}: {e}")
            return {
                "success": False,
                "message": f"Error checking coverage: {str(e)}",
                "coverage": 0.0
            }
    
    def fix(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fix coverage issues by generating test stubs.
        
        Args:
            target: Target to fix (module or directory)
            context: Additional context information
            
        Returns:
            Fix result
        """
        # This is a more complex fix that would require generating test stubs
        # For now, we'll just return a message
        return {
            "success": False,
            "message": "Automatic test generation not implemented yet",
            "fixes": []
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        data.update({
            "min_coverage": self.min_coverage,
            "coverage_tool": self.coverage_tool
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoverageStandard':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Coverage standard
        """
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            severity=data.get("severity", "high"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", ["testing", "coverage"]),
            min_coverage=data.get("min_coverage", 80.0),
            coverage_tool=data.get("coverage_tool", "coverage")
        )

class SecurityStandard(QualityStandard):
    """
    Base class for security standards.
    """
    
    def __init__(self, name: str, description: str = None, severity: str = "high",
                enabled: bool = True, tags: List[str] = None):
        """
        Initialize security standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity (low, medium, high, critical)
            enabled: Whether the standard is enabled
            tags: Standard tags
        """
        super().__init__(name, description, severity, enabled, tags or ["security"])

class PythonSecurityStandard(SecurityStandard):
    """
    Standard for Python security checks using bandit.
    """
    
    def __init__(self, name: str = "Python Security Standard", 
                description: str = "Checks Python code for security vulnerabilities",
                severity: str = "high", enabled: bool = True, 
                tags: List[str] = None, file_patterns: List[str] = None,
                confidence_level: str = "medium", severity_level: str = "medium"):
        """
        Initialize Python security standard.
        
        Args:
            name: Standard name
            description: Standard description
            severity: Violation severity
            enabled: Whether the standard is enabled
            tags: Standard tags
            file_patterns: File patterns to check
            confidence_level: Minimum confidence level for findings
            severity_level: Minimum severity level for findings
        """
        super().__init__(
            name=name,
            description=description,
            severity=severity,
            enabled=enabled,
            tags=tags or ["security", "python"]
        )
        self.file_patterns = file_patterns or ["*.py"]
        self.confidence_level = confidence_level
        self.severity_level = severity_level
    
    def applies_to_file(self, file_path: str) -> bool:
        """
        Check if the standard applies to a file.
        
        Args:
            file_path: File path
            
        Returns:
            Whether the standard applies to the file
        """
        import fnmatch
        
        for pattern in self.file_patterns:
            if fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
        
        return False
    
    def check(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if Python code meets security standards.
        
        Args:
            target: Target to check (file or directory)
            context: Additional context information
            
        Returns:
            Check result
        """
        try:
            # Try to use bandit
            cmd = [
                "bandit", "-r" if os.path.isdir(target) else "", target,
                "-f", "json",
