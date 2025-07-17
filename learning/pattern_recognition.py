#!/usr/bin/env python3
"""
Bug Pattern Recognition Module

This module provides bug pattern recognition capabilities, analyzing code and identifying 
recurring patterns in bugs to improve detection and resolution.
"""

import os
import sys
import json
import logging
import time
import re
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pattern_recognition.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PatternRecognition")

class BugPattern:
    """
    Represents a recognized bug pattern.
    """
    
    def __init__(self, pattern_id: str, pattern_type: str, description: str = None,
               signature: str = None, contexts: List[Dict[str, Any]] = None,
               confidence: float = 0.0, detection_count: int = 0):
        """
        Initialize bug pattern.
        
        Args:
            pattern_id: Unique pattern ID
            pattern_type: Type of bug pattern
            description: Description of the pattern
            signature: Pattern signature/fingerprint
            contexts: List of contexts where the pattern was observed
            confidence: Confidence score (0.0 to 1.0)
            detection_count: Number of times this pattern was detected
        """
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.description = description or ""
        self.signature = signature or ""
        self.contexts = contexts or []
        self.confidence = confidence
        self.detection_count = detection_count
        self.creation_time = time.time()
        self.last_updated = time.time()
        self.related_patterns = []
        
        logger.debug(f"Created bug pattern: {pattern_id} ({pattern_type})")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "signature": self.signature,
            "contexts": self.contexts,
            "confidence": self.confidence,
            "detection_count": self.detection_count,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "related_patterns": self.related_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BugPattern':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Bug pattern
        """
        pattern = cls(
            pattern_id=data.get("pattern_id", ""),
            pattern_type=data.get("pattern_type", ""),
            description=data.get("description", ""),
            signature=data.get("signature", ""),
            contexts=data.get("contexts", []),
            confidence=data.get("confidence", 0.0),
            detection_count=data.get("detection_count", 0)
        )
        
        # Restore timestamps and related patterns
        pattern.creation_time = data.get("creation_time", time.time())
        pattern.last_updated = data.get("last_updated", time.time())
        pattern.related_patterns = data.get("related_patterns", [])
        
        return pattern
    
    def update_confidence(self, success_count: int, total_count: int) -> float:
        """
        Update confidence based on successful fixes.
        
        Args:
            success_count: Number of successful fixes
            total_count: Total number of fix attempts
            
        Returns:
            Updated confidence score
        """
        if total_count > 0:
            self.confidence = success_count / total_count
        else:
            self.confidence = 0.0
        
        self.last_updated = time.time()
        return self.confidence
    
    def add_context(self, context: Dict[str, Any]) -> None:
        """
        Add a context where the pattern was observed.
        
        Args:
            context: Context dictionary with information about the bug instance
        """
        self.contexts.append(context)
        self.detection_count += 1
        self.last_updated = time.time()
        
        # Keep only the latest 50 contexts to avoid excessive memory usage
        if len(self.contexts) > 50:
            self.contexts = self.contexts[-50:]
        
        logger.debug(f"Added context to pattern {self.pattern_id}, now has {self.detection_count} detections")
    
    def relate_to_pattern(self, pattern_id: str) -> None:
        """
        Relate this pattern to another pattern.
        
        Args:
            pattern_id: ID of related pattern
        """
        if pattern_id not in self.related_patterns:
            self.related_patterns.append(pattern_id)
            self.last_updated = time.time()
            
            logger.debug(f"Related pattern {self.pattern_id} to {pattern_id}")

class PatternRecognizer:
    """
    Recognizes patterns in bugs and code.
    """
    
    def __init__(self, patterns_file: str = None):
        """
        Initialize pattern recognizer.
        
        Args:
            patterns_file: Path to patterns database file
        """
        self.patterns_file = patterns_file or "bug_patterns.json"
        self.patterns = {}
        self.pattern_types = set()
        self.feature_extractors = {}
        self.similarity_threshold = 0.8
        
        # Load patterns if file exists
        if os.path.exists(self.patterns_file):
            self._load_patterns()
        
        # Register default feature extractors
        self._register_default_extractors()
        
        logger.info("Pattern recognizer initialized")
    
    def _load_patterns(self) -> None:
        """Load patterns from database file."""
        try:
            with open(self.patterns_file, "r") as f:
                data = json.load(f)
            
            # Load patterns
            patterns_data = data.get("patterns", {})
            
            for pattern_id, pattern_data in patterns_data.items():
                self.patterns[pattern_id] = BugPattern.from_dict(pattern_data)
            
            # Extract pattern types
            self.pattern_types = set(pattern.pattern_type for pattern in self.patterns.values())
            
            # Load configuration
            self.similarity_threshold = data.get("similarity_threshold", 0.8)
            
            logger.info(f"Loaded {len(self.patterns)} patterns from {self.patterns_file}")
        except Exception as e:
            logger.error(f"Error loading patterns from {self.patterns_file}: {e}")
            self.patterns = {}
            self.pattern_types = set()
    
    def save_patterns(self) -> None:
        """Save patterns to database file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.patterns_file)), exist_ok=True)
            
            data = {
                "patterns": {pattern_id: pattern.to_dict() for pattern_id, pattern in self.patterns.items()},
                "similarity_threshold": self.similarity_threshold,
                "last_updated": time.time()
            }
            
            with open(self.patterns_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.patterns)} patterns to {self.patterns_file}")
        except Exception as e:
            logger.error(f"Error saving patterns to {self.patterns_file}: {e}")
    
    def _register_default_extractors(self) -> None:
        """Register default feature extractors."""
        # Code structure features
        self.register_feature_extractor("code_structure", self._extract_code_structure_features)
        
        # Error message features
        self.register_feature_extractor("error_message", self._extract_error_message_features)
        
        # Stack trace features
        self.register_feature_extractor("stack_trace", self._extract_stack_trace_features)
        
        # Variable usage features
        self.register_feature_extractor("variable_usage", self._extract_variable_usage_features)
        
        # API usage features
        self.register_feature_extractor("api_usage", self._extract_api_usage_features)
        
        logger.debug("Registered default feature extractors")
    
    def register_feature_extractor(self, name: str, extractor: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Register a feature extractor.
        
        Args:
            name: Extractor name
            extractor: Function that extracts features from a bug context
        """
        self.feature_extractors[name] = extractor
        logger.debug(f"Registered feature extractor: {name}")
    
    def _extract_code_structure_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract code structure features.
        
        Args:
            context: Bug context
            
        Returns:
            Extracted features
        """
        features = {}
        code = context.get("code", "")
        
        if not code:
            return features
        
        # Indentation pattern
        indentation_levels = set()
        for line in code.split("\n"):
            if line.strip():
                indentation = len(line) - len(line.lstrip())
                indentation_levels.add(indentation)
        
        features["indentation_levels"] = sorted(indentation_levels)
        
        # Function/method count
        function_pattern = r"(def|function)\s+\w+\s*\("
        functions = re.findall(function_pattern, code)
        features["function_count"] = len(functions)
        
        # Class count
        class_pattern = r"class\s+\w+"
        classes = re.findall(class_pattern, code)
        features["class_count"] = len(classes)
        
        # Loop count
        loop_pattern = r"(for|while)\s+"
        loops = re.findall(loop_pattern, code)
        features["loop_count"] = len(loops)
        
        # Conditional count
        conditional_pattern = r"if\s+"
        conditionals = re.findall(conditional_pattern, code)
        features["conditional_count"] = len(conditionals)
        
        # Exception handling count
        try_pattern = r"try\s*:"
        try_blocks = re.findall(try_pattern, code)
        features["try_count"] = len(try_blocks)
        
        return features
    
    def _extract_error_message_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract error message features.
        
        Args:
            context: Bug context
            
        Returns:
            Extracted features
        """
        features = {}
        error_message = context.get("error_message", "")
        
        if not error_message:
            return features
        
        # Error type
        error_type_pattern = r"^([A-Za-z0-9_]+(?:Error|Exception|Warning))"
        error_type_match = re.search(error_type_pattern, error_message)
        features["error_type"] = error_type_match.group(1) if error_type_match else "Unknown"
        
        # Error message words
        words = re.findall(r"\b[A-Za-z]+\b", error_message.lower())
        features["error_words"] = words
        
        # Mentioned variables/functions
        identifiers = re.findall(r"'([A-Za-z0-9_]+)'", error_message)
        features["mentioned_identifiers"] = identifiers
        
        # Line numbers
        line_numbers = re.findall(r"line\s+(\d+)", error_message)
        features["line_numbers"] = [int(num) for num in line_numbers]
        
        return features
    
    def _extract_stack_trace_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract stack trace features.
        
        Args:
            context: Bug context
            
        Returns:
            Extracted features
        """
        features = {}
        stack_trace = context.get("stack_trace", "")
        
        if not stack_trace:
            return features
        
        # Stack depth
        frame_pattern = r"File\s+\"([^\"]+)\",\s+line\s+(\d+)"
        frames = re.findall(frame_pattern, stack_trace)
        features["stack_depth"] = len(frames)
        
        # Files involved
        files = [frame[0] for frame in frames]
        features["files_involved"] = list(set(files))
        
        # Functions/methods involved
        function_pattern = r"in\s+([A-Za-z0-9_]+)"
        functions = re.findall(function_pattern, stack_trace)
        features["functions_involved"] = list(set(functions))
        
        return features
    
    def _extract_variable_usage_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract variable usage features.
        
        Args:
            context: Bug context
            
        Returns:
            Extracted features
        """
        features = {}
        code = context.get("code", "")
        
        if not code:
            return features
        
        # Variable assignments
        assignment_pattern = r"([A-Za-z0-9_]+)\s*="
        assignments = re.findall(assignment_pattern, code)
        features["assigned_variables"] = list(set(assignments))
        
        # Variable references
        # This is a simplified approach; a real implementation would use AST parsing
        variable_pattern = r"\b([A-Za-z][A-Za-z0-9_]*)\b"
        all_identifiers = re.findall(variable_pattern, code)
        
        # Filter out keywords
        keywords = {"if", "else", "elif", "for", "while", "try", "except", "finally",
                   "def", "class", "return", "import", "from", "as", "with"}
        variables = [v for v in all_identifiers if v not in keywords]
        features["referenced_variables"] = list(set(variables))
        
        return features
    
    def _extract_api_usage_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract API usage features.
        
        Args:
            context: Bug context
            
        Returns:
            Extracted features
        """
        features = {}
        code = context.get("code", "")
        
        if not code:
            return features
        
        # Import statements
        import_pattern = r"import\s+([A-Za-z0-9_.,\s]+)"
        imports = re.findall(import_pattern, code)
        
        # Clean up imports
        cleaned_imports = []
        for imp in imports:
            parts = [p.strip() for p in imp.split(",")]
            cleaned_imports.extend(parts)
        
        features["imports"] = cleaned_imports
        
        # From imports
        from_import_pattern = r"from\s+([A-Za-z0-9_.]+)\s+import\s+([A-Za-z0-9_.,\s]+)"
        from_imports = re.findall(from_import_pattern, code)
        
        # Clean up from imports
        from_import_modules = [fi[0] for fi in from_imports]
        features["from_import_modules"] = from_import_modules
        
        # Method calls
        method_call_pattern = r"([A-Za-z0-9_]+)\s*\.\s*([A-Za-z0-9_]+)\s*\("
        method_calls = re.findall(method_call_pattern, code)
        features["method_calls"] = [(obj, method) for obj, method in method_calls]
        
        # Function calls
        function_call_pattern = r"([A-Za-z0-9_]+)\s*\("
        function_calls = re.findall(function_call_pattern, code)
        
        # Filter out methods (already captured) and keywords
        function_keywords = {"if", "for", "while", "def", "class"}
        functions = [f for f in function_calls if f not in function_keywords and not any(m[1] == f for m in method_calls)]
        features["function_calls"] = functions
        
        return features
    
    def extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a bug context.
        
        Args:
            context: Bug context
            
        Returns:
            Extracted features
        """
        features = {}
        
        # Run all feature extractors
        for name, extractor in self.feature_extractors.items():
            try:
                features[name] = extractor(context)
            except Exception as e:
                logger.error(f"Error in feature extractor {name}: {e}")
                features[name] = {}
        
        return features
    
    def _generate_signature(self, features: Dict[str, Any]) -> str:
        """
        Generate a signature from features.
        
        Args:
            features: Extracted features
            
        Returns:
            Signature string
        """
        # Serialize features to a stable string representation
        serialized = json.dumps(features, sort_keys=True)
        
        # Create hash
        signature = hashlib.sha256(serialized.encode()).hexdigest()
        
        return signature
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Extract feature types available in both
        common_types = set(features1.keys()) & set(features2.keys())
        
        if not common_types:
            return 0.0
        
        similarities = []
        
        for feature_type in common_types:
            f1 = features1.get(feature_type, {})
            f2 = features2.get(feature_type, {})
            
            if not f1 or not f2:
                continue
            
            # Calculate Jaccard similarity for lists
            if feature_type == "code_structure":
                # Compare function count, class count, etc.
                for key in ["function_count", "class_count", "loop_count", "conditional_count", "try_count"]:
                    if key in f1 and key in f2:
                        # Normalize to 0-1 range
                        max_val = max(f1[key], f2[key])
                        if max_val > 0:
                            sim = 1.0 - abs(f1[key] - f2[key]) / max_val
                            similarities.append(sim)
            
            elif feature_type == "error_message":
                # Compare error type
                if "error_type" in f1 and "error_type" in f2:
                    if f1["error_type"] == f2["error_type"]:
                        similarities.append(1.0)
                    else:
                        similarities.append(0.0)
                
                # Compare error words using Jaccard similarity
                if "error_words" in f1 and "error_words" in f2:
                    set1 = set(f1["error_words"])
                    set2 = set(f2["error_words"])
                    
                    if set1 or set2:
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        similarities.append(jaccard)
            
            elif feature_type == "stack_trace":
                # Compare stack depth
                if "stack_depth" in f1 and "stack_depth" in f2:
                    max_depth = max(f1["stack_depth"], f2["stack_depth"])
                    if max_depth > 0:
                        sim = 1.0 - abs(f1["stack_depth"] - f2["stack_depth"]) / max_depth
                        similarities.append(sim)
                
                # Compare files involved
                if "files_involved" in f1 and "files_involved" in f2:
                    set1 = set(f1["files_involved"])
                    set2 = set(f2["files_involved"])
                    
                    if set1 or set2:
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        similarities.append(jaccard)
            
            elif feature_type == "variable_usage":
                # Compare variable usage
                for key in ["assigned_variables", "referenced_variables"]:
                    if key in f1 and key in f2:
                        set1 = set(f1[key])
                        set2 = set(f2[key])
                        
                        if set1 or set2:
                            jaccard = len(set1 & set2) / len(set1 | set2)
                            similarities.append(jaccard)
            
            elif feature_type == "api_usage":
                # Compare API usage
                for key in ["imports", "from_import_modules", "function_calls"]:
                    if key in f1 and key in f2:
                        set1 = set(f1[key])
                        set2 = set(f2[key])
                        
                        if set1 or set2:
                            jaccard = len(set1 & set2) / len(set1 | set2)
                            similarities.append(jaccard)
                
                # Compare method calls
                if "method_calls" in f1 and "method_calls" in f2:
                    methods1 = set((obj, method) for obj, method in f1["method_calls"])
                    methods2 = set((obj, method) for obj, method in f2["method_calls"])
                    
                    if methods1 or methods2:
                        jaccard = len(methods1 & methods2) / len(methods1 | methods2)
                        similarities.append(jaccard)
        
        # Calculate overall similarity
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0
    
    def recognize_pattern(self, context: Dict[str, Any], pattern_type: str = None) -> Optional[BugPattern]:
        """
        Recognize a pattern in a bug context.
        
        Args:
            context: Bug context
            pattern_type: Type of pattern to look for, or None for any type
            
        Returns:
            Recognized pattern, or None if no match
        """
        # Extract features
        features = self.extract_features(context)
        
        # Generate signature
        signature = self._generate_signature(features)
        
        # Filter patterns by type if specified
        candidates = self.patterns.values()
        if pattern_type:
            candidates = [p for p in candidates if p.pattern_type == pattern_type]
        
        # Check for exact signature match
        for pattern in candidates:
            if pattern.signature == signature:
                # Exact match found
                pattern.add_context(context)
                self.save_patterns()
                logger.info(f"Exact pattern match found: {pattern.pattern_id}")
                return pattern
        
        # Check for similar patterns
        best_match = None
        best_similarity = 0.0
        
        for pattern in candidates:
            for ctx in pattern.contexts:
                if "features" in ctx:
                    similarity = self._calculate_similarity(features, ctx["features"])
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = pattern
        
        if best_match:
            # Similar pattern found, add context
            context["features"] = features
            best_match.add_context(context)
            self.save_patterns()
            logger.info(f"Similar pattern match found: {best_match.pattern_id} (similarity: {best_similarity:.2f})")
            return best_match
        
        # No match found, create new pattern
        if pattern_type is None:
            # Try to infer pattern type
            pattern_type = self._infer_pattern_type(context)
        
        # Generate pattern ID
        pattern_id = f"pattern_{int(time.time())}_{hash(signature) % 10000}"
        
        # Create new pattern
        new_pattern = BugPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            signature=signature,
            description=self._generate_description(context, features),
            confidence=0.5,  # Initial confidence
            detection_count=1
        )
        
        # Add context with features
        context["features"] = features
        new_pattern.add_context(context)
        
        # Add to patterns
        self.patterns[pattern_id] = new_pattern
        
        # Update pattern types
        self.pattern_types.add(pattern_type)
        
        # Save patterns
        self.save_patterns()
        
        logger.info(f"Created new pattern: {pattern_id} ({pattern_type})")
        
        return new_pattern
    
    def _infer_pattern_type(self, context: Dict[str, Any]) -> str:
        """
        Infer pattern type from context.
        
        Args:
            context: Bug context
            
        Returns:
            Inferred pattern type
        """
        error_message = context.get("error_message", "")
        
        if "TypeError" in error_message:
            return "type_error"
        elif "AttributeError" in error_message:
            return "attribute_error"
        elif "IndexError" in error_message:
            return "index_error"
        elif "KeyError" in error_message:
            return "key_error"
        elif "NameError" in error_message:
            return "name_error"
        elif "SyntaxError" in error_message:
            return "syntax_error"
        elif "ImportError" in error_message or "ModuleNotFoundError" in error_message:
            return "import_error"
        elif "IOError" in error_message or "FileNotFoundError" in error_message:
            return "io_error"
        elif "PermissionError" in error_message:
            return "permission_error"
        elif "ValueError" in error_message:
            return "value_error"
        elif "ZeroDivisionError" in error_message:
            return "zero_division_error"
        elif "MemoryError" in error_message:
            return "memory_error"
        elif "RecursionError" in error_message:
            return "recursion_error"
        elif "TimeoutError" in error_message:
            return "timeout_error"
        elif "RuntimeError" in error_message:
            return "runtime_error"
        elif "AssertionError" in error_message:
            return "assertion_error"
        
        code = context.get("code", "")
        
        if re.search(r"async|await", code):
            return "async_error"
        elif re.search(r"(for|while)", code):
            return "loop_error"
        elif re.search(r"thread|threading|multiprocessing", code):
            return "concurrency_error"
        elif re.search(r"raise\s+Exception", code):
            return "custom_exception"
        
        # Default to unknown
        return "unknown_error"
    
    def _generate_description(self, context: Dict[str, Any], features: Dict[str, Any]) -> str:
        """
        Generate a description for a bug pattern.
        
        Args:
            context: Bug context
            features: Extracted features
            
        Returns:
            Generated description
        """
        description = "Bug pattern with the following characteristics:\n"
        
        # Add error information if available
        error_message = context.get("error_message", "")
        if error_message:
            description += f"Error: {error_message}\n"
        
        # Add code structure information
        code_structure = features.get("code_structure", {})
        if code_structure:
            description += "Code structure:\n"
            for key, value in code_structure.items():
                description += f"- {key}: {value}\n"
        
        # Add API usage information
        api_usage = features.get("api_usage", {})
        if api_usage:
            imports = api_usage.get("imports", [])
            if imports:
                description += f"Imports: {', '.join(imports)}\n"
            
            method_calls = api_usage.get("method_calls", [])
            if method_calls:
                methods = [f"{obj}.{method}" for obj, method in method_calls]
                description += f"Method calls: {', '.join(methods)}\n"
        
        # Add variable usage information
        variable_usage = features.get("variable_usage", {})
        if variable_usage:
            variables = variable_usage.get("referenced_variables", [])
            if variables:
                description += f"Variables: {', '.join(variables)}\n"
        
        return description
    
    def get_pattern(self, pattern_id: str) -> Optional[BugPattern]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Bug pattern, or None if not found
        """
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: str) -> List[BugPattern]:
        """
        Get patterns by type.
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            List of matching patterns
        """
        return [p for p in self.patterns.values() if p.pattern_type == pattern_type]
    
    def get_pattern_types(self) -> List[str]:
        """
        Get all pattern types.
        
        Returns:
            List of pattern types
        """
        return sorted(self.pattern_types)
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a pattern.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Whether the pattern was deleted
        """
        if pattern_id in self.patterns:
            deleted_pattern = self.patterns.pop(pattern_id)
            
            # Update pattern types
            self.pattern_types = set(p.pattern_type for p in self.patterns.values())
            
            # Save patterns
            self.save_patterns()
            
            logger.info(f"Deleted pattern: {pattern_id}")
            return True
        
        return False
