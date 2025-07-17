#!/usr/bin/env python3
"""
scope_filter_demo.py
────────────────────
Demonstration of the enhanced scope filter capabilities.

This script showcases:
1. Basic filtering to exclude build artifacts and dependencies
2. Language-specific filtering
3. Relevance-based sorting
4. Project language detection
5. Entropy calculation and reduction
6. Performance with large repositories

Usage:
    python scope_filter_demo.py [directory]
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

from scope_filter import ScopeFilter, _EXTENSION_MAPPINGS


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_section(title):
    """Print a section title."""
    print(f"\n--- {title} ---")


def demo_basic_filtering(repo_root: Path):
    """Demonstrate basic file filtering."""
    print_header("Basic Filtering")
    
    # Create a default filter
    filter = ScopeFilter()
    
    # Measure filtering time
    start_time = time.time()
    filtered_files = filter.filter(repo_root)
    elapsed = time.time() - start_time
    
    # Calculate stats
    all_files = sum(1 for _ in repo_root.rglob("*") if Path(_).is_file())
    filtered_count = len(filtered_files)
    reduction_percent = (1 - filtered_count / all_files) * 100 if all_files > 0 else 0
    
    print(f"Repository: {repo_root}")
    print(f"Total files: {all_files}")
    print(f"Filtered files: {filtered_count}")
    print(f"Reduction: {reduction_percent:.1f}%")
    print(f"Filtering time: {elapsed:.3f} seconds")
    
    # Show a sample of filtered files
    print("\nSample of included files:")
    for file in sorted(filtered_files)[:10]:
        rel_path = file.relative_to(repo_root)
        print(f"  {rel_path}")
    
    return filter, filtered_files


def demo_language_filtering(filter: ScopeFilter, repo_root: Path):
    """Demonstrate language-specific filtering."""
    print_header("Language-Specific Filtering")
    
    # Detect languages in the project
    languages = filter.detect_project_languages(repo_root)
    
    print("Detected languages:")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} files")
    
    # Filter for specific languages
    primary_languages = ["python", "javascript", "typescript", "web"]
    available_languages = [lang for lang in primary_languages if lang in languages]
    
    if not available_languages:
        print("\nNo primary languages detected in this repository.")
        return
    
    print("\nFiltering by specific languages:")
    for lang in available_languages:
        language_files = filter.filter_by_language(repo_root, lang)
        print(f"  {lang}: {len(language_files)} files")
        
        # Show a sample
        if language_files:
            print(f"    Sample {lang} files:")
            for file in sorted(language_files)[:3]:
                rel_path = file.relative_to(repo_root)
                print(f"      {rel_path}")


def demo_relevance_sorting(filter: ScopeFilter, repo_root: Path, keyword: str = ""):
    """Demonstrate relevance-based sorting."""
    print_header("Relevance-Based Sorting")
    
    # If no keyword was provided, try to find a good one
    if not keyword:
        # Look for common keywords in repositories
        candidates = ["main", "utils", "config", "test", "api", "model", "controller"]
        
        # Filter the repository
        filtered_files = filter.filter(repo_root)
        
        # Count occurrences of each candidate in filenames
        keyword_counts = {}
        for candidate in candidates:
            count = sum(1 for f in filtered_files if candidate.lower() in f.name.lower())
            if count > 0:
                keyword_counts[candidate] = count
        
        # Choose the most common keyword
        if keyword_counts:
            keyword = max(keyword_counts.items(), key=lambda x: x[1])[0]
        else:
            keyword = "test"  # Default to "test" if none found
    
    print(f"Sorting files by relevance to keyword: '{keyword}'")
    
    # Filter and sort
    filtered_files = filter.filter(repo_root)
    sorted_files = filter.sort_by_relevance(filtered_files, keyword=keyword)
    
    # Show top 10 most relevant files
    print("\nTop 10 most relevant files:")
    for i, file in enumerate(sorted_files[:10], 1):
        rel_path = file.relative_to(repo_root)
        print(f"  {i}. {rel_path}")
    
    # Show statistics
    matching_files = sum(1 for f in sorted_files if keyword.lower() in f.name.lower())
    print(f"\nFiles containing '{keyword}' in name: {matching_files}")
    print(f"Total files considered: {len(sorted_files)}")


def demo_entropy_calculation(filter: ScopeFilter, repo_root: Path):
    """Demonstrate entropy calculation and reduction."""
    print_header("Entropy Calculation")
    
    # Calculate entropy for the entire repository
    all_files = [p for p in repo_root.rglob("*") if p.is_file()]
    base_entropy = filter.entropy_bits(len(all_files))
    
    # Filter the repository
    filtered_files = filter.filter(repo_root)
    filtered_entropy = filter.entropy_bits(len(filtered_files))
    
    # Calculate enhanced entropy with language diversity
    enhanced_entropy = filter.calculate_entropy(filtered_files)
    
    print(f"Repository: {repo_root}")
    print(f"Total files: {len(all_files)}")
    print(f"Base entropy (log₂|files|): {base_entropy:.2f} bits")
    print(f"Filtered entropy: {filtered_entropy:.2f} bits")
    print(f"Enhanced entropy (with language diversity): {enhanced_entropy:.2f} bits")
    print(f"Entropy reduction: {base_entropy - filtered_entropy:.2f} bits ({(1 - filtered_entropy/base_entropy) * 100:.1f}%)")
    
    # Demonstrate entropy clamping
    print_section("Entropy Clamping")
    
    # Create filters with different entropy limits
    limits = [8, 10, 12]  # 256, 1024, 4096 files
    
    print("Effect of different entropy limits:")
    for bits in limits:
        limit_filter = ScopeFilter(max_entropy_bits=bits)
        limited_files = limit_filter.filter(repo_root)
        max_files = 1 << bits
        
        print(f"  {bits} bits ({max_files} max files): {len(limited_files)} files included")


def demo_custom_filtering(repo_root: Path):
    """Demonstrate custom filtering configurations."""
    print_header("Custom Filtering Configurations")
    
    # Configuration 1: Python-only filter
    python_filter = ScopeFilter(
        languages=["python"],
        exclude_languages=["javascript", "typescript"],
        max_entropy_bits=10  # Limit to 1024 files
    )
    
    # Configuration 2: Web development filter
    web_filter = ScopeFilter(
        languages=["javascript", "typescript", "web"],
        block_globs=["**/test_*.js", "**/test_*.ts", "**/node_modules/**"],
        max_entropy_bits=10
    )
    
    # Configuration 3: Documentation filter
    docs_filter = ScopeFilter(
        allow_globs=["**/*.md", "**/*.rst", "**/*.txt", "**/docs/**"],
        max_entropy_bits=8  # Limit to 256 files
    )
    
    # Run all configurations
    configs = [
        ("Python Filter", python_filter),
        ("Web Development Filter", web_filter),
        ("Documentation Filter", docs_filter)
    ]
    
    for name, filter_config in configs:
        start_time = time.time()
        files = filter_config.filter(repo_root)
        elapsed = time.time() - start_time
        
        print(f"\n{name}:")
        print(f"  Files included: {len(files)}")
        print(f"  Filter configuration: {filter_config}")
        print(f"  Filtering time: {elapsed:.3f} seconds")
        
        if files:
            print("  Sample files:")
            for file in sorted(files)[:3]:
                rel_path = file.relative_to(repo_root)
                print(f"    {rel_path}")


def main():
    """Run the demonstration with the specified or current directory."""
    print_header("SCOPE FILTER DEMONSTRATION")
    
    # Determine repository root
    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1]).resolve()
    else:
        repo_root = Path.cwd()
    
    if not repo_root.is_dir():
        print(f"Error: {repo_root} is not a valid directory")
        return
    
    # Run demonstrations
    filter, files = demo_basic_filtering(repo_root)
    demo_language_filtering(filter, repo_root)
    demo_relevance_sorting(filter, repo_root)
    demo_entropy_calculation(filter, repo_root)
    demo_custom_filtering(repo_root)
    
    print_header("DEMONSTRATION COMPLETE")
    print(f"Repository: {repo_root}")
    print(f"Enhanced ScopeFilter provides efficient, focused filtering for bug fixing,")
    print(f"reducing search space complexity and increasing debug velocity.")


if __name__ == "__main__":
    main()
