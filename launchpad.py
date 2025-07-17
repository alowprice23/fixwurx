"""
launchpad.py
───────────
Re-exports Launchpad and ComponentRegistry classes from components/launchpad.py.

This module is provided for backward compatibility and to support the 
integration tests that import directly from 'launchpad'.
"""

from components.launchpad import Launchpad, ComponentRegistry
