#!/usr/bin/env python3
"""
progress_indicators.py
─────────────────────
Progress indicator system for the FixWurx platform.

This module provides visual feedback for long-running operations,
with support for different indicator styles, nested operations,
and real-time updates.
"""

import os
import sys
import time
import threading
import logging
import math
import shutil
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from enum import Enum, auto
import io
import contextlib

# Internal imports
from shell_environment import register_event_handler, emit_event, EventType

# Configure logging
logger = logging.getLogger("ProgressIndicators")

# Constants
DEFAULT_UPDATE_INTERVAL = 0.1  # seconds
DEFAULT_WIDTH = 50  # characters
DEFAULT_SPINNER_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
DEFAULT_BAR_FILL_CHAR = '█'
DEFAULT_BAR_EMPTY_CHAR = '░'
DEFAULT_BAR_HEAD_CHAR = ''
DEFAULT_ETA_SMOOTHING = 0.3  # ETA smoothing factor (0-1)
DEFAULT_MIN_REFRESH_INTERVAL = 0.05  # seconds

class IndicatorStyle(Enum):
    """Progress indicator styles."""
    BAR = auto()  # Progress bar with percentage
    SPINNER = auto()  # Spinning animation
    TEXT = auto()  # Text only with percentage
    DOTS = auto()  # Animated dots
    COUNTER = auto()  # Simple counter (e.g., [1/10])
    DETAILED = auto()  # Detailed progress with rate and ETA

class ProgressState(Enum):
    """Progress indicator states."""
    PENDING = auto()  # Not started
    RUNNING = auto()  # In progress
    PAUSED = auto()  # Temporarily paused
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()  # Failed with error
    CANCELLED = auto()  # Cancelled by user

class ProgressManager:
    """
    Manages progress indicators for the FixWurx platform.
    
    This class provides a central management system for all progress indicators,
    ensuring they are updated correctly and handling nested progress operations.
    """
    
    def __init__(self):
        """Initialize the progress manager."""
        self._indicators: Dict[str, 'ProgressIndicator'] = {}
        self._indicators_lock = threading.RLock()
        self._update_thread = None
        self._stop_update = threading.Event()
        self._update_interval = DEFAULT_UPDATE_INTERVAL
        self._terminal_width = self._get_terminal_width()
        self._last_output_length = 0
        self._enabled = True
        self._output_mode = "terminal"  # "terminal", "log", "silent", "callback"
        self._output_callback = None
        self._active_indicator = None
        
        # Register with shell environment
        try:
            register_event_handler(EventType.TERMINAL_RESIZE, self._handle_terminal_resize)
        except Exception as e:
            logger.warning(f"Failed to register terminal resize handler: {e}")
        
        logger.info("Progress manager initialized")
    
    def _get_terminal_width(self) -> int:
        """Get terminal width."""
        try:
            width, _ = shutil.get_terminal_size()
            return width
        except Exception:
            return 80  # Default width
    
    def _handle_terminal_resize(self, event_data: Dict[str, Any]) -> None:
        """Handle terminal resize event."""
        self._terminal_width = self._get_terminal_width()
    
    def start_update_thread(self) -> None:
        """Start the update thread."""
        if self._update_thread is None or not self._update_thread.is_alive():
            self._stop_update.clear()
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True,
                name="ProgressUpdateThread"
            )
            self._update_thread.start()
    
    def stop_update_thread(self) -> None:
        """Stop the update thread."""
        if self._update_thread and self._update_thread.is_alive():
            self._stop_update.set()
            self._update_thread.join(timeout=2.0)
    
    def _update_loop(self) -> None:
        """Update all active progress indicators periodically."""
        while not self._stop_update.is_set():
            try:
                if self._enabled:
                    self._update_indicators()
            except Exception as e:
                logger.error(f"Error updating progress indicators: {e}")
            
            time.sleep(self._update_interval)
    
    def _update_indicators(self) -> None:
        """Update all active progress indicators."""
        with self._indicators_lock:
            if not self._indicators:
                return
            
            # Determine the active indicator (top-level or specifically activated)
            active_indicator = self._active_indicator
            if active_indicator is None:
                # Find the top-level indicator
                for indicator in self._indicators.values():
                    if indicator.parent_id is None and indicator.state == ProgressState.RUNNING:
                        active_indicator = indicator
                        break
            
            if active_indicator:
                self._render_indicator(active_indicator)
    
    def _render_indicator(self, indicator: 'ProgressIndicator') -> None:
        """
        Render a progress indicator.
        
        Args:
            indicator: Progress indicator to render.
        """
        # Collect all child indicators
        children = []
        for child in self._indicators.values():
            if child.parent_id == indicator.id:
                children.append(child)
        
        # Get output string for indicator and its children
        output = indicator.render(self._terminal_width, children)
        
        # Output based on mode
        if self._output_mode == "terminal":
            self._terminal_output(output)
        elif self._output_mode == "log":
            logger.info(output)
        elif self._output_mode == "callback" and self._output_callback:
            self._output_callback(output)
    
    def _terminal_output(self, output: str) -> None:
        """
        Output to terminal.
        
        Args:
            output: Text to output.
        """
        # Clear previous output
        if self._last_output_length > 0:
            sys.stdout.write('\r' + ' ' * self._last_output_length)
        
        # Write new output
        sys.stdout.write('\r' + output)
        sys.stdout.flush()
        
        # Store output length
        self._last_output_length = len(output)
    
    def create_indicator(self, task_name: str, total: int = 100,
                        style: IndicatorStyle = IndicatorStyle.BAR,
                        parent_id: Optional[str] = None) -> str:
        """
        Create a new progress indicator.
        
        Args:
            task_name: Name of the task.
            total: Total number of steps.
            style: Indicator style.
            parent_id: Parent indicator ID.
            
        Returns:
            Indicator ID.
        """
        with self._indicators_lock:
            # Create indicator
            indicator = ProgressIndicator(
                task_name=task_name,
                total=total,
                style=style,
                parent_id=parent_id
            )
            
            # Add to indicators
            self._indicators[indicator.id] = indicator
            
            # Start update thread if not running
            if not self._update_thread or not self._update_thread.is_alive():
                self.start_update_thread()
            
            return indicator.id
    
    def update_indicator(self, indicator_id: str, progress: int,
                        message: Optional[str] = None) -> None:
        """
        Update a progress indicator.
        
        Args:
            indicator_id: Indicator ID.
            progress: Current progress value.
            message: Optional status message.
        """
        with self._indicators_lock:
            if indicator_id not in self._indicators:
                logger.warning(f"Indicator {indicator_id} not found")
                return
            
            indicator = self._indicators[indicator_id]
            indicator.update(progress, message)
    
    def complete_indicator(self, indicator_id: str, 
                          message: Optional[str] = None) -> None:
        """
        Mark a progress indicator as completed.
        
        Args:
            indicator_id: Indicator ID.
            message: Optional completion message.
        """
        with self._indicators_lock:
            if indicator_id not in self._indicators:
                logger.warning(f"Indicator {indicator_id} not found")
                return
            
            indicator = self._indicators[indicator_id]
            indicator.complete(message)
    
    def fail_indicator(self, indicator_id: str, 
                      message: Optional[str] = None) -> None:
        """
        Mark a progress indicator as failed.
        
        Args:
            indicator_id: Indicator ID.
            message: Optional failure message.
        """
        with self._indicators_lock:
            if indicator_id not in self._indicators:
                logger.warning(f"Indicator {indicator_id} not found")
                return
            
            indicator = self._indicators[indicator_id]
            indicator.fail(message)
    
    def cancel_indicator(self, indicator_id: str,
                        message: Optional[str] = None) -> None:
        """
        Cancel a progress indicator.
        
        Args:
            indicator_id: Indicator ID.
            message: Optional cancellation message.
        """
        with self._indicators_lock:
            if indicator_id not in self._indicators:
                logger.warning(f"Indicator {indicator_id} not found")
                return
            
            indicator = self._indicators[indicator_id]
            indicator.cancel(message)
    
    def pause_indicator(self, indicator_id: str,
                       message: Optional[str] = None) -> None:
        """
        Pause a progress indicator.
        
        Args:
            indicator_id: Indicator ID.
            message: Optional pause message.
        """
        with self._indicators_lock:
            if indicator_id not in self._indicators:
                logger.warning(f"Indicator {indicator_id} not found")
                return
            
            indicator = self._indicators[indicator_id]
            indicator.pause(message)
    
    def resume_indicator(self, indicator_id: str) -> None:
        """
        Resume a paused progress indicator.
        
        Args:
            indicator_id: Indicator ID.
        """
        with self._indicators_lock:
            if indicator_id not in self._indicators:
                logger.warning(f"Indicator {indicator_id} not found")
                return
            
            indicator = self._indicators[indicator_id]
            indicator.resume()
    
    def remove_indicator(self, indicator_id: str) -> None:
        """
        Remove a progress indicator.
        
        Args:
            indicator_id: Indicator ID.
        """
        with self._indicators_lock:
            if indicator_id not in self._indicators:
                return
            
            # Remove indicator
            del self._indicators[indicator_id]
            
            # Reset active indicator if needed
            if self._active_indicator and self._active_indicator.id == indicator_id:
                self._active_indicator = None
            
            # Clear terminal
            if self._output_mode == "terminal":
                if self._last_output_length > 0:
                    sys.stdout.write('\r' + ' ' * self._last_output_length + '\r')
                    sys.stdout.flush()
                    self._last_output_length = 0
    
    def set_active_indicator(self, indicator_id: Optional[str]) -> None:
        """
        Set the active indicator for display.
        
        Args:
            indicator_id: Indicator ID or None to use top-level.
        """
        with self._indicators_lock:
            if indicator_id is None:
                self._active_indicator = None
                return
            
            if indicator_id not in self._indicators:
                logger.warning(f"Indicator {indicator_id} not found")
                return
            
            self._active_indicator = self._indicators[indicator_id]
    
    def set_output_mode(self, mode: str, callback: Optional[Callable] = None) -> None:
        """
        Set the output mode.
        
        Args:
            mode: Output mode ("terminal", "log", "silent", "callback").
            callback: Callback function for "callback" mode.
        """
        valid_modes = ["terminal", "log", "silent", "callback"]
        if mode not in valid_modes:
            logger.warning(f"Invalid output mode: {mode}")
            return
        
        if mode == "callback" and callback is None:
            logger.warning("Callback mode requires a callback function")
            return
        
        self._output_mode = mode
        self._output_callback = callback
    
    def enable(self) -> None:
        """Enable progress indicators."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable progress indicators."""
        self._enabled = False
        
        # Clear terminal
        if self._output_mode == "terminal":
            if self._last_output_length > 0:
                sys.stdout.write('\r' + ' ' * self._last_output_length + '\r')
                sys.stdout.flush()
                self._last_output_length = 0
    
    def is_enabled(self) -> bool:
        """Check if progress indicators are enabled."""
        return self._enabled
    
    def set_update_interval(self, interval: float) -> None:
        """
        Set the update interval.
        
        Args:
            interval: Update interval in seconds.
        """
        if interval < DEFAULT_MIN_REFRESH_INTERVAL:
            interval = DEFAULT_MIN_REFRESH_INTERVAL
        
        self._update_interval = interval
    
    def get_indicator(self, indicator_id: str) -> Optional['ProgressIndicator']:
        """
        Get a progress indicator.
        
        Args:
            indicator_id: Indicator ID.
            
        Returns:
            Progress indicator or None if not found.
        """
        with self._indicators_lock:
            return self._indicators.get(indicator_id)
    
    def get_all_indicators(self) -> Dict[str, 'ProgressIndicator']:
        """
        Get all progress indicators.
        
        Returns:
            Dictionary of progress indicators.
        """
        with self._indicators_lock:
            return self._indicators.copy()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_update_thread()
        
        # Clear terminal
        if self._output_mode == "terminal":
            if self._last_output_length > 0:
                sys.stdout.write('\r' + ' ' * self._last_output_length + '\r')
                sys.stdout.flush()
                self._last_output_length = 0
        
        with self._indicators_lock:
            self._indicators.clear()


class ProgressIndicator:
    """
    Progress indicator for a task.
    
    This class represents a single progress indicator for a task,
    with support for different styles and real-time updates.
    """
    
    def __init__(self, task_name: str, total: int = 100,
                style: IndicatorStyle = IndicatorStyle.BAR,
                parent_id: Optional[str] = None):
        """
        Initialize a progress indicator.
        
        Args:
            task_name: Name of the task.
            total: Total number of steps.
            style: Indicator style.
            parent_id: Parent indicator ID.
        """
        self.id = f"progress_{int(time.time() * 1000)}_{id(self)}"
        self.task_name = task_name
        self.total = max(1, total)  # Ensure total is at least 1
        self.progress = 0
        self.message = None
        self.style = style
        self.parent_id = parent_id
        self.state = ProgressState.PENDING
        self.created_at = time.time()
        self.started_at = None
        self.updated_at = None
        self.completed_at = None
        self.last_progress = 0
        self.last_update_time = None
        self.eta = None
        self.rate = 0.0  # Progress per second
        
        # Style-specific settings
        self.width = DEFAULT_WIDTH
        self.fill_char = DEFAULT_BAR_FILL_CHAR
        self.empty_char = DEFAULT_BAR_EMPTY_CHAR
        self.head_char = DEFAULT_BAR_HEAD_CHAR
        self.spinner_chars = DEFAULT_SPINNER_CHARS
        self.spinner_index = 0
        self.eta_smoothing = DEFAULT_ETA_SMOOTHING
    
    def update(self, progress: int, message: Optional[str] = None) -> None:
        """
        Update progress indicator.
        
        Args:
            progress: Current progress value.
            message: Optional status message.
        """
        now = time.time()
        
        # Check if first update
        if self.state == ProgressState.PENDING:
            self.state = ProgressState.RUNNING
            self.started_at = now
        
        # Check if already completed
        if self.state in [ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED]:
            return
        
        # Resume if paused
        if self.state == ProgressState.PAUSED:
            self.state = ProgressState.RUNNING
        
        # Ensure progress is within bounds
        self.progress = max(0, min(progress, self.total))
        
        # Update message if provided
        if message is not None:
            self.message = message
        
        # Calculate rate and ETA
        if self.last_update_time is not None:
            time_diff = now - self.last_update_time
            progress_diff = self.progress - self.last_progress
            
            if time_diff > 0:
                current_rate = progress_diff / time_diff
                
                # Use exponential moving average for rate
                if self.rate == 0:
                    self.rate = current_rate
                else:
                    self.rate = (self.eta_smoothing * current_rate + 
                                (1 - self.eta_smoothing) * self.rate)
                
                # Calculate ETA
                if self.rate > 0:
                    remaining = self.total - self.progress
                    eta_seconds = remaining / self.rate
                    
                    # Use exponential moving average for ETA
                    if self.eta is None:
                        self.eta = eta_seconds
                    else:
                        self.eta = (self.eta_smoothing * eta_seconds + 
                                   (1 - self.eta_smoothing) * self.eta)
        
        # Update timestamps
        self.last_progress = self.progress
        self.last_update_time = now
        self.updated_at = now
        
        # Check if completed
        if self.progress >= self.total:
            self.complete()
    
    def complete(self, message: Optional[str] = None) -> None:
        """
        Mark as completed.
        
        Args:
            message: Optional completion message.
        """
        self.progress = self.total
        self.state = ProgressState.COMPLETED
        self.completed_at = time.time()
        self.eta = 0
        
        if message is not None:
            self.message = message
    
    def fail(self, message: Optional[str] = None) -> None:
        """
        Mark as failed.
        
        Args:
            message: Optional failure message.
        """
        self.state = ProgressState.FAILED
        self.completed_at = time.time()
        
        if message is not None:
            self.message = message
    
    def cancel(self, message: Optional[str] = None) -> None:
        """
        Mark as cancelled.
        
        Args:
            message: Optional cancellation message.
        """
        self.state = ProgressState.CANCELLED
        self.completed_at = time.time()
        
        if message is not None:
            self.message = message
    
    def pause(self, message: Optional[str] = None) -> None:
        """
        Pause progress.
        
        Args:
            message: Optional pause message.
        """
        if self.state == ProgressState.RUNNING:
            self.state = ProgressState.PAUSED
            
            if message is not None:
                self.message = message
    
    def resume(self) -> None:
        """Resume paused progress."""
        if self.state == ProgressState.PAUSED:
            self.state = ProgressState.RUNNING
    
    def get_percentage(self) -> float:
        """
        Get progress percentage.
        
        Returns:
            Progress percentage (0-100).
        """
        return (self.progress / self.total) * 100
    
    def get_elapsed_time(self) -> float:
        """
        Get elapsed time.
        
        Returns:
            Elapsed time in seconds.
        """
        if self.started_at is None:
            return 0.0
        
        if self.completed_at is not None:
            return self.completed_at - self.started_at
        
        return time.time() - self.started_at
    
    def get_eta_string(self) -> str:
        """
        Get ETA as a string.
        
        Returns:
            ETA string.
        """
        if self.state in [ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED]:
            return "0s"
        
        if self.eta is None or self.eta < 0:
            return "calculating..."
        
        # Format ETA
        eta = max(0, self.eta)
        if eta < 60:
            return f"{int(eta)}s"
        elif eta < 3600:
            minutes = int(eta // 60)
            seconds = int(eta % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def render(self, width: int, children: List['ProgressIndicator'] = None) -> str:
        """
        Render the progress indicator.
        
        Args:
            width: Terminal width.
            children: Child progress indicators.
            
        Returns:
            Rendered progress indicator.
        """
        # Calculate available width
        available_width = width - 2  # Allow for margins
        
        # Create output
        if self.style == IndicatorStyle.BAR:
            output = self._render_bar(available_width)
        elif self.style == IndicatorStyle.SPINNER:
            output = self._render_spinner(available_width)
        elif self.style == IndicatorStyle.TEXT:
            output = self._render_text(available_width)
        elif self.style == IndicatorStyle.DOTS:
            output = self._render_dots(available_width)
        elif self.style == IndicatorStyle.COUNTER:
            output = self._render_counter(available_width)
        elif self.style == IndicatorStyle.DETAILED:
            output = self._render_detailed(available_width)
        else:
            output = self._render_text(available_width)
        
        # Add children if any
        if children:
            child_outputs = []
            for child in children:
                if child.state != ProgressState.PENDING:
                    child_output = child.render(available_width - 2)
                    child_outputs.append("  " + child_output)
            
            if child_outputs:
                output += "\n" + "\n".join(child_outputs)
        
        return output
    
    def _render_bar(self, width: int) -> str:
        """
        Render progress bar.
        
        Args:
            width: Available width.
            
        Returns:
            Rendered progress bar.
        """
        # Calculate bar width
        bar_width = min(width - 20, self.width)  # Reserve space for text and percentage
        
        # Calculate filled width
        percentage = self.get_percentage()
        filled_width = int(bar_width * (percentage / 100))
        
        # Create bar
        if filled_width == bar_width:
            bar = self.fill_char * bar_width
        elif filled_width == 0:
            bar = self.empty_char * bar_width
        else:
            bar = (self.fill_char * filled_width + 
                  (self.head_char if self.head_char else '') + 
                  self.empty_char * (bar_width - filled_width - (1 if self.head_char else 0)))
        
        # Format percentage
        percentage_str = f"{percentage:.1f}%"
        
        # Format task name and message
        max_task_name_width = width - bar_width - len(percentage_str) - 5
        task_display = self.task_name
        if len(task_display) > max_task_name_width:
            task_display = task_display[:max_task_name_width-3] + "..."
        
        # Add state indicator
        if self.state == ProgressState.PAUSED:
            state_indicator = " [PAUSED]"
        elif self.state == ProgressState.COMPLETED:
            state_indicator = " [DONE]"
        elif self.state == ProgressState.FAILED:
            state_indicator = " [FAILED]"
        elif self.state == ProgressState.CANCELLED:
            state_indicator = " [CANCELLED]"
        else:
            state_indicator = ""
        
        # Assemble output
        output = f"{task_display}{state_indicator}: [{bar}] {percentage_str}"
        
        # Add message if available
        if self.message:
            max_message_width = width - len(output) - 2
            if max_message_width > 10:
                message_display = self.message
                if len(message_display) > max_message_width:
                    message_display = message_display[:max_message_width-3] + "..."
                output += f" {message_display}"
        
        return output
    
    def _render_spinner(self, width: int) -> str:
        """
        Render spinner.
        
        Args:
            width: Available width.
            
        Returns:
            Rendered spinner.
        """
        # Update spinner index
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
        spinner = self.spinner_chars[self.spinner_index]
        
        # Format task name and message
        max_task_name_width = width - 10
        task_display = self.task_name
        if len(task_display) > max_task_name_width:
            task_display = task_display[:max_task_name_width-3] + "..."
        
        # Add state indicator
        if self.state == ProgressState.PAUSED:
            state_indicator = " [PAUSED]"
            spinner = "⏸"
        elif self.state == ProgressState.COMPLETED:
            state_indicator = " [DONE]"
            spinner = "✓"
        elif self.state == ProgressState.FAILED:
            state_indicator = " [FAILED]"
            spinner = "✗"
        elif self.state == ProgressState.CANCELLED:
            state_indicator = " [CANCELLED]"
            spinner = "⨯"
        else:
            state_indicator = ""
        
        # Assemble output
        output = f"{spinner} {task_display}{state_indicator}"
        
        # Add message if available
        if self.message:
            max_message_width = width - len(output) - 2
            if max_message_width > 10:
                message_display = self.message
                if len(message_display) > max_message_width:
                    message_display = message_display[:max_message_width-3] + "..."
                output += f": {message_display}"
        
        return output
    
    def _render_text(self, width: int) -> str:
        """
        Render text.
        
        Args:
            width: Available width.
            
        Returns:
            Rendered text.
        """
        # Format percentage
        percentage = self.get_percentage()
        percentage_str = f"{percentage:.1f}%"
        
        # Format task name and message
        max_task_name_width = width - len(percentage_str) - 5
        task_display = self.task_name
        if len(task_display) > max_task_name_width:
            task_display = task_display[:max_task_name_width-3] + "..."
        
        # Add state indicator
        if self.state == ProgressState.PAUSED:
            state_indicator = " [PAUSED]"
        elif self.state == ProgressState.COMPLETED:
            state_indicator = " [DONE]"
        elif self.state == ProgressState.FAILED:
            state_indicator = " [FAILED]"
        elif self.state == ProgressState.CANCELLED:
            state_indicator = " [CANCELLED]"
        else:
            state_indicator = ""
        
        # Assemble output
        output = f"{task_display}{state_indicator}: {percentage_str}"
        
        # Add message if available
        if self.message:
            max_message_width = width - len(output) - 2
            if max_message_width > 10:
                message_display = self.message
                if len(message_display) > max_message_width:
                    message_display = message_display[:max_message_width-3] + "..."
                output += f" {message_display}"
        
        return output
    
    def _render_dots(self, width: int) -> str:
        """
        Render animated dots.
        
        Args:
            width: Available width.
            
        Returns:
            Rendered dots.
        """
        # Calculate number of dots
        num_dots = (int(time.time() * 2) % 4)
        dots = "." * num_dots + " " * (3 - num_dots)
        
        # Format task name and message
        max_task_name_width = width - 10
        task_display = self.task_name
        if len(task_display) > max_task_name_width:
            task_display = task_display[:max_task_name_width-3] + "..."
        
        # Add state indicator
        if self.state == ProgressState.PAUSED:
            state_indicator = " [PAUSED]"
            dots = ".."
        elif self.state == ProgressState.COMPLETED:
            state_indicator = " [DONE]"
            dots = "   "
        elif self.state == ProgressState.COMPLETED:
            state_indicator = " [DONE]"
            dots = ""
        elif self.state == ProgressState.FAILED:
            state_indicator = " [FAILED]"
            dots = ""
        elif self.state == ProgressState.CANCELLED:
            state_indicator = " [CANCELLED]"
            dots = ""
        else:
            state_indicator = ""
        
        # Assemble output
        output = f"{task_display}{state_indicator}{dots}"
        
        # Add message if available
        if self.message:
            max_message_width = width - len(output) - 2
            if max_message_width > 10:
                message_display = self.message
                if len(message_display) > max_message_width:
                    message_display = message_display[:max_message_width-3] + "..."
                output += f" {message_display}"
        
        return output
    
    def _render_counter(self, width: int) -> str:
        """
        Render counter.
        
        Args:
            width: Available width.
            
        Returns:
            Rendered counter.
        """
        # Format counter
        counter = f"[{self.progress}/{self.total}]"
        
        # Format task name and message
        max_task_name_width = width - len(counter) - 5
        task_display = self.task_name
        if len(task_display) > max_task_name_width:
            task_display = task_display[:max_task_name_width-3] + "..."
        
        # Add state indicator
        if self.state == ProgressState.PAUSED:
            state_indicator = " [PAUSED]"
        elif self.state == ProgressState.COMPLETED:
            state_indicator = " [DONE]"
        elif self.state == ProgressState.FAILED:
            state_indicator = " [FAILED]"
        elif self.state == ProgressState.CANCELLED:
            state_indicator = " [CANCELLED]"
        else:
            state_indicator = ""
        
        # Assemble output
        output = f"{task_display}{state_indicator}: {counter}"
        
        # Add message if available
        if self.message:
            max_message_width = width - len(output) - 2
            if max_message_width > 10:
                message_display = self.message
                if len(message_display) > max_message_width:
                    message_display = message_display[:max_message_width-3] + "..."
                output += f" {message_display}"
        
        return output
    
    def _render_detailed(self, width: int) -> str:
        """
        Render detailed progress.
        
        Args:
            width: Available width.
            
        Returns:
            Rendered detailed progress.
        """
        # Calculate bar width
        bar_width = min(width - 40, self.width)  # Reserve space for text, percentage, rate, ETA
        
        # Calculate filled width
        percentage = self.get_percentage()
        filled_width = int(bar_width * (percentage / 100))
        
        # Create bar
        if filled_width == bar_width:
            bar = self.fill_char * bar_width
        elif filled_width == 0:
            bar = self.empty_char * bar_width
        else:
            bar = (self.fill_char * filled_width + 
                  (self.head_char if self.head_char else '') + 
                  self.empty_char * (bar_width - filled_width - (1 if self.head_char else 0)))
        
        # Format percentage
        percentage_str = f"{percentage:.1f}%"
        
        # Format rate
        if self.rate > 1000:
            rate_str = f"{self.rate/1000:.1f}k/s"
        else:
            rate_str = f"{self.rate:.1f}/s"
        
        # Format ETA
        eta_str = self.get_eta_string()
        
        # Format elapsed time
        elapsed = self.get_elapsed_time()
        if elapsed < 60:
            elapsed_str = f"{int(elapsed)}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            elapsed_str = f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            elapsed_str = f"{hours}h {minutes}m"
        
        # Format task name
        max_task_name_width = width - bar_width - len(percentage_str) - len(rate_str) - len(eta_str) - 15
        task_display = self.task_name
        if len(task_display) > max_task_name_width:
            task_display = task_display[:max_task_name_width-3] + "..."
        
        # Add state indicator
        if self.state == ProgressState.PAUSED:
            state_indicator = " [PAUSED]"
        elif self.state == ProgressState.COMPLETED:
            state_indicator = " [DONE]"
        elif self.state == ProgressState.FAILED:
            state_indicator = " [FAILED]"
        elif self.state == ProgressState.CANCELLED:
            state_indicator = " [CANCELLED]"
        else:
            state_indicator = ""
        
        # Assemble output
        output = f"{task_display}{state_indicator}: [{bar}] {percentage_str} {rate_str} ETA: {eta_str}"
        
        # Add message if available
        if self.message:
            # If enough space, add message on same line
            if len(output) + 3 + len(self.message) <= width:
                output += f" {self.message}"
            # Otherwise, add message on next line with clear indent
            else:
                output += f"\n  {self.message}"
        
        return output


# Create global instance
progress_manager = ProgressManager()

def create_progress(task_name: str, total: int = 100,
                  style: IndicatorStyle = IndicatorStyle.BAR,
                  parent_id: Optional[str] = None) -> str:
    """
    Create a new progress indicator.
    
    Args:
        task_name: Name of the task.
        total: Total number of steps.
        style: Indicator style.
        parent_id: Parent indicator ID.
        
    Returns:
        Indicator ID.
    """
    return progress_manager.create_indicator(task_name, total, style, parent_id)

def update_progress(indicator_id: str, progress: int,
                  message: Optional[str] = None) -> None:
    """
    Update a progress indicator.
    
    Args:
        indicator_id: Indicator ID.
        progress: Current progress value.
        message: Optional status message.
    """
    progress_manager.update_indicator(indicator_id, progress, message)

def complete_progress(indicator_id: str, 
                    message: Optional[str] = None) -> None:
    """
    Mark a progress indicator as completed.
    
    Args:
        indicator_id: Indicator ID.
        message: Optional completion message.
    """
    progress_manager.complete_indicator(indicator_id, message)

def fail_progress(indicator_id: str, 
                message: Optional[str] = None) -> None:
    """
    Mark a progress indicator as failed.
    
    Args:
        indicator_id: Indicator ID.
        message: Optional failure message.
    """
    progress_manager.fail_indicator(indicator_id, message)

def cancel_progress(indicator_id: str,
                  message: Optional[str] = None) -> None:
    """
    Cancel a progress indicator.
    
    Args:
        indicator_id: Indicator ID.
        message: Optional cancellation message.
    """
    progress_manager.cancel_indicator(indicator_id, message)

def pause_progress(indicator_id: str,
                 message: Optional[str] = None) -> None:
    """
    Pause a progress indicator.
    
    Args:
        indicator_id: Indicator ID.
        message: Optional pause message.
    """
    progress_manager.pause_indicator(indicator_id, message)

def resume_progress(indicator_id: str) -> None:
    """
    Resume a paused progress indicator.
    
    Args:
        indicator_id: Indicator ID.
    """
    progress_manager.resume_indicator(indicator_id)

def remove_progress(indicator_id: str) -> None:
    """
    Remove a progress indicator.
    
    Args:
        indicator_id: Indicator ID.
    """
    progress_manager.remove_indicator(indicator_id)

def enable_progress() -> None:
    """Enable progress indicators."""
    progress_manager.enable()

def disable_progress() -> None:
    """Disable progress indicators."""
    progress_manager.disable()

def set_progress_output_mode(mode: str, callback: Optional[Callable] = None) -> None:
    """
    Set the output mode for progress indicators.
    
    Args:
        mode: Output mode ("terminal", "log", "silent", "callback").
        callback: Callback function for "callback" mode.
    """
    progress_manager.set_output_mode(mode, callback)

def cleanup_progress() -> None:
    """Clean up progress indicators."""
    progress_manager.cleanup()

@contextlib.contextmanager
def progress_bar(task_name: str, total: int = 100, 
                style: IndicatorStyle = IndicatorStyle.BAR,
                parent_id: Optional[str] = None):
    """
    Context manager for progress bar.
    
    Args:
        task_name: Name of the task.
        total: Total number of steps.
        style: Indicator style.
        parent_id: Parent indicator ID.
        
    Yields:
        Progress indicator ID.
    """
    indicator_id = create_progress(task_name, total, style, parent_id)
    try:
        yield indicator_id
        # If not explicitly completed, complete it
        if progress_manager.get_indicator(indicator_id) and progress_manager.get_indicator(indicator_id).state not in [
                ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED]:
            complete_progress(indicator_id)
    except Exception as e:
        fail_progress(indicator_id, str(e))
        raise

# Initialize on module load
progress_manager.start_update_thread()
