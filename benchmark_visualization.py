"""
FixWurx Auditor - Benchmark Visualization System

This module provides visualization tools for analyzing benchmark data
and sensor reports to identify trends, anomalies, and patterns.
"""

import logging
import time
import datetime
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger('benchmark_visualization')

class BenchmarkVisualization:
    """
    Tools for visualizing benchmark data to identify trends and anomalies.
    
    This system creates various types of visualizations tailored for different
    aspects of benchmark data analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BenchmarkVisualization.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Output directory for visualizations
        self.output_dir = self.config.get("output_dir", "auditor_data/visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('ggplot')
        
        # Default figure settings
        self.default_figsize = self.config.get("figsize", (12, 8))
        self.default_dpi = self.config.get("dpi", 100)
        
        # Color schemes
        self.color_schemes = {
            "default": plt.cm.tab10,
            "sequential": plt.cm.viridis,
            "diverging": plt.cm.RdYlBu,
            "qualitative": plt.cm.Set3,
            "error_severity": {
                "CRITICAL": "#FF0000",  # Red
                "HIGH": "#FF6600",      # Orange
                "MEDIUM": "#FFCC00",    # Yellow
                "LOW": "#BBBBBB",       # Gray
                "INFO": "#00CCFF"       # Blue
            }
        }
        
        # Default plot theme
        self.theme = self.config.get("theme", "default")
        
        logger.info(f"Initialized BenchmarkVisualization in {self.output_dir}")
    
    def time_series_plot(self, data: List[Dict[str, Any]], 
                        metrics: List[str],
                        title: str = "Performance Metrics Over Time",
                        filename: Optional[str] = None) -> str:
        """
        Create a time series plot for one or more metrics.
        
        Args:
            data: List of benchmark data points
            metrics: List of metric names to plot
            title: Plot title
            filename: Optional filename to save the plot
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            # Extract timestamps and convert to datetime
            timestamps = [datetime.datetime.fromtimestamp(point["timestamp"]) 
                         for point in data]
            
            # Plot each metric
            for i, metric in enumerate(metrics):
                values = []
                valid_timestamps = []
                
                # Extract values, handling missing data
                for j, point in enumerate(data):
                    if metric in point:
                        values.append(point[metric])
                        valid_timestamps.append(timestamps[j])
                
                if values:
                    color = self.color_schemes["default"](i % 10)
                    ax.plot(valid_timestamps, values, 
                           label=metric, marker='o', markersize=4, 
                           linestyle='-', linewidth=2, color=color)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Add legend
            ax.legend(loc='best', fontsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"time_series_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Created time series plot: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            return ""
    
    def multi_session_comparison(self, 
                               session_data: Dict[str, List[Dict[str, Any]]],
                               metric: str,
                               title: Optional[str] = None,
                               filename: Optional[str] = None) -> str:
        """
        Create a comparison plot of a metric across multiple sessions.
        
        Args:
            session_data: Dictionary mapping session names to benchmark data
            metric: Metric name to compare
            title: Optional plot title
            filename: Optional filename to save the plot
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            # Generate title if not provided
            if title is None:
                title = f"Comparison of {metric} Across Sessions"
            
            # Plot each session
            for i, (session_name, data) in enumerate(session_data.items()):
                # Extract timestamps and values
                timestamps = [point["timestamp"] for point in data if metric in point]
                values = [point[metric] for point in data if metric in point]
                
                if timestamps and values:
                    # Convert timestamps to relative time (hours from start)
                    min_time = min(timestamps)
                    rel_times = [(t - min_time) / 3600 for t in timestamps]
                    
                    color = self.color_schemes["default"](i % 10)
                    ax.plot(rel_times, values, 
                           label=session_name, marker='o', markersize=4, 
                           linestyle='-', linewidth=2, color=color)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time (hours from start)", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            
            # Add legend
            ax.legend(loc='best', fontsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"session_comparison_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Created session comparison plot: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating session comparison plot: {str(e)}")
            return ""
    
    def error_distribution_chart(self, 
                               error_data: Dict[str, int],
                               title: str = "Error Distribution",
                               filename: Optional[str] = None) -> str:
        """
        Create a pie or bar chart showing the distribution of errors by type.
        
        Args:
            error_data: Dictionary mapping error types to counts
            title: Plot title
            filename: Optional filename to save the plot
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            # Extract data
            error_types = list(error_data.keys())
            counts = list(error_data.values())
            
            # Use pie chart if few categories, bar chart if many
            if len(error_types) <= 7:
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    counts, 
                    labels=error_types, 
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=False
                )
                
                # Style the text
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                    autotext.set_weight('bold')
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
            else:
                # Sort by count (descending)
                sorted_data = sorted(zip(error_types, counts), key=lambda x: x[1], reverse=True)
                error_types, counts = zip(*sorted_data)
                
                # Create bar chart
                bars = ax.bar(error_types, counts)
                
                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height}', ha='center', va='bottom', fontsize=10)
                
                # Rotate x-axis labels for readability
                plt.xticks(rotation=45, ha='right')
                
                # Add labels
                ax.set_xlabel("Error Type", fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
            
            # Set title
            ax.set_title(title, fontsize=16)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"error_distribution_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Created error distribution chart: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating error distribution chart: {str(e)}")
            return ""
    
    def component_health_heatmap(self, 
                               component_health: Dict[str, Dict[str, float]],
                               title: str = "Component Health Heatmap",
                               filename: Optional[str] = None) -> str:
        """
        Create a heatmap showing the health of different components over time.
        
        Args:
            component_health: Dictionary mapping component names to dictionaries 
                             of timestamp -> health score
            title: Plot title
            filename: Optional filename to save the plot
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            # Extract unique timestamps and components
            all_timestamps = set()
            for component, health_data in component_health.items():
                all_timestamps.update(health_data.keys())
            
            components = list(component_health.keys())
            timestamps = sorted(all_timestamps)
            
            # Convert timestamps to datetime strings
            time_labels = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') 
                          for ts in timestamps]
            
            # Create data matrix
            data = np.zeros((len(components), len(timestamps)))
            
            # Fill matrix with health scores
            for i, component in enumerate(components):
                for j, ts in enumerate(timestamps):
                    if ts in component_health[component]:
                        data[i, j] = component_health[component][ts]
                    else:
                        # Use NaN for missing data
                        data[i, j] = np.nan
            
            # Create heatmap
            im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Health Score", rotation=-90, va="bottom", fontsize=10)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Component", fontsize=12)
            
            # Set tick labels
            ax.set_yticks(np.arange(len(components)))
            ax.set_yticklabels(components, fontsize=10)
            
            # Set x-tick labels (show subset to avoid overcrowding)
            num_times = len(timestamps)
            tick_indices = np.linspace(0, num_times-1, min(10, num_times)).astype(int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([time_labels[i] for i in tick_indices], 
                              rotation=45, ha="right", fontsize=8)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"health_heatmap_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Created component health heatmap: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating component health heatmap: {str(e)}")
            return ""
    
    def error_trend_analysis(self, 
                           error_history: Dict[str, List[Tuple[float, int]]],
                           title: str = "Error Trend Analysis",
                           filename: Optional[str] = None) -> str:
        """
        Create a trend analysis chart for errors over time.
        
        Args:
            error_history: Dictionary mapping error types to lists of (timestamp, count) tuples
            title: Plot title
            filename: Optional filename to save the plot
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            # Plot each error type
            for i, (error_type, history) in enumerate(error_history.items()):
                # Extract timestamps and counts
                timestamps, counts = zip(*history) if history else ([], [])
                
                if timestamps:
                    # Convert to datetime
                    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
                    
                    color = self.color_schemes["default"](i % 10)
                    ax.plot(dates, counts, 
                           label=error_type, marker='o', markersize=4, 
                           linestyle='-', linewidth=2, color=color)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Error Count", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Use integer y-axis
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Add legend
            ax.legend(loc='best', fontsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"error_trend_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Created error trend analysis: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating error trend analysis: {str(e)}")
            return ""
    
    def correlation_matrix(self, 
                         data: List[Dict[str, Any]],
                         metrics: List[str],
                         title: str = "Metric Correlation Matrix",
                         filename: Optional[str] = None) -> str:
        """
        Create a correlation matrix heatmap for multiple metrics.
        
        Args:
            data: List of benchmark data points
            metrics: List of metrics to include in the correlation
            title: Plot title
            filename: Optional filename to save the plot
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Extract values for each metric
            metric_values = {}
            for metric in metrics:
                values = []
                for point in data:
                    if metric in point and isinstance(point[metric], (int, float)):
                        values.append(point[metric])
                if values:
                    metric_values[metric] = values
            
            # Need at least 2 metrics with data
            if len(metric_values) < 2:
                logger.warning("Not enough data for correlation matrix")
                return ""
            
            # Ensure all metrics have the same number of values
            min_length = min(len(values) for values in metric_values.values())
            for metric in metric_values:
                metric_values[metric] = metric_values[metric][:min_length]
            
            # Convert to numpy array
            data_array = np.array([metric_values[metric] for metric in metric_values]).T
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data_array.T)
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Correlation Coefficient", rotation=-90, va="bottom", fontsize=10)
            
            # Set title
            ax.set_title(title, fontsize=16)
            
            # Set tick labels
            metrics_list = list(metric_values.keys())
            ax.set_xticks(np.arange(len(metrics_list)))
            ax.set_yticks(np.arange(len(metrics_list)))
            ax.set_xticklabels(metrics_list, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(metrics_list, fontsize=10)
            
            # Add correlation values in cells
            for i in range(len(metrics_list)):
                for j in range(len(metrics_list)):
                    text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                                  ha="center", va="center", 
                                  color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                                  fontsize=9)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"correlation_matrix_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Created correlation matrix: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            return ""
    
    def anomaly_detection_plot(self, 
                             data: List[Dict[str, Any]],
                             metric: str,
                             window_size: int = 10,
                             threshold: float = 2.0,
                             title: Optional[str] = None,
                             filename: Optional[str] = None) -> str:
        """
        Create a plot highlighting anomalies in a metric time series.
        
        Args:
            data: List of benchmark data points
            metric: Metric to analyze for anomalies
            window_size: Size of the rolling window for z-score calculation
            threshold: Z-score threshold for anomaly detection
            title: Optional plot title
            filename: Optional filename to save the plot
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Extract values and timestamps
            values = []
            timestamps = []
            
            for point in data:
                if metric in point and isinstance(point[metric], (int, float)):
                    values.append(point[metric])
                    timestamps.append(point["timestamp"])
            
            if len(values) < window_size + 1:
                logger.warning(f"Not enough data for anomaly detection (need at least {window_size+1} points)")
                return ""
            
            # Convert values to numpy array
            values_array = np.array(values)
            
            # Calculate rolling mean and std
            anomalies = []
            anomaly_timestamps = []
            
            for i in range(window_size, len(values)):
                window = values_array[i-window_size:i]
                mean = np.mean(window)
                std = np.std(window)
                
                # Avoid division by zero
                if std == 0:
                    std = 1e-6
                
                # Calculate z-score
                z_score = (values[i] - mean) / std
                
                # Check if it's an anomaly
                if abs(z_score) > threshold:
                    anomalies.append(values[i])
                    anomaly_timestamps.append(timestamps[i])
            
            # Convert timestamps to datetime
            dt_timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
            dt_anomaly_timestamps = [datetime.datetime.fromtimestamp(ts) for ts in anomaly_timestamps]
            
            # Generate title if not provided
            if title is None:
                title = f"Anomaly Detection for {metric}"
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            # Plot the time series
            ax.plot(dt_timestamps, values, 
                   label=metric, marker='.', markersize=4, 
                   linestyle='-', linewidth=1, color='blue')
            
            # Plot anomalies
            if anomalies:
                ax.scatter(dt_anomaly_timestamps, anomalies, 
                          color='red', s=100, label='Anomalies', zorder=5)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Add legend
            ax.legend(loc='best', fontsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"anomaly_detection_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close(fig)
            
            logger.info(f"Created anomaly detection plot: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating anomaly detection plot: {str(e)}")
            return ""
    
    def generate_dashboard(self, 
                         data: Dict[str, Any],
                         title: str = "Performance Dashboard",
                         filename: Optional[str] = None) -> str:
        """
        Generate a comprehensive dashboard with multiple visualizations.
        
        Args:
            data: Dictionary containing various datasets for visualization
            title: Dashboard title
            filename: Optional filename to save the dashboard
            
        Returns:
            Path to the saved dashboard
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12), dpi=self.default_dpi)
            
            # Set up grid layout
            gs = fig.add_gridspec(3, 3)
            
            # Add title
            fig.suptitle(title, fontsize=20, y=0.98)
            
            # Time series subplot
            if "time_series_data" in data and "time_series_metrics" in data:
                ax1 = fig.add_subplot(gs[0, :])
                
                ts_data = data["time_series_data"]
                ts_metrics = data["time_series_metrics"]
                
                # Extract timestamps and convert to datetime
                timestamps = [datetime.datetime.fromtimestamp(point["timestamp"]) 
                             for point in ts_data]
                
                # Plot each metric
                for i, metric in enumerate(ts_metrics):
                    values = []
                    valid_timestamps = []
                    
                    # Extract values, handling missing data
                    for j, point in enumerate(ts_data):
                        if metric in point:
                            values.append(point[metric])
                            valid_timestamps.append(timestamps[j])
                    
                    if values:
                        color = self.color_schemes["default"](i % 10)
                        ax1.plot(valid_timestamps, values, 
                               label=metric, marker='.', markersize=3, 
                               linestyle='-', linewidth=1.5, color=color)
                
                # Set labels
                ax1.set_title("Performance Metrics Over Time", fontsize=14)
                ax1.set_xlabel("Time", fontsize=10)
                ax1.set_ylabel("Value", fontsize=10)
                
                # Format x-axis as dates
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add legend
                ax1.legend(loc='upper right', fontsize=8)
                
                # Add grid
                ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Error distribution subplot
            if "error_distribution" in data:
                ax2 = fig.add_subplot(gs[1, 0])
                
                error_data = data["error_distribution"]
                
                # Extract data
                error_types = list(error_data.keys())
                counts = list(error_data.values())
                
                # Create pie chart
                wedges, texts, autotexts = ax2.pie(
                    counts, 
                    labels=error_types, 
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=False
                )
                
                # Style the text
                for text in texts:
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax2.axis('equal')
                
                # Set title
                ax2.set_title("Error Distribution", fontsize=14)
            
            # Component health subplot
            if "component_health" in data:
                ax3 = fig.add_subplot(gs[1, 1:])
                
                health_data = data["component_health"]
                components = list(health_data.keys())
                health_scores = [health_data[comp] for comp in components]
                
                # Create horizontal bar chart
                bars = ax3.barh(components, health_scores, color=plt.cm.RdYlGn(np.array(health_scores)/100))
                
                # Add score labels
                for bar in bars:
                    width = bar.get_width()
                    ax3.text(width + 1, bar.get_y() + bar.get_height()/2,
                           f'{width:.1f}', ha='left', va='center', fontsize=8)
                
                # Set limits
                ax3.set_xlim(0, 100)
                
                # Set title and labels
                ax3.set_title("Component Health Scores", fontsize=14)
                ax3.set_xlabel("Health Score", fontsize=10)
                
                # Add color guide
                sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 100))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax3, orientation='horizontal', pad=0.2)
                cbar.set_label("Health Score", fontsize=8)
            
            # Error trend subplot
            if "error_trend" in data:
                ax4 = fig.add_subplot(gs[2, :2])
                
                error_trend = data["error_trend"]
                
                # Plot each error type
                for i, (error_type, history) in enumerate(error_trend.items()):
                    # Extract timestamps and counts
                    timestamps, counts = zip(*history) if history else ([], [])
                    
                    if timestamps:
                        # Convert to datetime
                        dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
                        
                        color = self.color_schemes["default"](i % 10)
                        ax4.plot(dates, counts, 
                               label=error_type, marker='.', markersize=3, 
                               linestyle='-', linewidth=1.5, color=color)
                
                # Set title and labels
                ax4.set_title("Error Trend Analysis", fontsize=14)
                ax4.set_xlabel("Time", fontsize=10)
                ax4.set_ylabel("Error Count", fontsize=10)
                
                # Format x-axis as dates
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Use integer y-axis
                ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
                
                # Add legend
                ax4.legend(loc='upper right', fontsize=8)
                
                # Add grid
                ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Anomaly subplot
            if "anomaly_data" in data and "anomaly_metric" in data:
                ax5 = fig.add_subplot(gs[2, 2])
                
                anomaly_data = data["anomaly_data"]
                metric = data["anomaly_metric"]
                
                # Extract values and check for anomalies
                values = []
                timestamps = []
                anomalies = []
                anomaly_indices = []
                
                # Extract data points
                for i, point in enumerate(anomaly_data):
                    if metric in point:
                        timestamps.append(point["timestamp"])
                        values.append(point[metric])
                        
                        # Check if this point is marked as an anomaly
                        if "anomalies" in point and metric in point["anomalies"]:
                            anomalies.append(point[metric])
                            anomaly_indices.append(i)
                
                if values:
                    # Convert timestamps to datetime
                    dt_timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
                    
                    # Plot the time series
                    ax5.plot(dt_timestamps, values, 
                           label=metric, marker='.', markersize=2, 
                           linestyle='-', linewidth=1, color='blue')
                    
                    # Plot anomalies if any
                    if anomalies:
                        anomaly_times = [dt_timestamps[i] for i in anomaly_indices]
                        ax5.scatter(anomaly_times, anomalies, 
                                  color='red', s=50, label='Anomalies', zorder=5)
                    
                    # Set title and labels
                    ax5.set_title("Anomaly Detection", fontsize=14)
                    ax5.set_xlabel("Time", fontsize=10)
                    ax5.set_ylabel(metric, fontsize=10)
                    
                    # Format x-axis as dates
                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    # Add legend
                    if anomalies:
                        ax5.legend(loc='upper right', fontsize=8)
                    
                    # Add grid
                    ax5.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
            
            # Save figure
            if filename is None:
                timestamp = int(time.time())
                filename = f"dashboard_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.default_dpi)
            plt.close(fig)
            
            logger.info(f"Created dashboard: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return ""
            
    def interactive_html_export(self, 
                              data: Dict[str, Any],
                              title: str = "Interactive Performance Dashboard",
                              filename: Optional[str] = None) -> str:
        """
        Generate an interactive HTML dashboard with JavaScript visualizations.
        
        This method uses Plotly to create interactive visualizations and exports
        them as a standalone HTML file.
        
        Args:
            data: Dictionary containing various datasets for visualization
            title: Dashboard title
            filename: Optional filename to save the HTML file
            
        Returns:
            Path to the saved HTML file
        """
        try:
            # Import plotly here to make it optional
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            
            # Create a subplot figure
            fig = make_subplots(
                rows=3, cols=2,
                specs=[
                    [{"colspan": 2}, None],
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"colspan": 2}, None]
                ],
                subplot_titles=(
                    "Performance Metrics Over Time",
                    "Error Distribution",
                    "Component Health",
                    "Error Trend Analysis"
                )
            )
            
            # Add time series
            if "time_series_data" in data and "time_series_metrics" in data:
                ts_data = data["time_series_data"]
                ts_metrics = data["time_series_metrics"]
                
                # Extract timestamps
                timestamps = [datetime.datetime.fromtimestamp(point["timestamp"]) 
                             for point in ts_data]
                
                # Plot each metric
                for i, metric in enumerate(ts_metrics):
                    values = []
                    valid_timestamps = []
                    
                    # Extract values, handling missing data
                    for j, point in enumerate(ts_data):
                        if metric in point:
                            values.append(point[metric])
                            valid_timestamps.append(timestamps[j])
                    
                    if values:
                        fig.add_trace(
                            go.Scatter(
                                x=valid_timestamps,
                                y=values,
                                mode='lines+markers',
                                name=metric
                            ),
                            row=1, col=1
                        )
            
            # Add error distribution pie chart
            if "error_distribution" in data:
                error_data = data["error_distribution"]
                error_types = list(error_data.keys())
                counts = list(error_data.values())
                
                fig.add_trace(
                    go.Pie(
                        labels=error_types,
                        values=counts,
                        textinfo='percent+label'
                    ),
                    row=2, col=1
                )
            
            # Add component health bar chart
            if "component_health" in data:
                health_data = data["component_health"]
                components = list(health_data.keys())
                health_scores = [health_data[comp] for comp in components]
                
                # Create color scale
                colors = [f'rgb({int(255*(1-score/100))},{int(255*score/100)},0)' 
                         for score in health_scores]
                
                fig.add_trace(
                    go.Bar(
                        x=health_scores,
                        y=components,
                        orientation='h',
                        marker=dict(color=colors)
                    ),
                    row=2, col=2
                )
            
            # Add error trend
            if "error_trend" in data:
                error_trend = data["error_trend"]
                
                # Plot each error type
                for error_type, history in error_trend.items():
                    if history:
                        # Extract timestamps and counts
                        timestamps, counts = zip(*history)
                        
                        # Convert to datetime
                        dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=counts,
                                mode='lines+markers',
                                name=error_type
                            ),
                            row=3, col=1
                        )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=1000,
                width=1200,
                showlegend=True
            )
            
            # Save as HTML
            if filename is None:
                timestamp = int(time.time())
                filename = f"interactive_dashboard_{timestamp}.html"
            
            filepath = os.path.join(self.output_dir, filename)
            pio.write_html(fig, file=filepath, auto_open=False)
            
            logger.info(f"Created interactive HTML dashboard: {filepath}")
            return filepath
            
        except ImportError:
            logger.error("Plotly is required for interactive HTML export. Install with 'pip install plotly'")
            return ""
        except Exception as e:
            logger.error(f"Error creating interactive HTML dashboard: {str(e)}")
            return ""
