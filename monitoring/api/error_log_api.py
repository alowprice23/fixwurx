"""
monitoring/api/error_log_api.py
─────────────────────────────────
REST API endpoints for advanced error log analysis and visualization.

This module provides:
1. Error trend analysis endpoints
2. Severity distribution data
3. Component-based error grouping
4. Pattern detection in error logs
5. Comprehensive error summaries
6. Export capabilities for various formats

Usage with FastAPI:
    ```python
    from fastapi import FastAPI
    from monitoring.api.error_log_api import router as error_log_router
    
    app = FastAPI()
    app.include_router(error_log_router, prefix="/api/error-logs")
    ```
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel

from monitoring.error_log import ErrorLog, ErrorSeverity
from monitoring.error_visualizer import ErrorVisualizer


# Initialize router
router = APIRouter(prefix="/error-analysis", tags=["Error Analysis"])


# Data models for API
class ErrorTrendParams(BaseModel):
    """Parameters for error trend analysis."""
    days: int = 7
    group_by: str = "day"
    min_severity: Optional[str] = None
    component: Optional[str] = None


class ErrorDistributionParams(BaseModel):
    """Parameters for error distribution analysis."""
    days: Optional[int] = None
    component: Optional[str] = None
    min_severity: Optional[str] = None


class ErrorPatternParams(BaseModel):
    """Parameters for error pattern detection."""
    days: Optional[int] = None
    min_severity: Optional[str] = None
    min_occurrences: int = 2


class ErrorSummaryParams(BaseModel):
    """Parameters for error summary generation."""
    days: int = 7
    min_severity: str = "WARNING"


class ExportParams(BaseModel):
    """Parameters for error log export."""
    format: str = "html"
    days: Optional[int] = None
    min_severity: Optional[str] = None
    include_summary: bool = True


# Helper function to get error log
def get_error_log() -> ErrorLog:
    """
    Get the global error log instance.
    
    Returns:
        ErrorLog instance from monitoring.error_log
    """
    try:
        # Try to import the global instance from error_log module
        from monitoring.error_log import global_error_log
        return global_error_log
    except (ImportError, AttributeError):
        # If that fails, create a new one
        from monitoring.error_log import ErrorLog
        return ErrorLog()


# Helper function to get visualizer
def get_visualizer() -> ErrorVisualizer:
    """
    Get an ErrorVisualizer instance.
    
    Returns:
        ErrorVisualizer instance
    """
    error_log = get_error_log()
    return ErrorVisualizer(error_log)


# API Endpoints
@router.get("/trends")
async def get_error_trends(
    days: int = 7,
    group_by: str = "day",
    min_severity: Optional[str] = None,
    component: Optional[str] = None
) -> Dict[str, List]:
    """
    Get error trends over time.
    
    Args:
        days: Number of days to analyze
        group_by: Grouping interval ('hour', 'day', 'week')
        min_severity: Minimum severity to include
        component: Filter by component
        
    Returns:
        Dictionary mapping components to time series data
    """
    visualizer = get_visualizer()
    
    try:
        # Get trends for all components or specific component
        if component:
            # For single component, we need to filter the results
            all_trends = visualizer.get_error_trends(
                days=days,
                group_by=group_by,
                min_severity=min_severity
            )
            return {k: v for k, v in all_trends.items() if k == component}
        else:
            return visualizer.get_error_trends(
                days=days,
                group_by=group_by,
                min_severity=min_severity
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing trends: {str(e)}")


@router.get("/distribution/severity")
async def get_severity_distribution(
    days: Optional[int] = None,
    component: Optional[str] = None
) -> Dict[str, int]:
    """
    Get distribution of errors by severity.
    
    Args:
        days: Optional time range in days
        component: Optional component filter
        
    Returns:
        Dictionary mapping severity levels to counts
    """
    visualizer = get_visualizer()
    
    try:
        return visualizer.get_severity_distribution(
            component=component,
            days=days
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing severity distribution: {str(e)}")


@router.get("/distribution/component")
async def get_component_distribution(
    days: Optional[int] = None,
    min_severity: Optional[str] = None
) -> Dict[str, int]:
    """
    Get distribution of errors by component.
    
    Args:
        days: Optional time range in days
        min_severity: Optional minimum severity filter
        
    Returns:
        Dictionary mapping components to counts
    """
    visualizer = get_visualizer()
    
    try:
        return visualizer.get_component_distribution(
            min_severity=min_severity,
            days=days
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing component distribution: {str(e)}")


@router.get("/patterns")
async def get_error_patterns(
    days: Optional[int] = None,
    min_severity: Optional[str] = None,
    min_occurrences: int = 2
) -> List[Dict[str, Any]]:
    """
    Detect patterns in error messages.
    
    Args:
        days: Optional time range in days
        min_severity: Optional minimum severity filter
        min_occurrences: Minimum number of occurrences to consider a pattern
        
    Returns:
        List of detected patterns with counts
    """
    visualizer = get_visualizer()
    
    try:
        return visualizer.get_error_patterns(
            min_severity=min_severity,
            min_occurrences=min_occurrences,
            days=days
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting patterns: {str(e)}")


@router.get("/summary")
async def get_error_summary(
    days: int = 7,
    min_severity: str = "WARNING"
) -> Dict[str, Any]:
    """
    Get a comprehensive error summary.
    
    Args:
        days: Number of days to analyze
        min_severity: Minimum severity to include
        
    Returns:
        Dictionary with summary information
    """
    visualizer = get_visualizer()
    
    try:
        return visualizer.get_error_summary(
            days=days,
            min_severity=min_severity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@router.get("/export")
async def export_logs(
    background_tasks: BackgroundTasks,
    format: str = "html",
    days: Optional[int] = None,
    min_severity: Optional[str] = None,
    include_summary: bool = True
) -> Dict[str, str]:
    """
    Export error logs to various formats.
    
    Args:
        format: Export format ('html' or 'csv')
        days: Optional time range in days
        min_severity: Minimum severity to include
        include_summary: Whether to include summary in HTML
        
    Returns:
        Dictionary with download link
    """
    visualizer = get_visualizer()
    
    try:
        # Create export directory if it doesn't exist
        export_dir = Path(".triangulum/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp and filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "html":
            filename = f"error_log_export_{timestamp}.html"
            filepath = export_dir / filename
            
            # Export in background to avoid blocking
            def export_html():
                visualizer.export_to_html(
                    str(filepath),
                    days=days,
                    min_severity=min_severity,
                    include_summary=include_summary
                )
            
            background_tasks.add_task(export_html)
            
        elif format.lower() == "csv":
            filename = f"error_log_export_{timestamp}.csv"
            filepath = export_dir / filename
            
            # Export in background to avoid blocking
            def export_csv():
                visualizer.export_to_csv(
                    str(filepath),
                    days=days
                )
            
            background_tasks.add_task(export_csv)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Return download link
        return {
            "status": "success",
            "message": f"Export started. The file will be available shortly.",
            "filename": filename,
            "download_link": f"/api/error-logs/download/{filename}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting logs: {str(e)}")


@router.get("/download/{filename}")
async def download_export(filename: str) -> Response:
    """
    Download an exported file.
    
    Args:
        filename: Name of the exported file
        
    Returns:
        File as response
    """
    # Check if the file exists
    export_dir = Path(".triangulum/exports")
    filepath = export_dir / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    # Determine content type
    if filename.endswith(".html"):
        media_type = "text/html"
    elif filename.endswith(".csv"):
        media_type = "text/csv"
    else:
        media_type = "application/octet-stream"
    
    # Return file content
    content = filepath.read_text()
    return Response(content=content, media_type=media_type)


# Add the router to the dashboard API
def register_with_dashboard():
    """Register the error log API with the dashboard."""
    try:
        from monitoring.dashboard import api_router as dashboard_router
        dashboard_router.include_router(router, prefix="/error-logs")
        print("Error log API registered with dashboard")
    except ImportError:
        print("Dashboard not available, error log API will run standalone")


# Initialize when this module is imported
register_with_dashboard()
