"""
monitoring/api/__init__.py
─────────────────────────
Package initialization for the monitoring API modules.

This package contains RESTful API endpoints for:
- Error log visualization and analysis
- Integration with the main dashboard
- Data export capabilities
"""

# Import the router for external use
from monitoring.api.error_log_api import router as error_log_router
