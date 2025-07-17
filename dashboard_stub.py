"""
monitoring/dashboard_stub.py
────────────────────────────
🚧 REDIRECTION MODULE - This stub has been replaced by the full implementation 🚧

This file now serves as a redirection wrapper to the complete dashboard implementation
in `monitoring/dashboard.py`. It provides backward compatibility for any code
still importing the stub version.

For the complete dashboard with:
• Comprehensive metrics visualization
• Agent status monitoring
• Entropy tracking with trend analysis
• Rollback management interface
• Alert system integration
• Error log visualization

Please use `monitoring.dashboard` directly:

    from monitoring.dashboard import app as dashboard_app
    import uvicorn
    
    uvicorn.run(dashboard_app, host="0.0.0.0", port=8000)

Third-party deps: **fastapi>=0.110**, **uvicorn**, **jinja2**, **chart.js**
"""

from __future__ import annotations

import logging
import importlib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dashboard_stub")

# Try to import the complete dashboard
try:
    from monitoring.dashboard import app, MetricsBus
    logger.info("Imported full dashboard implementation from monitoring.dashboard")
except ImportError as e:
    logger.warning(f"Could not import full dashboard: {e}")
    # Fall back to local implementation
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, Response, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import asyncio
    import json
    import time
    from collections import deque
    from typing import Any, Deque, Dict, List, Optional
    
    logger.warning("Using legacy dashboard_stub implementation. Consider upgrading to the full dashboard.")

    class FastAPIBus:
        """Legacy MetricBus implementation."""
        def __init__(self, maxlen: int = 256) -> None:
            self._hist: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
            self._listeners: List[asyncio.Queue] = []

        def send(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
            rec = {
                "ts": time.time(),
                "name": name,
                "value": value,
                "tags": tags or {},
            }
            self._hist.append(rec)
            for q in self._listeners:
                q.put_nowait(rec)

        def new_queue(self) -> asyncio.Queue:
            q: asyncio.Queue = asyncio.Queue(maxsize=0)
            self._listeners.append(q)
            return q

        def history_json(self) -> str:
            return json.dumps(list(self._hist))
    
    # Create legacy app
    app = FastAPI(title="Triangulum Dashboard (Legacy)")
    templates = Jinja2Templates(directory=str((Path(__file__).parent / "templates").resolve()))
    app.mount("/static", StaticFiles(directory="."), name="static")
    
    @app.on_event("startup")
    async def _init_state() -> None:
        if not hasattr(app.state, "metric_bus"):
            app.state.metric_bus = FastAPIBus()
            logger.info("Initialized legacy FastAPIBus")
            
            # Create a redirection message
            logger.warning(
                "Dashboard stub is deprecated. Please use monitoring.dashboard instead."
            )
    
    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Redirects to full dashboard if available, or shows legacy view."""
        # Check if the main dashboard exists
        full_dashboard_path = Path(__file__).parent / "dashboard.py"
        if full_dashboard_path.exists():
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Dashboard Redirect</title>
                <meta http-equiv="refresh" content="5;url=/dashboard/">
                <style>
                    body { font-family: system-ui; margin: 2rem; }
                    .alert { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; padding: 1rem; border-radius: 0.25rem; }
                    .btn { display: inline-block; padding: 0.375rem 0.75rem; background-color: #007bff; color: white; text-decoration: none; border-radius: 0.25rem; }
                </style>
            </head>
            <body>
                <h1>Dashboard Upgrade Available</h1>
                <div class="alert">
                    <p><strong>This dashboard stub is deprecated.</strong></p>
                    <p>A full-featured dashboard is now available. You will be redirected in 5 seconds.</p>
                    <p>If you are not redirected, please <a href="/dashboard/" class="btn">click here</a>.</p>
                </div>
                <p>The new dashboard includes:</p>
                <ul>
                    <li>Comprehensive metrics visualization</li>
                    <li>Agent status monitoring</li>
                    <li>Entropy tracking with trend analysis</li>
                    <li>Rollback management interface</li>
                    <li>Alert system integration</li>
                    <li>Error log visualization</li>
                </ul>
                <p>To use the new dashboard in your code:</p>
                <pre>
from monitoring.dashboard import app as dashboard_app
import uvicorn

uvicorn.run(dashboard_app, host="0.0.0.0", port=8000)
                </pre>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        else:
            return templates.TemplateResponse("dashboard.html", {"request": request})
    
    # Keep legacy endpoints
    @app.get("/events")
    async def sse_events(request: Request):
        bus = app.state.metric_bus
        queue = bus.new_queue()

        async def event_stream():
            yield f"data: {bus.history_json()}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                item = await queue.get()
                yield f"data: {json.dumps(item)}\n\n"

        return Response(event_stream(), media_type="text/event-stream")
    
    @app.get("/entropy-narrative", response_class=PlainTextResponse)
    async def entropy_story():
        bus = app.state.metric_bus
        bits = next(
            (rec["value"] for rec in reversed(bus._hist) if rec["name"].endswith("entropy_bits")),
            None,
        )
        
        if bits is None:
            return "No entropy data yet."

        return (
            f"### Entropy outlook\n"
            f"Current *H₀* ≈ **{bits:.2f} bits**  \n"
            f"- Candidate file space ≤ 2^{bits:.2f}  \n"
            f"- Expected remaining attempts *(g≈1)* ≤ {2**bits:.0f}  \n\n"
            f"System remains within deterministic bound."
        )

# Ensure template directory exists
_template_dir = Path(__file__).parent / "templates"
_template_dir.mkdir(exist_ok=True)

# Create HTML template if it doesn't exist
dashboard_template = _template_dir / "dashboard.html"
if not dashboard_template.exists():
    dashboard_template.write_text(
        """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Triangulum Stats (Legacy)</title>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <script src="https://unpkg.com/htmx.org/dist/ext/sse.js"></script>
  <style>
    body{font-family:system-ui;margin:2rem}
    pre{background:#f4f4f4;padding:0.5rem;border-radius:5px}
    .upgrade-notice{background:#ffeeba;border:1px solid #ffc107;padding:1rem;margin-bottom:1rem;border-radius:5px}
  </style>
</head>
<body>
  <div class="upgrade-notice">
    <strong>Note:</strong> This is the legacy dashboard. A full-featured dashboard is now available. 
    Please update your code to use <code>monitoring.dashboard</code> instead.
  </div>
  <h2>Triangulum Live Metrics (Legacy View)</h2>
  <p id="meta"></p>
  <pre id="log" hx-ext="sse" sse-connect="/events"
       sse-swap="message: append"></pre>

  <script>
    // simple log limiter
    const log = document.getElementById("log");
    const meta = document.getElementById("meta");
    function prune(){
      const lines = log.textContent.trim().split("\\n");
      if(lines.length>200) log.textContent = lines.slice(-200).join("\\n") + "\\n";
    }
    log.addEventListener("htmx:sseMessage", e=>{
      prune();
      try{
        const obj = JSON.parse(e.detail.data);
        if(Array.isArray(obj)){ // history
          obj.forEach(o=>log.append(o.name+" "+o.value+"\\n"));
        }else{
          log.append(obj.name+" "+obj.value+"\\n");
        }
      }catch{ log.append(e.detail.data+"\\n"); }
    });
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )

# For backwards compatibility
def main():
    """Run the dashboard server directly."""
    import uvicorn
    
    print("Starting FixWurx Dashboard (Legacy)")
    print("NOTE: This stub is deprecated. Please use monitoring.dashboard instead.")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
