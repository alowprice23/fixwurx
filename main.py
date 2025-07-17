#!/usr/bin/env python3
"""
main.py
───────
**Triangulum container entry-point**

1.  Parse `config/system_config.yaml`
2.  Build an initial **backlog** (bug-queue) either from
    *   `--backlog-file <json>` (CLI) **or**
    *   synthetic demo bugs `[bug-0 … bug-9]`.
3.  Wire up **ResourceManager → Scheduler → ParallelExecutor**.
4.  Start the **async event-loop**:
       • every 100 ms  ⇒ ParallelExecutor.tick()
       • every tick    ⇒ SystemMonitor.emit_tick()
5.  Graceful SIGINT/SIGTERM shutdown.

The file deliberately stays under 150 lines to remain auditable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ─── Core modules ─────────────────────────────────────────────────────────────
from triangulation_engine import TriangulationEngine, MetricBus
from triangulum_resource_manager import ResourceManager
from scheduler import Scheduler
from parallel_executor import ParallelExecutor
from data_structures import BugState, FamilyTree
from system_monitor import SystemMonitor
from llm_integrations import LLMManager
from credential_manager import CredentialManager
from resource_allocation_optimizer import ResourceAllocationOptimizer

# ---------------------------------------------------------------------------—
# MetricBus (stdout demo)
# ---------------------------------------------------------------------------—
class StdoutBus(MetricBus):
    def send(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:  # noqa: D401
        print(f"[metric] {name} {value} {tags}", file=sys.stderr)


# ---------------------------------------------------------------------------—
# LLM setup helpers
# ---------------------------------------------------------------------------—
def setup_llm_providers(config: Dict[str, Any]) -> Optional[LLMManager]:
    """Configure LLM API keys and return a manager."""
    # Initialize credential manager
    cred_manager = CredentialManager()
    
    # Try to load the API key from the text file as a fallback
    try:
        with open("oPEN AI API KEY.txt", "r") as f:
            api_key = f.read().strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("Using API key from file")
    except Exception as e:
        print(f"Warning: Could not load API key from file: {e}")
    
    # First, try to load keys from environment
    env_results = cred_manager.load_all_from_env()
    
    # Determine the preferred provider from config
    preferred = config.get("llm", {}).get("preferred", "openai")
    
    # Only try to load the preferred provider's key if not already loaded
    if preferred not in env_results or not env_results[preferred]:
        key = cred_manager.get_api_key(preferred)
        if key:
            print(f"Successfully loaded {preferred.upper()} API key.")
        else:
            print(f"Failed to load {preferred.upper()} API key.")
    
    # Initialize LLM manager with the configured keys
    try:
        # Get preferred provider from config
        preferred = config.get("llm", {}).get("preferred", "openai")
        
        # Pass preferred provider to LLMManager to initialize only that provider
        llm_manager = LLMManager(
            credential_manager=cred_manager,
            config={"llm": {"preferred": preferred}}
        )
        
        available_providers = llm_manager.available()
        
        if not available_providers:
            print("Warning: No LLM providers available. Check API keys.", file=sys.stderr)
            return None
            
        print(f"Available LLM providers: {available_providers}")
        
        return llm_manager
    except Exception as e:
        print(f"Error initializing LLM manager: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------—
# LLM usage demo
# ---------------------------------------------------------------------------—
def demonstrate_llm_usage(llm_manager: Optional[LLMManager]) -> None:
    """Demonstrate LLM usage with a simple prompt."""
    if not llm_manager:
        print("LLM manager not available - skipping demo.", file=sys.stderr)
        return
    
    try:
        print("\n=== LLM DEMO ===")
        
        # Use the preferred provider
        response = llm_manager.chat(
            role="user",
            content="Explain entropy in 3 sentences.",
            task_type="explain",
            complexity="low"
        )
        
        if response:
            print(f"Using {response.provider} provider:")
            print(f"Response: {response.text}")
            print(f"Tokens: {response.tokens}, Cost: ${response.cost_usd:.6f}")
        else:
            print("Failed to get a response from any provider.")
        
        print("\n=== END LLM DEMO ===\n")
    except Exception as e:
        print(f"Error during LLM demo: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------—
# Backlog helpers
# ---------------------------------------------------------------------------—
def _load_backlog(file: Path) -> List[BugState]:
    with file.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return [BugState.from_json(b) if isinstance(b, dict) else b for b in raw]


def _demo_backlog(n: int) -> List[BugState]:
    return [BugState(bug_id=f"demo-{i}") for i in range(n)]


# ---------------------------------------------------------------------------—
# Async driver
# ---------------------------------------------------------------------------—
async def run_loop(par_exec: ParallelExecutor, monitor: SystemMonitor, interval_ms: int):
    try:
        while True:
            await par_exec.tick_all()
            monitor.emit_tick()
            await asyncio.sleep(interval_ms / 1000)
    except asyncio.CancelledError:
        print("[main] cancel received – shutting down‥", file=sys.stderr)


# ---------------------------------------------------------------------------—
# CLI
# ---------------------------------------------------------------------------—
def parse_args():
    p = argparse.ArgumentParser(description="Triangulum entry-point")
    p.add_argument("--config", default="system_config.yaml")
    p.add_argument("--backlog-file", help="JSON list of bug dicts")
    p.add_argument("--demo-bugs", type=int, default=10, help="generate demo backlog")
    p.add_argument("--tick-ms", type=int, help="override tick duration")
    p.add_argument("--skip-llm", action="store_true", help="skip LLM demo")
    return p.parse_args()


# ---------------------------------------------------------------------------—
# entry
# ---------------------------------------------------------------------------—
def main():
    ns = parse_args()
    
    print("Starting FixWurx Triangulation Engine...")
    
    # Load configuration
    cfg_path = Path(ns.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    
    # Set up LLM providers
    llm_manager = setup_llm_providers(cfg)
    
    # Run LLM demo if requested
    if not ns.skip_llm and llm_manager:
        demonstrate_llm_usage(llm_manager)
    
    # ── backlog
    if ns.backlog_file:
        backlog = _load_backlog(Path(ns.backlog_file))
    else:
        backlog = _demo_backlog(ns.demo_bugs)

    # ── core plumbing
    res_mgr = ResourceManager(total_agents=cfg["agents"]["total"])
    engine = TriangulationEngine(resource_manager=res_mgr, config=cfg)
    sched = Scheduler(engine, backlog)
    par_exec = ParallelExecutor(scheduler=sched, resource_manager=res_mgr)

    # ── monitoring
    bus = StdoutBus()
    monitor = SystemMonitor(engine, metric_bus=bus, env="prod")
    
    # ── resource optimization (for horizontal scaling)
    scaling_enabled = cfg.get("scaling", {}).get("enabled", False)
    cluster_mgr = None  # Only initialized when using horizontal scaling
    optimizer = None
    
    if scaling_enabled:
        # Import cluster-related components only if scaling is enabled
        try:
            from resource_manager_extension import ClusterResourceManager
            from scaling_coordinator import ScalingCoordinator
            from advanced_load_balancer import AdvancedLoadBalancer, BalancingStrategy
            
            # Set up scaling components
            print("Horizontal scaling enabled - initializing components...")
            
            # Initialize scaling coordinator
            scaling_coordinator = ScalingCoordinator(
                config=cfg.get("scaling", {})
            )
            
            # Get load balancing configuration
            lb_config = cfg.get("load_balancer", {})
            
            # Parse strategy from config
            strategy_name = lb_config.get("strategy", "weighted_capacity")
            strategy_map = {
                "round_robin": BalancingStrategy.ROUND_ROBIN,
                "least_connections": BalancingStrategy.LEAST_CONNECTIONS,
                "weighted_capacity": BalancingStrategy.WEIGHTED_CAPACITY,
                "random": BalancingStrategy.RANDOM
            }
            strategy = strategy_map.get(strategy_name, BalancingStrategy.WEIGHTED_CAPACITY)
            
            # Initialize advanced load balancer
            load_balancer = AdvancedLoadBalancer(
                strategy=strategy,
                scaling_coordinator=scaling_coordinator,
                health_check_interval_sec=lb_config.get("health_check_interval_sec", 30),
                config=lb_config
            )
            
            # Wrap resource manager with cluster-aware version
            cluster_mgr = ClusterResourceManager(
                base_resource_manager=res_mgr,
                scaling_coordinator=scaling_coordinator,
                load_balancer=load_balancer
            )
            
            # Initialize resource allocation optimizer
            optimizer = ResourceAllocationOptimizer(
                cluster_resource_manager=cluster_mgr,
                scaling_coordinator=scaling_coordinator,
                optimization_interval_sec=cfg.get("resource_optimization", {}).get("interval_sec", 30),
                usage_history_size=cfg.get("resource_optimization", {}).get("history_size", 60),
                burst_capacity_factor=cfg.get("resource_optimization", {}).get("burst_factor", 1.2)
            )
            
            # Start components
            cluster_mgr.start_sync()
            scaling_coordinator.start()
            load_balancer.start()
            optimizer.start()
            
            print("Resource allocation optimizer started")
            
        except ImportError as e:
            print(f"Warning: Horizontal scaling requested but components not available: {e}")
            scaling_enabled = False

    # ── async loop
    interval = ns.tick_ms or cfg["metrics"]["tick-interval-ms"]
    
    # Create output directory
    os.makedirs(".triangulum", exist_ok=True)
    
    # Use asyncio.run() which handles event loop creation/cleanup automatically
    try:
        print("Starting engine loop...")
        asyncio.run(run_loop(par_exec, monitor, interval))
    except KeyboardInterrupt:
        print("[main] interrupt received – shutting down‥", file=sys.stderr)
    finally:
        # Clean up resources
        if scaling_enabled:
            print("Shutting down scaling components...")
            if optimizer:
                optimizer.stop()
                print("Resource allocation optimizer stopped")
            
            if cluster_mgr:
                cluster_mgr.stop_sync()
                print("Cluster resource manager stopped")
            
            try:
                if 'scaling_coordinator' in locals() and scaling_coordinator:
                    scaling_coordinator.stop()
                    print("Scaling coordinator stopped")
                
                if 'load_balancer' in locals() and load_balancer:
                    load_balancer.stop()
                    print("Load balancer stopped")
            except Exception as e:
                print(f"Error during scaling component shutdown: {e}")
        
        print("[main] shutdown complete", file=sys.stderr)


if __name__ == "__main__":
    main()
