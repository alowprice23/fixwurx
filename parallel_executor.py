"""
core/parallel_executor.py
────────────────────────
Implements multi-bug parallelism by spawning one TriangulationEngine per live bug.

High-level contract
───────────────────
1. Acts as a facade over the per-bug engines, exposing a single `tick_all()` method.
2. Coordinates with the ResourceManager to ensure proper agent allocation.
3. Ensures deterministic ordering of bug processing to maintain consistent behavior.

No third-party dependencies—pure Python async/await for concurrency.
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any

class ParallelExecutor:
    """
    Executes multiple bugs in parallel, each with its own TriangulationEngine instance.
    
    This class manages concurrent processing of bugs, ensuring proper resource allocation
    and deterministic execution order.
    """
    
    def __init__(self, scheduler, resource_manager):
        """
        Initialize the ParallelExecutor.
        
        Args:
            scheduler: The scheduler responsible for bug prioritization and selection
            resource_manager: The resource manager for agent allocation
        """
        self.scheduler = scheduler
        self.resource_manager = resource_manager
        self.active_engines = {}  # bug_id -> engine mapping
    
    async def tick_all(self) -> None:
        """
        Execute one tick for all active bugs in parallel.
        
        This method:
        1. Gets the next set of bugs from the scheduler
        2. Creates engines for new bugs
        3. Executes a tick on all active engines
        4. Cleans up completed bugs
        """
        # Get next batch of bugs from scheduler
        active_bugs = await self.scheduler.get_active_bugs()
        
        # Create engines for new bugs
        for bug in active_bugs:
            if bug.bug_id not in self.active_engines:
                # Allocate resources for this bug
                if self.resource_manager.can_allocate():
                    self.resource_manager.allocate(bug.bug_id)
                    engine = self.scheduler.create_engine_for_bug(bug)
                    self.active_engines[bug.bug_id] = engine
        
        # Execute ticks in parallel
        tick_tasks = []
        for bug_id, engine in self.active_engines.items():
            tick_tasks.append(self._execute_engine_tick(engine))
        
        if tick_tasks:
            await asyncio.gather(*tick_tasks)
        
        # Clean up completed bugs
        completed_bugs = []
        for bug_id, engine in self.active_engines.items():
            if engine.all_done():
                completed_bugs.append(bug_id)
        
        for bug_id in completed_bugs:
            self.resource_manager.free_agents(bug_id)
            del self.active_engines[bug_id]
            await self.scheduler.mark_bug_complete(bug_id)
    
    async def _execute_engine_tick(self, engine):
        """Execute a single tick on the given engine."""
        # Execute the tick (synchronously)
        engine.execute_tick()
        
        # Return control to event loop to allow other tasks to run
        await asyncio.sleep(0)
        
        return engine
