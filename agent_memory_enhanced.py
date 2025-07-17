"""
agents/agent_memory_enhanced.py
──────────────────────────────
Enhanced **cross-bug knowledge base** with compressed storage and planner integration.

• Keeps a JSON file (`.triangulum/memory.json`) that stores every bug the system
  solved successfully, together with an *embedding* produced by a trivial token
  TF vectoriser.  (No external ML deps; ≈ 50 lines of math.)

• Supports compressed storage using the compress.py module for efficient token usage
  and storage of large context windows.

• Provides general-purpose key-value storage for agent relationships, planner solutions,
  and cross-session learning.

• Enhanced family tree storage with traversal and query capabilities specifically
  designed for planner agent integration.

• Advanced solution path versioning with planner-specific metadata and indexing.

• Presents capabilities:
      ▸ `add_entry(bug_id, summary, patch)`     – persist a solved bug
      ▸ `query_similar(text, k)`                – cosine-similarity lookup
      ▸ `store(key, value)`                     – general key-value storage 
      ▸ `retrieve(key)`                         – retrieve stored value
      ▸ `store_compressed(key, value)`          – store with compression
      ▸ `retrieve_compressed(key)`              – retrieve and decompress
      ▸ `store_family_tree(tree_data)`          – store agent family tree
      ▸ `get_family_tree()`                     – retrieve agent family tree
      ▸ `query_family_tree(query)`              – search the family tree
      ▸ `store_solution_path(path_id, data)`    – store versioned solution path
      ▸ `get_solution_path(path_id, revision)`  – get solution path by version

All logic fits into one file, self-contained, with minimal dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import time
import hashlib
import uuid
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

from compress import Compressor

# Compression import - handles if not available
try:
    from compress import compress_text, decompress_text
    COMPRESSION_AVAILABLE = True
except ImportError:
    # Fallback implementations if compression module not available
    def compress_text(text: str) -> str:
        return text
        
    def decompress_text(compressed: str) -> str:
        return compressed
        
    COMPRESSION_AVAILABLE = False

# Setup logging
logger = logging.getLogger("agent_memory")

# ---------------------------------------------------------------------------—
# Storage paths
# ---------------------------------------------------------------------------—
TRIANGULUM_DIR = Path(".triangulum")
MEM_PATH = TRIANGULUM_DIR / "memory.json"
KV_PATH = TRIANGULUM_DIR / "kv_store.json"
COMPRESSED_PATH = TRIANGULUM_DIR / "compressed_store.json"
FAMILY_TREE_PATH = TRIANGULUM_DIR / "family_tree.json"
FAMILY_TREE_INDEX_PATH = TRIANGULUM_DIR / "family_tree_index.json"
SOLUTION_PATHS_INDEX_PATH = TRIANGULUM_DIR / "solution_paths_index.json"
TRIANGULUM_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------—
# Helper: tokeniser + TF vector
# ---------------------------------------------------------------------------—
_DEF_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")  # ignore tiny tokens


def _tokenise(text: str) -> Counter[str]:
    tokens = _DEF_TOKEN_RE.findall(text.lower())
    return Counter(tokens)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    # dot
    dot = sum(a[t] * b.get(t, 0) for t in a)
    # norms
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------—
# Solution path versioning helper
# ---------------------------------------------------------------------------—
class SolutionPathVersion:
    """
    Represents a versioned solution path with revision history.
    
    Features:
    - Tracks all revisions of a solution path
    - Provides rollback capability
    - Stores compressed versions of large paths
    - Includes metadata for each revision
    """
    
    def __init__(self, path_id: str, compressor: Optional[Compressor] = None):
        self.path_id = path_id
        self.revisions = []  # List of revision dictionaries
        self.current_revision = -1
        self.compressor = compressor or Compressor()
    
    def add_revision(self, path_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> int:
        """
        Add a new revision of the solution path.
        
        Args:
            path_data: The solution path data
            metadata: Additional metadata about this revision
            
        Returns:
            The revision number
        """
        # Create a unique revision ID
        revision_id = str(uuid.uuid4())
        
        # Create a deep copy of the path data to ensure each revision is isolated
        # This avoids issues where changes to a mutable object would affect all revisions
        import copy
        path_data_copy = copy.deepcopy(path_data)
        
        # Ensure metadata is not None
        metadata = metadata or {}
        
        # Setup metadata
        revision_metadata = {
            "revision_id": revision_id,
            "revision_number": len(self.revisions),
            "timestamp": time.time(),
            "creator": metadata.get("creator", "unknown"),
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
            "parent_revision": self.current_revision if self.current_revision >= 0 else None
        }
        
        # Add user metadata
        if metadata:
            for key, value in metadata.items():
                if key not in revision_metadata:
                    revision_metadata[key] = value
        
        # Compress the path data if it's large
        original_size = len(json.dumps(path_data))
        should_compress = original_size > 1024  # Only compress if over 1KB
        
        if should_compress:
            # Convert to JSON string for compression
            path_json = json.dumps(path_data)
            compressed_data, bits_saved = self.compressor.compress(path_json)
            
            # Calculate checksum for integrity
            checksum = hashlib.sha256(path_json.encode()).hexdigest()
            
            # Store compression metadata
            revision_metadata["compressed"] = True
            revision_metadata["original_size"] = original_size
            revision_metadata["compressed_size"] = len(compressed_data)
            revision_metadata["compression_ratio"] = len(compressed_data) / original_size if original_size > 0 else 1.0
            revision_metadata["bits_saved"] = bits_saved
            revision_metadata["checksum"] = checksum
            
            # Create revision with compressed data
            revision = {
                "data": compressed_data,
                "metadata": revision_metadata
            }
        else:
            # Store uncompressed
            revision_metadata["compressed"] = False
            
            # Create revision with uncompressed data
            revision = {
                "data": path_data,
                "metadata": revision_metadata
            }
        
        # Add to revisions list
        self.revisions.append(revision)
        self.current_revision = len(self.revisions) - 1
        
        return self.current_revision
    
    def get_current_revision(self) -> Dict[str, Any]:
        """
        Get the current revision of the solution path.
        
        Returns:
            The path data for the current revision
        """
        if self.current_revision < 0 or self.current_revision >= len(self.revisions):
            raise ValueError("No current revision available")
            
        revision = self.revisions[self.current_revision]
        return self._get_revision_data(revision)
    
    def get_revision(self, revision_number: int) -> Dict[str, Any]:
        """
        Get a specific revision of the solution path.
        
        Args:
            revision_number: The revision number to retrieve
            
        Returns:
            The path data for the specified revision
        """
        if revision_number < 0 or revision_number >= len(self.revisions):
            raise ValueError(f"Revision {revision_number} does not exist")
            
        revision = self.revisions[revision_number]
        return self._get_revision_data(revision)
    
    def rollback(self, revision_number: int) -> Dict[str, Any]:
        """
        Rollback to a previous revision.
        
        Args:
            revision_number: The revision number to roll back to
            
        Returns:
            The path data for the rolled-back revision
        """
        if revision_number < 0 or revision_number >= len(self.revisions):
            raise ValueError(f"Cannot rollback to non-existent revision {revision_number}")
            
        self.current_revision = revision_number
        return self.get_current_revision()
    
    def get_revision_history(self) -> List[Dict[str, Any]]:
        """
        Get the revision history metadata.
        
        Returns:
            List of revision metadata (without the actual path data)
        """
        return [rev["metadata"] for rev in self.revisions]
    
    def _get_revision_data(self, revision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the data from a revision, decompressing if necessary.
        
        Args:
            revision: Revision dictionary
            
        Returns:
            The path data
        """
        if revision["metadata"].get("compressed", False):
            # Decompress the data
            compressed_data = revision["data"]
            try:
                decompressed_json = decompress_text(compressed_data) if COMPRESSION_AVAILABLE else compressed_data
                
                # Verify integrity
                checksum = hashlib.sha256(decompressed_json.encode()).hexdigest()
                if checksum != revision["metadata"]["checksum"]:
                    logger.warning(f"Checksum verification failed for revision {revision['metadata']['revision_number']}")
                
                # Parse JSON
                return json.loads(decompressed_json)
            except Exception as e:
                logger.error(f"Error decompressing revision: {e}")
                return {"error": "Failed to decompress revision", "metadata": revision["metadata"]}
        else:
            # Uncompressed data
            return revision["data"]
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the solution path version to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the solution path version
        """
        return {
            "path_id": self.path_id,
            "revisions": self.revisions,
            "current_revision": self.current_revision
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any], compressor: Optional[Compressor] = None) -> SolutionPathVersion:
        """
        Create a SolutionPathVersion from a JSON dictionary.
        
        Args:
            data: JSON dictionary
            compressor: Optional compressor to use
            
        Returns:
            New SolutionPathVersion instance
        """
        instance = cls(data["path_id"], compressor)
        instance.revisions = data["revisions"]
        instance.current_revision = data["current_revision"]
        return instance


# ---------------------------------------------------------------------------—
# Family tree traversal helper
# ---------------------------------------------------------------------------—
class FamilyTreeTraverser:
    """
    Helper class for traversing and querying the agent family tree.
    
    Features:
    - Depth-first and breadth-first traversal
    - Finding paths between agents
    - Querying by capability or role
    - Extracting subtrees
    """
    
    def __init__(self, tree_data: Dict[str, Any]):
        """
        Initialize with family tree data.
        
        Args:
            tree_data: Family tree data structure
        """
        self.tree_data = tree_data
        self._build_index()
    
    def _build_index(self) -> None:
        """Build an index of all agents in the tree for faster lookup."""
        self.agent_index = {}
        self.capability_index = defaultdict(list)
        self.role_index = defaultdict(list)
        
        # Index the root
        root_name = self.tree_data.get("root")
        if root_name:
            self.agent_index[root_name] = {
                "path": [root_name],
                "role": "root",
                "capabilities": self.tree_data.get("capabilities", []),
                "parent": None
            }
            
            # Index capabilities
            for capability in self.tree_data.get("capabilities", []):
                self.capability_index[capability].append(root_name)
            
            # Index role
            self.role_index["root"].append(root_name)
        
        # Traverse children
        self._index_children(self.tree_data.get("children", {}), [root_name] if root_name else [])
    
    def _index_children(self, children: Dict[str, Any], path: List[str]) -> None:
        """
        Recursively index all children.
        
        Args:
            children: Dictionary of child agents
            path: Current path from root
        """
        for name, data in children.items():
            # Calculate full path
            agent_path = path + [name]
            
            # Store in agent index
            parent = path[-1] if path else None
            role = data.get("role")
            capabilities = data.get("capabilities", [])
            
            self.agent_index[name] = {
                "path": agent_path,
                "role": role,
                "capabilities": capabilities,
                "parent": parent
            }
            
            # Index capabilities
            for capability in capabilities:
                self.capability_index[capability].append(name)
            
            # Index role
            if role:
                self.role_index[role].append(name)
            
            # Process children recursively
            self._index_children(data.get("children", {}), agent_path)
    
    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent.
        
        Args:
            name: Agent name
            
        Returns:
            Agent information or None if not found
        """
        return self.agent_index.get(name)
    
    def get_path(self, from_agent: str, to_agent: str) -> Optional[List[str]]:
        """
        Find the path between two agents.
        
        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            
        Returns:
            List of agent names in the path or None if no path
        """
        # If agents are the same, return empty list
        if from_agent == to_agent:
            return []
            
        from_info = self.agent_index.get(from_agent)
        to_info = self.agent_index.get(to_agent)
        
        if not from_info or not to_info:
            return None
        
        # Get paths from root
        from_path = from_info["path"]
        to_path = to_info["path"]
        
        # Find common ancestor
        common_prefix = []
        for i, (f, t) in enumerate(zip(from_path, to_path)):
            if f == t:
                common_prefix.append(f)
            else:
                break
        
        if not common_prefix:
            return None
        
        # Path is from_path up to common ancestor, then to_path from common ancestor
        from_suffix = from_path[len(common_prefix):]
        to_suffix = to_path[len(common_prefix):]
        
        return from_suffix[::-1] + common_prefix + to_suffix
    
    def find_by_capability(self, capability: str) -> List[str]:
        """
        Find all agents with a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agent names with the capability
        """
        return self.capability_index.get(capability, [])
    
    def find_by_role(self, role: str) -> List[str]:
        """
        Find all agents with a specific role.
        
        Args:
            role: Role to search for
            
        Returns:
            List of agent names with the role
        """
        return self.role_index.get(role, [])
    
    def get_subtree(self, root_agent: str) -> Dict[str, Any]:
        """
        Extract a subtree rooted at a specific agent.
        
        Args:
            root_agent: Name of the agent to use as root
            
        Returns:
            Subtree data or empty dict if agent not found
        """
        # Find the agent
        agent_info = self.agent_index.get(root_agent)
        if not agent_info:
            return {}
        
        # Find the agent in the tree
        current = self.tree_data
        for name in agent_info["path"][:-1]:  # All but the agent itself
            if name == self.tree_data.get("root"):
                current = current.get("children", {})
            else:
                current = current.get("children", {}).get(name, {}).get("children", {})
        
        # Get the subtree
        if agent_info["path"][-1] == self.tree_data.get("root"):
            # Agent is the root of the main tree
            return self.tree_data
        else:
            # Agent is somewhere in the tree
            agent_data = current.get(agent_info["path"][-1], {})
            
            # Create a new tree with this agent as root
            return {
                "root": root_agent,
                "children": agent_data.get("children", {}),
                "role": agent_data.get("role"),
                "capabilities": agent_data.get("capabilities", []),
                "original_path": agent_info["path"]
            }
    
    def get_descendants(self, agent_name: str) -> List[str]:
        """
        Get all descendants of an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of descendant agent names
        """
        # Find the agent's path
        agent_info = self.agent_index.get(agent_name)
        if not agent_info:
            return []
        
        agent_path = agent_info["path"]
        
        # Find all agents whose paths start with this agent's path
        descendants = []
        for name, info in self.agent_index.items():
            other_path = info["path"]
            if (len(other_path) > len(agent_path) and 
                other_path[:len(agent_path)] == agent_path):
                descendants.append(name)
        
        return descendants
    
    def get_ancestors(self, agent_name: str) -> List[str]:
        """
        Get all ancestors of an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of ancestor agent names
        """
        # Find the agent's path
        agent_info = self.agent_index.get(agent_name)
        if not agent_info:
            return []
        
        # The ancestors are all agents in the path except the agent itself
        return agent_info["path"][:-1]
    
    def query(self, query: Dict[str, Any]) -> List[str]:
        """
        Find agents matching a query.
        
        Args:
            query: Query dictionary with criteria
                - role: Role to match
                - capability: Capability to match
                - parent: Parent agent name
                - ancestor: Ancestor agent name
                - descendant_of: Agent whose descendants to find
                
        Returns:
            List of matching agent names
        """
        matches = set(self.agent_index.keys())
        
        # Filter by role
        if "role" in query:
            role_matches = set(self.find_by_role(query["role"]))
            matches = matches.intersection(role_matches)
        
        # Filter by capability
        if "capability" in query:
            cap_matches = set(self.find_by_capability(query["capability"]))
            matches = matches.intersection(cap_matches)
        
        # Filter by parent
        if "parent" in query:
            parent = query["parent"]
            parent_matches = {name for name, info in self.agent_index.items() 
                              if info["parent"] == parent}
            matches = matches.intersection(parent_matches)
        
        # Filter by ancestor
        if "ancestor" in query:
            ancestor = query["ancestor"]
            ancestor_matches = set()
            for name, info in self.agent_index.items():
                if ancestor in info["path"][:-1]:
                    ancestor_matches.add(name)
            matches = matches.intersection(ancestor_matches)
        
        # Filter by descendant_of
        if "descendant_of" in query:
            descendant_of = query["descendant_of"]
            descendant_matches = set(self.get_descendants(descendant_of))
            matches = matches.intersection(descendant_matches)
        
        return list(matches)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the index to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the index
        """
        return {
            "agent_index": self.agent_index,
            "capability_index": {k: v for k, v in self.capability_index.items()},
            "role_index": {k: v for k, v in self.role_index.items()}
        }
    
    @classmethod
    def from_json(cls, tree_data: Dict[str, Any], index_data: Dict[str, Any]) -> FamilyTreeTraverser:
        """
        Create a FamilyTreeTraverser from JSON data.
        
        Args:
            tree_data: Family tree data
            index_data: Index data
            
        Returns:
            New FamilyTreeTraverser instance
        """
        instance = cls(tree_data)
        instance.agent_index = index_data["agent_index"]
        
        # Convert defaultdicts
        instance.capability_index = defaultdict(list)
        for k, v in index_data["capability_index"].items():
            instance.capability_index[k] = v
        
        instance.role_index = defaultdict(list)
        for k, v in index_data["role_index"].items():
            instance.role_index[k] = v
        
        return instance


# ---------------------------------------------------------------------------—
# Agent-facing API
# ---------------------------------------------------------------------------—
class AgentMemory:
    """
    Singleton-ish: create once and share. Thread-unsafe by design, higher layers
    call from the single-threaded scheduler/event-loop.
    
    Enhanced with compression, family tree management, and general key-value storage.
    """

    def __init__(
        self, 
        mem_path: Path = MEM_PATH, 
        kv_path: Path = KV_PATH,
        compressed_path: Path = COMPRESSED_PATH,
        family_tree_path: Path = FAMILY_TREE_PATH,
        family_tree_index_path: Path = FAMILY_TREE_INDEX_PATH,
        solution_paths_index_path: Path = SOLUTION_PATHS_INDEX_PATH
    ) -> None:
        self._mem_path = mem_path
        self._kv_path = kv_path
        self._compressed_path = compressed_path
        self._family_tree_path = family_tree_path
        self._family_tree_index_path = family_tree_index_path
        self._solution_paths_index_path = solution_paths_index_path
        
        # Storage containers
        self._db: Dict[str, Dict] = {}  # Bug embeddings
        self._kv_store: Dict[str, Any] = {}  # General key-value store
        self._compressed_store: Dict[str, Dict[str, Any]] = {}  # Compressed storage with metadata
        
        # Solution paths index
        self._solution_paths_index: Dict[str, Dict[str, Any]] = {}
        
        # Family tree data and traverser
        self._family_tree: Dict[str, Any] = {}
        self._family_tree_traverser: Optional[FamilyTreeTraverser] = None
        
        # Load all data
        self._load_all()

    # .................................................. public bug storage and search
    def add_entry(self, bug_id: str, summary: str, patch: str, metadata: Dict[str, Any] = None) -> None:
        """
        Store solved bug. If bug_id already exists, update it with new information.
        
        Args:
            bug_id: Unique identifier for the bug
            summary: Text description of the bug
            patch: The solution/patch for the bug
            metadata: Optional additional data about the bug
        """
        vec = _tokenise(summary + " " + patch)
        self._db[bug_id] = {
            "summary": summary,
            "patch": patch,
            "vec": vec,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self._save_bugs()

    def query_similar(self, text: str, k: int = 5, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Return top-k most similar stored bug_ids with cosine similarity ≥ threshold.
        
        Args:
            text: Query text to find similar bugs
            k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (bug_id, similarity_score) tuples
        """
        query_vec = _tokenise(text)
        scored = [
            (bug_id, _cosine(query_vec, entry["vec"]))
            for bug_id, entry in self._db.items()
        ]
        scored = [(bid, sc) for bid, sc in scored if sc >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # .................................................. general key-value storage
    def store(self, key: str, value: Any) -> None:
        """
        Store arbitrary data in the key-value store.
        
        Args:
            key: Unique identifier for the data
            value: Data to store (must be JSON serializable)
        """
        self._kv_store[key] = {
            "value": value,
            "timestamp": time.time()
        }
        self._save_kv()
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve data from the key-value store.
        
        Args:
            key: Key to retrieve
            default: Value to return if key not found
            
        Returns:
            The stored value or default if not found
        """
        entry = self._kv_store.get(key)
        return entry["value"] if entry else default
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List all keys in the key-value store with an optional prefix filter.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of matching keys
        """
        return [k for k in self._kv_store.keys() if k.startswith(prefix)]
    
    # .................................................. compressed storage
    def store_compressed(self, key: str, value: str) -> Dict[str, Any]:
        """
        Store text data with compression.
        
        Args:
            key: Unique identifier for the data
            value: Text data to compress and store
            
        Returns:
            Metadata about the storage operation
        """
        original_size = len(value)
        compressed = compress_text(value) if COMPRESSION_AVAILABLE else value
        compressed_size = len(compressed)
        
        # Calculate a hash for integrity verification
        checksum = hashlib.sha256(value.encode()).hexdigest()
        
        metadata = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0,
            "timestamp": time.time(),
            "checksum": checksum,
            "compression_available": COMPRESSION_AVAILABLE
        }
        
        self._compressed_store[key] = {
            "data": compressed,
            "metadata": metadata
        }
        
        self._save_compressed()
        return metadata
    
    def retrieve_compressed(self, key: str, verify: bool = True) -> Optional[str]:
        """
        Retrieve and decompress stored data.
        
        Args:
            key: Key to retrieve
            verify: Whether to verify data integrity with checksum
            
        Returns:
            Decompressed data or None if not found
        """
        entry = self._compressed_store.get(key)
        if not entry:
            return None
        
        compressed = entry["data"]
        decompressed = decompress_text(compressed) if COMPRESSION_AVAILABLE else compressed
        
        # Verify integrity if requested
        if verify:
            checksum = hashlib.sha256(decompressed.encode()).hexdigest()
            if checksum != entry["metadata"]["checksum"]:
                logger.warning(f"Checksum verification failed for key '{key}'")
                return None
        
        return decompressed
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get statistics about compressed storage.
        
        Returns:
            Dictionary of compression statistics
        """
        if not self._compressed_store:
            return {
                "count": 0,
                "total_original_size": 0,
                "total_compressed_size": 0,
                "avg_compression_ratio": 0,
                "compression_available": COMPRESSION_AVAILABLE
            }
        
        total_original = sum(entry["metadata"]["original_size"] for entry in self._compressed_store.values())
        total_compressed = sum(entry["metadata"]["compressed_size"] for entry in self._compressed_store.values())
        avg_ratio = sum(entry["metadata"]["compression_ratio"] for entry in self._compressed_store.values()) / len(self._compressed_store)
        
        return {
            "count": len(self._compressed_store),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "avg_compression_ratio": avg_ratio,
            "total_space_saved": total_original - total_compressed,
            "percent_saved": (1 - (total_compressed / total_original)) * 100 if total_original > 0 else 0,
            "compression_available": COMPRESSION_AVAILABLE
        }
    
    # .................................................. family tree management
    def store_family_tree(self, tree_data: Dict[str, Any]) -> None:
        """
        Store the agent family tree with enhanced indexing.
        
        Args:
            tree_data: Family tree data structure
        """
        self._family_tree = tree_data
        
        # Create traverser
        self._family_tree_traverser = FamilyTreeTraverser(tree_data)
        
        # Store traverser index
        with self._family_tree_index_path.open("w", encoding="utf-8") as fh:
            json.dump(self._family_tree_traverser.to_json(), fh, indent=2)
        
        # Store tree
        with self._family_tree_path.open("w", encoding="utf-8") as fh:
            json.dump(tree_data, fh, indent=2)
        
        # Also store in key-value for easy access
        self.store("family_tree", tree_data)
    
    def get_family_tree(self) -> Dict[str, Any]:
        """
        Retrieve the agent family tree.
        
        Returns:
            Family tree data structure or empty dict if not found
        """
        return self._family_tree
    
    def query_family_tree(self, query: Dict[str, Any]) -> List[str]:
        """
        Query the family tree for agents matching criteria.
        
        Args:
            query: Query dictionary with criteria
                - role: Role to match
                - capability: Capability to match
                - parent: Parent agent name
                - ancestor: Ancestor agent name
                - descendant_of: Agent whose descendants to find
                
        Returns:
            List of matching agent names
        """
        if not self._family_tree_traverser:
            return []
        
        return self._family_tree_traverser.query(query)
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent information or None if not found
        """
        if not self._family_tree_traverser:
            return None
        
        return self._family_tree_traverser.get_agent(agent_name)
    
    def get_agent_path(self, from_agent: str, to_agent: str) -> Optional[List[str]]:
        """
        Find the path between two agents in the family tree.
        
        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            
        Returns:
            List of agent names in the path or None if no path
        """
        if not self._family_tree_traverser:
            return None
        
        return self._family_tree_traverser.get_path(from_agent, to_agent)
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """
        Find all agents with a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agent names with the capability
        """
        if not self._family_tree_traverser:
            return []
        
        return self._family_tree_traverser.find_by_capability(capability)
    
    def get_agents_by_role(self, role: str) -> List[str]:
        """
        Find all agents with a specific role.
        
        Args:
            role: Role to search for
            
        Returns:
            List of agent names with the role
        """
        if not self._family_tree_traverser:
            return []
        
        return self._family_tree_traverser.find_by_role(role)
    
    def get_agent_descendants(self, agent_name: str) -> List[str]:
        """
        Get all descendants of an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of descendant agent names
        """
        if not self._family_tree_traverser:
            return []
        
        return self._family_tree_traverser.get_descendants(agent_name)
    
    def get_agent_ancestors(self, agent_name: str) -> List[str]:
        """
        Get all ancestors of an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of ancestor agent names
        """
        if not self._family_tree_traverser:
            return []
        
        return self._family_tree_traverser.get_ancestors(agent_name)
    
    def get_agent_subtree(self, agent_name: str) -> Dict[str, Any]:
        """
        Extract a subtree rooted at a specific agent.
        
        Args:
            agent_name: Name of the agent to use as root
            
        Returns:
            Subtree data or empty dict if agent not found
        """
        if not self._family_tree_traverser:
            return {}
        
        return self._family_tree_traverser.get_subtree(agent_name)
    
    # .................................................. solution path versioning
    def store_solution_path(self, path_id: str, path_data: Dict[str, Any], 
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Store a solution path with version control.
        
        Args:
            path_id: Unique identifier for the solution path
            path_data: The solution path data
            metadata: Additional metadata about this revision
            
        Returns:
            Metadata about the storage operation
        """
        # Key for solution path storage
        key = f"solution_path:{path_id}"
        
        # Check if this path already exists
        existing_data = self.retrieve(key)
        if existing_data:
            # Path exists, load it
            path_version = SolutionPathVersion.from_json(existing_data, Compressor())
        else:
            # New path
            path_version = SolutionPathVersion(path_id, Compressor())
        
        # Add a new revision
        revision_number = path_version.add_revision(path_data, metadata)
        
        # Store the updated path version
        self.store(key, path_version.to_json())
        
        # Update the solution paths index
        self._update_solution_path_index(path_id, path_data, metadata, revision_number)
        
        return {
            "path_id": path_id,
            "revision_number": revision_number,
            "timestamp": time.time()
        }
    
    def _update_solution_path_index(self, path_id: str, path_data: Dict[str, Any],
                                 metadata: Dict[str, Any], revision_number: int) -> None:
        """
        Update the solution paths index.
        
        Args:
            path_id: Path ID
            path_data: Path data
            metadata: Revision metadata
            revision_number: Revision number
        """
        # Get or create index entry
        if path_id not in self._solution_paths_index:
            self._solution_paths_index[path_id] = {
                "revisions": [],
                "tags": set(),
                "actions": set(),
                "agents": set(),
                "last_updated": time.time(),
                "creator": metadata.get("creator", "unknown") if metadata else "unknown"
            }
        
        # Update the index
        index_entry = self._solution_paths_index[path_id]
        index_entry["last_updated"] = time.time()
        
        # Add revision
        revision_info = {
            "revision_number": revision_number,
            "timestamp": time.time(),
            "description": metadata.get("description", "") if metadata else "",
            "tags": metadata.get("tags", []) if metadata else []
        }
        
        # Ensure revisions list is long enough
        while len(index_entry["revisions"]) <= revision_number:
            index_entry["revisions"].append(None)
        
        # Update revision info
        index_entry["revisions"][revision_number] = revision_info
        
        # Update tags
        if metadata and "tags" in metadata:
            for tag in metadata["tags"]:
                index_entry["tags"].add(tag)
        
        # Update actions and agents from path data
        if "actions" in path_data:
            for action in path_data["actions"]:
                if "type" in action:
                    index_entry["actions"].add(action["type"])
                if "agent" in action:
                    index_entry["agents"].add(action["agent"])
        
        # Save the index
        self._save_solution_paths_index()
    
    def get_solution_path(self, path_id: str, 
                        revision: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a solution path.
        
        Args:
            path_id: Unique identifier for the solution path
            revision: Optional specific revision to retrieve
            
        Returns:
            The solution path data or None if not found
        """
        key = f"solution_path:{path_id}"
        path_data = self.retrieve(key)
        
        if not path_data:
            return None
            
        # Load the path version
        path_version = SolutionPathVersion.from_json(path_data, Compressor())
        
        if revision is not None:
            # Get specific revision
            try:
                return path_version.get_revision(revision)
            except ValueError:
                return None
        else:
            # Get current revision
            try:
                return path_version.get_current_revision()
            except ValueError:
                return None
    
    def list_solution_paths(self) -> List[str]:
        """
        List all solution path IDs.
        
        Returns:
            List of solution path IDs
        """
        prefix = "solution_path:"
        return [k[len(prefix):] for k in self.list_keys(prefix)]
    
    def get_revision_history(self, path_id: str) -> List[Dict[str, Any]]:
        """
        Get the revision history for a solution path.
        
        Args:
            path_id: Unique identifier for the solution path
            
        Returns:
            List of revision metadata or empty list if path not found
        """
        key = f"solution_path:{path_id}"
        path_data = self.retrieve(key)
        
        if not path_data:
            return []
            
        # Load the path version
        path_version = SolutionPathVersion.from_json(path_data, Compressor())
        return path_version.get_revision_history()
    
    def rollback_solution_path(self, path_id: str, 
                             revision: int) -> Optional[Dict[str, Any]]:
        """
        Rollback a solution path to a specific revision.
        
        Args:
            path_id: Unique identifier for the solution path
            revision: Revision number to roll back to
            
        Returns:
            The rolled-back solution path data or None if rollback failed
        """
        key = f"solution_path:{path_id}"
        path_data = self.retrieve(key)
        
        if not path_data:
            return None
            
        # Load the path version
        path_version = SolutionPathVersion.from_json(path_data, Compressor())
        
        try:
            # Rollback to specified revision
            result = path_version.rollback(revision)
            
            # Store the updated path version (with new current_revision)
            self.store(key, path_version.to_json())
            
            return result
        except ValueError:
            return None
    
    def find_solution_paths_by_tag(self, tag: str) -> List[str]:
        """
        Find solution paths with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching path IDs
        """
        results = []
        for path_id, index in self._solution_paths_index.items():
            if tag in index["tags"]:
                results.append(path_id)
        return results
    
    def find_solution_paths_by_action(self, action_type: str) -> List[str]:
        """
        Find solution paths that contain a specific action type.
        
        Args:
            action_type: Action type to search for
            
        Returns:
            List of matching path IDs
        """
        results = []
        for path_id, index in self._solution_paths_index.items():
            if action_type in index["actions"]:
                results.append(path_id)
        return results
    
    def find_solution_paths_by_agent(self, agent: str) -> List[str]:
        """
        Find solution paths that involve a specific agent.
        
        Args:
            agent: Agent name to search for
            
        Returns:
            List of matching path IDs
        """
        results = []
        for path_id, index in self._solution_paths_index.items():
            if agent in index["agents"]:
                results.append(path_id)
        return results
    
    def get_solution_paths_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the solution paths index.
        
        Returns:
            Copy of the solution paths index
        """
        # Return a copy to prevent accidental modification
        import copy
        return copy.deepcopy(self._solution_paths_index)
    
    # .................................................. learning storage
    def store_learning_data(self, model: str, data: Dict[str, Any]) -> None:
        """
        Store learning data for a specific model.
        
        Args:
            model: Model identifier
            data: Learning data to store
        """
        key = f"learning:{model}"
        self.store(key, data)
    
    def get_learning_data(self, model: str) -> Dict[str, Any]:
        """
        Retrieve learning data for a specific model.
        
        Args:
            model: Model identifier
            
        Returns:
            Learning data or empty dict if not found
        """
        key = f"learning:{model}"
        return self.retrieve(key, {})
    
    def list_learning_models(self) -> List[str]:
        """
        List all models with stored learning data.
        
        Returns:
            List of model identifiers
        """
        prefix = "learning:"
        return [k[len(prefix):] for k in self.list_keys(prefix)]
    
    def store_cross_session_data(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Store cross-session data.
        
        Args:
            session_id: Session identifier
            data: Session data to store
        """
        key = f"session:{session_id}"
        self.store(key, data)
    
    def get_cross_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve cross-session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        key = f"session:{session_id}"
        return self.retrieve(key, {})
    
    def list_sessions(self) -> List[str]:
        """
        List all sessions with stored data.
        
        Returns:
            List of session identifiers
        """
        prefix = "session:"
        return [k[len(prefix):] for k in self.list_keys(prefix)]
    
    # .................................................. planner-specific integration
    def store_planner_state(self, planner_id: str, state: Dict[str, Any]) -> None:
        """
        Store planner state.
        
        Args:
            planner_id: Planner identifier
            state: Planner state to store
        """
        key = f"planner_state:{planner_id}"
        self.store(key, state)
    
    def get_planner_state(self, planner_id: str) -> Dict[str, Any]:
        """
        Retrieve planner state.
        
        Args:
            planner_id: Planner identifier
            
        Returns:
            Planner state or empty dict if not found
        """
        key = f"planner_state:{planner_id}"
        return self.retrieve(key, {})
    
    def list_planner_states(self) -> List[str]:
        """
        List all planners with stored state.
        
        Returns:
            List of planner identifiers
        """
        prefix = "planner_state:"
        return [k[len(prefix):] for k in self.list_keys(prefix)]
    
    def store_planner_execution_history(self, planner_id: str, 
                                    execution_id: str, 
                                    history: List[Dict[str, Any]]) -> None:
        """
        Store planner execution history.
        
        Args:
            planner_id: Planner identifier
            execution_id: Execution identifier
            history: Execution history to store
        """
        key = f"planner_execution:{planner_id}:{execution_id}"
        
        # Large histories should be compressed
        if len(json.dumps(history)) > 1024:
            history_json = json.dumps(history)
            self.store_compressed(key, history_json)
            
            # Store a reference
            self.store(key, {
                "compressed": True,
                "compressed_key": key,
                "timestamp": time.time()
            })
        else:
            self.store(key, {
                "compressed": False,
                "history": history,
                "timestamp": time.time()
            })
    
    def get_planner_execution_history(self, planner_id: str, 
                                   execution_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve planner execution history.
        
        Args:
            planner_id: Planner identifier
            execution_id: Execution identifier
            
        Returns:
            Execution history or empty list if not found
        """
        key = f"planner_execution:{planner_id}:{execution_id}"
        data = self.retrieve(key)
        
        if not data:
            return []
        
        if data.get("compressed", False):
            # Retrieve from compressed storage
            compressed_key = data["compressed_key"]
            history_json = self.retrieve_compressed(compressed_key)
            if history_json:
                try:
                    return json.loads(history_json)
                except json.JSONDecodeError:
                    return []
            return []
        else:
            # Directly retrieve history
            return data.get("history", [])
    
    def list_planner_executions(self, planner_id: str = None) -> List[str]:
        """
        List all planner executions.
        
        Args:
            planner_id: Optional planner identifier to filter by
            
        Returns:
            List of execution identifiers
        """
        prefix = "planner_execution:"
        if planner_id:
            prefix = f"{prefix}{planner_id}:"
        
        execution_keys = [k for k in self.list_keys(prefix)]
        
        # Extract execution IDs
        execution_ids = []
        for key in execution_keys:
            parts = key.split(":")
            if len(parts) >= 3:
                execution_ids.append(parts[2])
        
        return execution_ids
    
    # .................................................. persistence
    def _load_all(self) -> None:
        """Load all data stores."""
        self._load_bugs()
        self._load_kv()
        self._load_compressed()
        self._load_family_tree()
        self._load_solution_paths_index()
    
    def _load_bugs(self) -> None:
        """Load bug database."""
        if self._mem_path.exists():
            with self._mem_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            # restore Counters - handle format from _save_bugs where we save as list of items
            for bug_id, entry in raw.items():
                if "vec" in entry:
                    # Convert the list of [key, value] pairs back to a Counter
                    counter_items = dict(entry["vec"]) if isinstance(entry["vec"], list) else entry["vec"]
                    entry["vec"] = Counter(counter_items)
            self._db = raw
    
    def _load_kv(self) -> None:
        """Load key-value store."""
        if self._kv_path.exists():
            with self._kv_path.open("r", encoding="utf-8") as fh:
                self._kv_store = json.load(fh)
    
    def _load_compressed(self) -> None:
        """Load compressed store."""
        if self._compressed_path.exists():
            with self._compressed_path.open("r", encoding="utf-8") as fh:
                self._compressed_store = json.load(fh)
    
    def _load_family_tree(self) -> None:
        """Load family tree and index."""
        # Load family tree
        if self._family_tree_path.exists():
            with self._family_tree_path.open("r", encoding="utf-8") as fh:
                self._family_tree = json.load(fh)
        
        # Load or create traverser
        if self._family_tree:
            if self._family_tree_index_path.exists():
                with self._family_tree_index_path.open("r", encoding="utf-8") as fh:
                    index_data = json.load(fh)
                self._family_tree_traverser = FamilyTreeTraverser.from_json(
                    self._family_tree, index_data)
            else:
                self._family_tree_traverser = FamilyTreeTraverser(self._family_tree)
    
    def _load_solution_paths_index(self) -> None:
        """Load solution paths index."""
        if self._solution_paths_index_path.exists():
            with self._solution_paths_index_path.open("r", encoding="utf-8") as fh:
                raw_index = json.load(fh)
            
            # Convert lists to sets
            for path_id, index in raw_index.items():
                if "tags" in index:
                    index["tags"] = set(index["tags"])
                if "actions" in index:
                    index["actions"] = set(index["actions"])
                if "agents" in index:
                    index["agents"] = set(index["agents"])
            
            self._solution_paths_index = raw_index

    def _save_bugs(self) -> None:
        """Save bug database."""
        serialisable = {
            bug_id: {**entry, "vec": list(entry["vec"].items())}
            for bug_id, entry in self._db.items()
        }
        with self._mem_path.open("w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)
    
    def _save_kv(self) -> None:
        """Save key-value store."""
        with self._kv_path.open("w", encoding="utf-8") as fh:
            json.dump(self._kv_store, fh, indent=2)
    
    def _save_compressed(self) -> None:
        """Save compressed store."""
        with self._compressed_path.open("w", encoding="utf-8") as fh:
            json.dump(self._compressed_store, fh, indent=2)
    
    def _save_solution_paths_index(self) -> None:
        """Save solution paths index."""
        # Convert sets to lists for JSON serialization
        serialisable = {}
        for path_id, index in self._solution_paths_index.items():
            serialisable[path_id] = {
                **index,
                "tags": list(index["tags"]) if isinstance(index["tags"], set) else index["tags"],
                "actions": list(index["actions"]) if isinstance(index["actions"], set) else index["actions"],
                "agents": list(index["agents"]) if isinstance(index["agents"], set) else index["agents"]
            }
        
        with self._solution_paths_index_path.open("w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)
            
    def save_all(self) -> None:
        """Save all data stores."""
        self._save_bugs()
        self._save_kv()
        self._save_compressed()
        self._save_solution_paths_index()

    # .................................................. debug
    def __len__(self) -> int:  # noqa: Dunder
        return len(self._db)

    def __repr__(self) -> str:  # noqa: Dunder
        return (
            f"<AgentMemory bugs={len(self._db)} "
            f"kv_entries={len(self._kv_store)} "
            f"compressed_entries={len(self._compressed_store)} "
            f"solution_paths={len(self._solution_paths_index)}>"
        )
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the memory stores."""
        compressed_stats = self.get_compression_stats()
        
        family_tree_stats = {
            "agents": len(self._family_tree_traverser.agent_index) if self._family_tree_traverser else 0,
            "roles": len(self._family_tree_traverser.role_index) if self._family_tree_traverser else 0,
            "capabilities": len(self._family_tree_traverser.capability_index) if self._family_tree_traverser else 0
        }
        
        return {
            "bugs": {
                "count": len(self._db),
                "path": str(self._mem_path)
            },
            "kv_store": {
                "count": len(self._kv_store),
                "path": str(self._kv_path)
            },
            "compressed_store": {
                **compressed_stats,
                "path": str(self._compressed_path)
            },
            "family_tree": {
                **family_tree_stats,
                "path": str(self._family_tree_path)
            },
            "solution_paths": {
                "count": len(self._solution_paths_index),
                "path": str(self._solution_paths_index_path)
            }
        }


# ---------------------------------------------------------------------------—
# Quick demo
# ---------------------------------------------------------------------------—
if __name__ == "__main__":  # pragma: no cover
    # Print header
    print("\n" + "=" * 80)
    print(" ENHANCED AGENT MEMORY DEMONSTRATION ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Create memory instance
    memory = AgentMemory()
    
    # Add a demo bug
    memory.add_entry(
        "BUG-42",
        "Fix import path typo causing ModuleNotFoundError in utils/date_parser",
        "diff --git a/utils/date_parser.py b/utils/date_parser.py\n- import dattime\n+ import datetime",
        {"severity": "medium", "author": "planner"}
    )
    
    # Store a family tree
    family_tree = {
        "root": "planner_agent",
        "capabilities": ["planning", "coordination"],
        "children": {
            "observer_agent": {
                "role": "analysis",
                "capabilities": ["bug_reproduction", "log_analysis"],
                "children": {}
            },
            "analyst_agent": {
                "role": "solution",
                "capabilities": ["code_analysis", "patch_generation"],
                "children": {
                    "specialist_agent": {
                        "role": "specialized_solution",
                        "capabilities": ["database_fix"],
                        "children": {}
                    }
                }
            },
            "verifier_agent": {
                "role": "verification",
                "capabilities": ["testing", "validation"],
                "children": {}
            }
        }
    }
    
    memory.store_family_tree(family_tree)
    print(f"Stored family tree with {len(memory.get_family_tree()['children'])} top-level agents")
    
    # Query family tree
    analysts = memory.get_agents_by_role("solution")
    print(f"Found {len(analysts)} agents with role 'solution': {analysts}")
    
    testers = memory.get_agents_by_capability("testing")
    print(f"Found {len(testers)} agents with capability 'testing': {testers}")
    
    descendants = memory.get_agent_descendants("analyst_agent")
    print(f"Found {len(descendants)} descendants of analyst_agent: {descendants}")
    
    path = memory.get_agent_path("specialist_agent", "verifier_agent")
    print(f"Path from specialist_agent to verifier_agent: {path}")
    
    # Store solution paths
    print("\nStoring solution paths:")
    
    # Create an initial solution path
    path_id = "solution-path-123"
    initial_solution = {
        "bug_id": "BUG-42",
        "description": "Fix null pointer in UserService.java",
        "actions": [
            {"type": "analyze", "agent": "observer_agent", "description": "Identify root cause"},
            {"type": "patch", "agent": "analyst_agent", "description": "Generate fix"}
        ],
        "priority": 0.8
    }
    
    # Store the initial solution
    result = memory.store_solution_path(path_id, initial_solution, {
        "creator": "planner_agent",
        "description": "Initial solution proposal",
        "tags": ["null-check", "critical"]
    })
