"""
patch_replay_buffer.py
─────────────────────
Specialized replay buffer for storing and retrieving code patch diffs with
comprehensive metadata and integrity verification.

This module builds on the CompressedReplayBuffer by adding specific functionality
for patch storage, including:
- Storage of code diffs with efficient compression
- Comprehensive metadata including timestamps, bug ID, success status, and entropy reduction
- Secure hash verification for patch integrity
- Advanced query capabilities by bug type, success status, etc.

The buffer serializes all data to disk for persistence between sessions.

Usage example:
    ```python
    # Create a buffer
    buffer = PatchReplayBuffer()
    
    # Add a patch with metadata
    buffer.add_patch(
        diff="--- old.py\n+++ new.py\n@@ -1,5 +1,5 @@\n-def broken():\n+def fixed():",
        bug_id="bug-123",
        success=True,
        entropy_reduction=0.75,
        additional_metadata={"fix_strategy": "rename_function"}
    )
    
    # Query for successful patches
    successful_patches = buffer.query_patches(success=True)
    
    # Get all patches for a specific bug
    bug_patches = buffer.query_patches(bug_id="bug-123")
    ```
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from compress import Compressor
from replay_buffer import CompressedReplayBuffer

# Default paths for persistence
DEFAULT_PATCH_BUFFER_DIR = Path(".triangulum")
DEFAULT_PATCH_BUFFER_FILE = DEFAULT_PATCH_BUFFER_DIR / "patch_buffer.json"


class PatchReplayBuffer(CompressedReplayBuffer):
    """
    Specialized replay buffer for storing code patch diffs with metadata.
    
    Features:
    - Storage of code diffs with efficient compression
    - Comprehensive metadata: timestamp, bug ID, success status, entropy reduction
    - Secure hash verification for patch integrity
    - Advanced query capabilities
    - Persistent storage to disk
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        max_tokens: int = 4096,
        buffer_file: Path = DEFAULT_PATCH_BUFFER_FILE
    ) -> None:
        """
        Initialize a patch replay buffer.
        
        Args:
            capacity: Maximum number of patches to store
            max_tokens: Maximum tokens for compression
            buffer_file: Path to the buffer file for persistence
        """
        # Initialize patch stats before calling parent constructor which may load from disk
        self._initialize_patch_stats()
        
        # Call parent constructor
        super().__init__(capacity, max_tokens, buffer_file)
        
        # Ensure patch stats are properly initialized
        self._ensure_patch_stats()
        
    def _initialize_patch_stats(self) -> None:
        """Initialize patch statistics."""
        self._patch_stats = {
            "total_patches": 0,
            "successful_patches": 0,
            "failed_patches": 0,
            "average_entropy_reduction": 0.0,
            "total_entropy_reduction": 0.0,
            "patches_by_bug": {}
        }
        
    def add_patch(
        self,
        diff: str,
        bug_id: str,
        success: bool = False,
        entropy_reduction: float = 0.0,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a code patch with metadata to the buffer.
        
        Args:
            diff: The code diff/patch text
            bug_id: Identifier for the bug being fixed
            success: Whether the patch was successful
            entropy_reduction: Measure of entropy reduction achieved (0.0 to 1.0)
            additional_metadata: Any additional metadata to store
            
        Returns:
            Dictionary with compression and storage metadata
        """
        # Ensure patch stats are properly initialized
        self._ensure_patch_stats()
        
        # Create patch record
        patch_record = {
            "type": "patch",
            "diff": diff,
            "bug_id": bug_id,
            "success": success,
            "entropy_reduction": entropy_reduction,
            "timestamp": int(time.time())
        }
        
        # Add any additional metadata
        if additional_metadata:
            patch_record.update(additional_metadata)
        
        # Calculate integrity hash for the diff content
        patch_record["diff_hash"] = hashlib.sha256(diff.encode()).hexdigest()
        
        # Update patch statistics
        self._update_patch_stats(patch_record)
        
        # Add to the buffer using parent class method
        compression_metadata = super().add(patch_record)
        
        return {
            "compression": compression_metadata,
            "patch_id": patch_record["diff_hash"][:8],
            "stored": True,
            "stats": self.get_patch_stats()
        }
    
    def query_patches(
        self,
        bug_id: Optional[str] = None,
        success: Optional[bool] = None,
        min_entropy_reduction: Optional[float] = None,
        max_count: Optional[int] = None,
        most_recent_first: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query patches with specific criteria.
        
        Args:
            bug_id: Filter by bug identifier
            success: Filter by success status
            min_entropy_reduction: Minimum entropy reduction
            max_count: Maximum number of results to return
            most_recent_first: Sort by timestamp descending
            
        Returns:
            List of matching patch records
        """
        # Get all patch records
        all_patches = list(self._buf)
        
        # Decompress all patches
        decompressed_patches = [self._decompress_episode(patch) for patch in all_patches]
        
        # Filter patches based on criteria
        filtered_patches = []
        for patch in decompressed_patches:
            # Skip non-patch records
            if not isinstance(patch, dict) or patch.get("type") != "patch":
                continue
                
            # Apply filters
            if bug_id is not None and patch.get("bug_id") != bug_id:
                continue
                
            if success is not None and patch.get("success") != success:
                continue
                
            if min_entropy_reduction is not None and patch.get("entropy_reduction", 0) < min_entropy_reduction:
                continue
                
            # Check patch integrity
            if self._verify_patch_integrity(patch):
                filtered_patches.append(patch)
            else:
                print(f"Warning: Patch integrity check failed for patch {patch.get('diff_hash', 'unknown')[:8]}")
        
        # Sort by timestamp
        if most_recent_first:
            filtered_patches.sort(key=lambda p: p.get("timestamp", 0), reverse=True)
        else:
            filtered_patches.sort(key=lambda p: p.get("timestamp", 0))
        
        # Apply max_count limit
        if max_count is not None and max_count > 0:
            filtered_patches = filtered_patches[:max_count]
            
        return filtered_patches
    
    def get_latest_patch(self, bug_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent patch, optionally for a specific bug.
        
        Args:
            bug_id: Optional bug identifier to filter by
            
        Returns:
            Most recent patch record or None if no matches
        """
        patches = self.query_patches(bug_id=bug_id, max_count=1, most_recent_first=True)
        return patches[0] if patches else None
    
    def get_patch_stats(self) -> Dict[str, Any]:
        """Get statistics about stored patches."""
        # Ensure patch stats are properly initialized
        self._ensure_patch_stats()
        return {**self._patch_stats}
    
    def _update_patch_stats(self, patch_record: Dict[str, Any]) -> None:
        """Update patch statistics with new patch."""
        # Ensure patch stats are properly initialized
        self._ensure_patch_stats()
        
        # Update total counts
        self._patch_stats["total_patches"] += 1
        
        # Update success/failure counts
        if patch_record.get("success", False):
            self._patch_stats["successful_patches"] += 1
        else:
            self._patch_stats["failed_patches"] += 1
        
        # Update entropy reduction
        entropy_reduction = patch_record.get("entropy_reduction", 0.0)
        self._patch_stats["total_entropy_reduction"] += entropy_reduction
        self._patch_stats["average_entropy_reduction"] = (
            self._patch_stats["total_entropy_reduction"] / self._patch_stats["total_patches"]
            if self._patch_stats["total_patches"] > 0 else 0.0
        )
        
        # Update bug-specific counts
        bug_id = patch_record.get("bug_id", "unknown")
        if bug_id not in self._patch_stats["patches_by_bug"]:
            self._patch_stats["patches_by_bug"][bug_id] = {
                "total": 0,
                "successful": 0,
                "latest_timestamp": 0
            }
            
        self._patch_stats["patches_by_bug"][bug_id]["total"] += 1
        if patch_record.get("success", False):
            self._patch_stats["patches_by_bug"][bug_id]["successful"] += 1
            
        self._patch_stats["patches_by_bug"][bug_id]["latest_timestamp"] = patch_record.get("timestamp", 0)
    
    def _verify_patch_integrity(self, patch_record: Dict[str, Any]) -> bool:
        """
        Verify the integrity of a patch record.
        
        Args:
            patch_record: The patch record to verify
            
        Returns:
            True if the patch passes integrity verification, False otherwise
        """
        # Check if this is a patch record
        if not isinstance(patch_record, dict) or patch_record.get("type") != "patch":
            return False
            
        # Get the stored hash
        stored_hash = patch_record.get("diff_hash")
        if not stored_hash:
            return False
            
        # Get the diff content
        diff = patch_record.get("diff")
        if not diff:
            return False
            
        # Calculate the hash
        calculated_hash = hashlib.sha256(diff.encode()).hexdigest()
        
        # Compare hashes
        return calculated_hash == stored_hash
    
    def _save_to_disk(self) -> None:
        """Save the buffer to disk with patch statistics."""
        # Ensure patch stats are properly initialized
        self._ensure_patch_stats()
        
        try:
            data = {
                "episodes": list(self._buf),
                "compression_stats": self._compression_stats,
                "patch_stats": self._patch_stats,
                "capacity": self.capacity,
                "max_tokens": self.max_tokens,
                "version": "1.0.0",
                "saved_at": int(time.time())
            }
            
            # Ensure directory exists
            self.buffer_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Write to file
            with open(self.buffer_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving patch buffer to disk: {e}")
    
    def _load_from_disk(self) -> bool:
        """
        Load the buffer from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        # Ensure patch stats are properly initialized before loading
        self._ensure_patch_stats()
        
        if not self.buffer_file.exists():
            return False
            
        try:
            with open(self.buffer_file, 'r') as f:
                data = json.load(f)
                
            # Load episodes
            episodes = data.get("episodes", [])
            for episode in episodes:
                if len(self._buf) < self.capacity:
                    self._buf.append(episode)
                    
            # Load stats with proper fallback
            self._compression_stats = data.get("compression_stats", self._compression_stats)
            
            # Initialize _patch_stats if it doesn't exist in the loaded data
            if "patch_stats" in data:
                self._patch_stats = data["patch_stats"]
            else:
                # If patch_stats not in data, reconstruct from loaded episodes
                self._reconstruct_patch_stats()
                
            return True
        except Exception as e:
            print(f"Error loading patch buffer from disk: {e}")
            # Initialize empty stats if loading fails
            self._initialize_patch_stats()
            return False
            
    def _ensure_patch_stats(self) -> None:
        """Ensure patch stats are initialized properly."""
        if not hasattr(self, '_patch_stats') or self._patch_stats is None:
            self._initialize_patch_stats()
            
    def _reconstruct_patch_stats(self) -> None:
        """Reconstruct patch statistics from loaded episodes."""
        # Initialize clean stats
        self._initialize_patch_stats()
        
        # Process all episodes
        for compressed_episode in self._buf:
            # Skip non-compressed episodes
            if not isinstance(compressed_episode, dict) or not compressed_episode.get("_compressed", False):
                continue
                
            # Decompress the episode
            episode = self._decompress_episode(compressed_episode)
            
            # Skip non-patch episodes
            if not isinstance(episode, dict) or episode.get("type") != "patch":
                continue
                
            # Update stats for this patch
            self._update_patch_stats(episode)
    
    def verify_all_patches(self) -> Tuple[int, int]:
        """
        Verify the integrity of all patches in the buffer.
        
        Returns:
            Tuple of (valid_count, invalid_count)
        """
        valid_count = 0
        invalid_count = 0
        
        # Get all patch records
        all_patches = list(self._buf)
        
        # Decompress all patches
        decompressed_patches = [self._decompress_episode(patch) for patch in all_patches]
        
        # Check each patch
        for patch in decompressed_patches:
            # Skip non-patch records
            if not isinstance(patch, dict) or patch.get("type") != "patch":
                continue
                
            # Verify integrity
            if self._verify_patch_integrity(patch):
                valid_count += 1
            else:
                invalid_count += 1
                print(f"Warning: Patch integrity check failed for patch {patch.get('diff_hash', 'unknown')[:8]}")
        
        return valid_count, invalid_count
    
    def clear(self) -> None:
        """Clear the buffer and reset statistics."""
        super().clear()
        self._initialize_patch_stats()
        self._save_to_disk()
    
    def __repr__(self) -> str:
        # Ensure patch stats are properly initialized
        self._ensure_patch_stats()
        
        ratio = self._compression_stats.get("compression_ratio", 1.0)
        bits = self._compression_stats.get("bits_saved", 0.0)
        patches = self._patch_stats.get("total_patches", 0)
        success_rate = (
            self._patch_stats["successful_patches"] / self._patch_stats["total_patches"]
            if self._patch_stats.get("total_patches", 0) > 0 else 0.0
        )
        return (
            f"<PatchReplayBuffer {len(self)}/{self.capacity} "
            f"patches={patches} success_rate={success_rate:.2f} "
            f"compression={ratio:.2f} bits_saved={bits:.1f}>"
        )


# ---------------------------------------------------------------------------—
# Example usage (manual test)
# ---------------------------------------------------------------------------—
if __name__ == "__main__":  # pragma: no cover
    # Test PatchReplayBuffer
    print("Patch Replay Buffer Test:")
    prb = PatchReplayBuffer(capacity=10)
    
    # Add some test patches
    test_diffs = [
        "--- old.py\n+++ new.py\n@@ -1,5 +1,5 @@\n-def broken():\n+def fixed():",
        "--- old.js\n+++ new.js\n@@ -10,7 +10,7 @@\n-  const error = true;\n+  const error = false;",
        "--- old.cpp\n+++ new.cpp\n@@ -20,6 +20,8 @@\n+  // Add null check\n+  if (ptr == nullptr) return;"
    ]
    
    bug_ids = ["bug-123", "bug-456", "bug-123"]
    success = [True, False, True]
    
    for i, (diff, bug_id, success_val) in enumerate(zip(test_diffs, bug_ids, success)):
        metadata = prb.add_patch(
            diff=diff,
            bug_id=bug_id,
            success=success_val,
            entropy_reduction=0.5 + i*0.1,
            additional_metadata={"attempt": i+1, "strategy": f"strategy-{i}"}
        )
        print(f"Added patch {i+1}, compression ratio: {metadata['compression']['compression_ratio']:.2f}")
    
    print(prb)
    print("Patch stats:", prb.get_patch_stats())
    
    # Test querying
    print("\nSuccessful patches:")
    for patch in prb.query_patches(success=True):
        print(f"  Bug {patch['bug_id']}, Hash: {patch['diff_hash'][:8]}, Entropy: {patch['entropy_reduction']}")
    
    print("\nPatches for bug-123:")
    for patch in prb.query_patches(bug_id="bug-123"):
        print(f"  Success: {patch['success']}, Strategy: {patch['strategy']}, Attempt: {patch['attempt']}")
    
    # Test integrity verification
    valid, invalid = prb.verify_all_patches()
    print(f"\nIntegrity check: {valid} valid, {invalid} invalid patches")
