"""
learning/replay_buffer.py
─────────────────────────
Tiny cyclic **experience replay buffer** that keeps the last *N* bug-resolution
"episodes" so the `AdaptiveOptimizer` (or any other learner) can sample
i.i.d. mini-batches.

Why a replay buffer?
────────────────────
*  Stabilises gradient updates – prevents bias towards the most recent episode.
*  Enables **off-policy** learning: Optimiser can re-evaluate old experience
   when its internal target changes (e.g. new reward function).
*  Lightweight –  <80 LOC, pure std-lib.

Data model
──────────
Every episode is stored as a *dict* with the canonical keys produced by
`MetaAgent.maybe_update()` **plus** anything the caller wishes to include.
Typical example::

    {
      "bugs_seen": 314,
      "success_rate": 0.92,
      "mean_tokens": 1034,
      "reward": 1.0,                # optional
      "timestamp": 1_723_456_789
    }

Public API
──────────
    buf = ReplayBuffer(capacity=500)
    buf.add(episode_dict)
    batch = buf.sample(batch_size=32)
"""
import hashlib
import json
import os
import secrets
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, TypeVar, Sequence, Optional, Tuple, Any

from compress import Compressor

__all__ = ["ReplayBuffer", "CompressedReplayBuffer"]

# Default paths for persistence
DEFAULT_BUFFER_DIR = Path(".triangulum")
DEFAULT_BUFFER_FILE = DEFAULT_BUFFER_DIR / "replay_buffer.json"
DEFAULT_BUFFER_DIR.mkdir(exist_ok=True, parents=True)

# Maximum tokens for compression
DEFAULT_MAX_TOKENS = 4096

T = TypeVar('T')

def secure_sample(population: Sequence[T], k: int) -> List[T]:
    """
    Cryptographically secure random sampling without replacement.
    Replacement for random.sample() using the secrets module.
    
    Args:
        population: The sequence to sample from
        k: Number of samples to take
        
    Returns:
        List of k unique elements from population
    """
    if k > len(population):
        raise ValueError("Sample size cannot exceed population size")
    
    # Convert to list for indexing if not already
    pop_list = list(population)
    result = []
    
    # Create a copy of indices that we'll sample from
    indices = list(range(len(pop_list)))
    
    for _ in range(k):
        # Securely select a random index from remaining indices
        idx_pos = secrets.randbelow(len(indices))
        # Get the actual index from our remaining indices list
        idx = indices.pop(idx_pos)
        # Add the corresponding item to our result
        result.append(pop_list[idx])
    
    return result


class ReplayBuffer:
    """
    Cyclic deque with O(1) append; uniform random sampling without replacement.
    """

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf: Deque[Dict] = deque(maxlen=capacity)
        self.capacity = capacity

    # --------------------------------------------------------------------- add
    def add(self, episode: Dict) -> None:
        """
        Append a new episode dictionary; adds a 'timestamp' if missing.
        """
        if "timestamp" not in episode:
            episode = {**episode, "timestamp": int(time.time())}
        self._buf.append(episode)

    # --------------------------------------------------------------------- sample
    def sample(self, batch_size: int) -> List[Dict]:
        """
        Random sample without replacement (uniform probability).
        
        Args:
            batch_size: Number of episodes to sample
            
        Returns:
            List of sampled episodes
            
        Raises:
            ValueError: If batch_size > current buffer size
        """
        n = len(self._buf)
        k = min(batch_size, n)  # can't sample more than we have
        
        if k == 0:
            return []
        
        if k == n:
            return list(self._buf)  # return everything if asked for entire buffer
        
        # Use secure_sample for cryptographically secure sampling
        return secure_sample(self._buf, k)

    # -------------------------------------------------------------- utilities
    def __len__(self) -> int:  # noqa: Dunder
        return len(self._buf)

    def is_full(self) -> bool:
        """Return True when buffer reached `capacity`."""
        return len(self._buf) == self.capacity

    def clear(self) -> None:
        """Drop all stored episodes (mainly for unit tests)."""
        self._buf.clear()

    # -------------------------------------------------------------- repr/debug
    def __repr__(self) -> str:  # noqa: Dunder
        return f"<ReplayBuffer {len(self)}/{self.capacity}>"


# ---------------------------------------------------------------------------—
# Enhanced CompressedReplayBuffer
# ---------------------------------------------------------------------------—
class CompressedReplayBuffer(ReplayBuffer):
    """
    Enhanced replay buffer with compression and persistence capabilities.
    
    Features:
    - Compresses episodes using the Compressor from compress.py
    - Persists buffer contents to disk in JSON format
    - Provides integrity verification with checksums
    - Tracks compression statistics
    """
    
    def __init__(
        self, 
        capacity: int = 1000, 
        max_tokens: int = DEFAULT_MAX_TOKENS,
        buffer_file: Path = DEFAULT_BUFFER_FILE
    ) -> None:
        """
        Initialize a compressed replay buffer.
        
        Args:
            capacity: Maximum number of episodes to store
            max_tokens: Maximum tokens for compression
            buffer_file: Path to the buffer file for persistence
        """
        super().__init__(capacity)
        self.max_tokens = max_tokens
        self.buffer_file = buffer_file
        self.compressor = Compressor(max_tokens)
        self._compression_stats = {
            "total_original_size": 0,
            "total_compressed_size": 0,
            "compression_ratio": 1.0,
            "bits_saved": 0.0,
            "episodes_compressed": 0
        }
        self._load_from_disk()
    
    def add(self, episode: Dict) -> Dict[str, Any]:
        """
        Compress and add an episode to the buffer.
        
        Args:
            episode: Episode data dictionary
            
        Returns:
            Compression statistics
        """
        # Add timestamp if missing
        if "timestamp" not in episode:
            episode = {**episode, "timestamp": int(time.time())}
        
        # Create a compressed version of the episode
        compressed_episode, metadata = self._compress_episode(episode)
        
        # Add to buffer
        self._buf.append(compressed_episode)
        
        # Update stats
        self._update_compression_stats(metadata)
        
        # Save to disk
        self._save_to_disk()
        
        return metadata
    
    def sample(self, batch_size: int) -> List[Dict]:
        """
        Sample episodes and decompress them.
        
        Args:
            batch_size: Number of episodes to sample
            
        Returns:
            List of decompressed episodes
        """
        # Sample compressed episodes
        compressed_batch = super().sample(batch_size)
        
        # Decompress each episode
        return [self._decompress_episode(episode) for episode in compressed_batch]
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get statistics about compression."""
        return {**self._compression_stats}
    
    def _compress_episode(self, episode: Dict) -> Tuple[Dict, Dict[str, Any]]:
        """
        Compress an episode and return both the compressed episode and metadata.
        
        Args:
            episode: Original episode dictionary
            
        Returns:
            Tuple of (compressed_episode, metadata)
        """
        # Serialize the episode to JSON
        episode_json = json.dumps(episode)
        original_size = len(episode_json)
        
        # Calculate checksum for integrity verification
        checksum = hashlib.sha256(episode_json.encode()).hexdigest()
        
        # Compress the episode text
        compressed_text, bits_saved = self.compressor.compress(episode_json)
        compressed_size = len(compressed_text)
        
        # Create the compressed episode
        compressed_episode = {
            "_compressed": True,
            "data": compressed_text,
            "metadata": {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0,
                "bits_saved": bits_saved,
                "checksum": checksum,
                "timestamp": episode.get("timestamp", int(time.time()))
            }
        }
        
        return compressed_episode, compressed_episode["metadata"]
    
    def _decompress_episode(self, compressed_episode: Dict) -> Dict:
        """
        Decompress an episode.
        
        Args:
            compressed_episode: Compressed episode dictionary
            
        Returns:
            Original episode dictionary
        """
        # Check if this is a compressed episode
        if not compressed_episode.get("_compressed", False):
            return compressed_episode
        
        # Get the compressed data and metadata
        compressed_text = compressed_episode["data"]
        metadata = compressed_episode["metadata"]
        
        # Decompress the JSON
        decompressed_json = compressed_text
        
        # Verify integrity with checksum
        checksum = hashlib.sha256(decompressed_json.encode()).hexdigest()
        if checksum != metadata["checksum"]:
            print(f"Warning: Checksum verification failed for episode. Using compressed version.")
            return {"_error": "Checksum verification failed", **metadata}
        
        # Parse the JSON back to a dictionary
        try:
            episode = json.loads(decompressed_json)
            return episode
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON for episode. Using compressed version.")
            return {"_error": "JSON decode error", **metadata}
    
    def _update_compression_stats(self, metadata: Dict[str, Any]) -> None:
        """Update compression statistics with new episode metadata."""
        self._compression_stats["total_original_size"] += metadata["original_size"]
        self._compression_stats["total_compressed_size"] += metadata["compressed_size"]
        self._compression_stats["episodes_compressed"] += 1
        self._compression_stats["bits_saved"] += metadata["bits_saved"]
        
        # Update average compression ratio
        if self._compression_stats["episodes_compressed"] > 0:
            self._compression_stats["compression_ratio"] = (
                self._compression_stats["total_compressed_size"] / 
                self._compression_stats["total_original_size"]
                if self._compression_stats["total_original_size"] > 0 else 1.0
            )
    
    def _save_to_disk(self) -> None:
        """Save the buffer to disk."""
        try:
            data = {
                "episodes": list(self._buf),
                "stats": self._compression_stats,
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
            print(f"Error saving buffer to disk: {e}")
    
    def _load_from_disk(self) -> bool:
        """
        Load the buffer from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
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
                    
            # Load stats
            self._compression_stats = data.get("stats", self._compression_stats)
            
            return True
        except Exception as e:
            print(f"Error loading buffer from disk: {e}")
            return False
    
    def clear(self) -> None:
        """Clear the buffer and reset statistics."""
        super().clear()
        self._compression_stats = {
            "total_original_size": 0,
            "total_compressed_size": 0,
            "compression_ratio": 1.0,
            "bits_saved": 0.0,
            "episodes_compressed": 0
        }
        self._save_to_disk()
    
    def __repr__(self) -> str:
        ratio = self._compression_stats["compression_ratio"]
        bits = self._compression_stats["bits_saved"]
        return (
            f"<CompressedReplayBuffer {len(self)}/{self.capacity} "
            f"ratio={ratio:.2f} bits_saved={bits:.1f}>"
        )


# ---------------------------------------------------------------------------—
# Example usage (manual test)
# ---------------------------------------------------------------------------—
if __name__ == "__main__":  # pragma: no cover
    # Test standard ReplayBuffer
    print("Standard ReplayBuffer:")
    rb = ReplayBuffer(capacity=5)
    for i in range(7):  # add more than capacity to test cyclic behaviour
        rb.add({"bugs_seen": i, "success_rate": 0.8 + 0.01 * i})
    print(rb)
    print("Sample 3:", rb.sample(3))
    
    # Test CompressedReplayBuffer
    print("\nCompressed ReplayBuffer:")
    crb = CompressedReplayBuffer(capacity=5)
    for i in range(7):
        metadata = crb.add({
            "bugs_seen": i, 
            "success_rate": 0.8 + 0.01 * i,
            "details": "This is a longer description that will benefit from compression " * 5
        })
        print(f"Added episode {i}, compression ratio: {metadata['compression_ratio']:.2f}")
        
    print(crb)
    print("Compression stats:", crb.get_compression_stats())
    print("Sample 3:", crb.sample(3))
