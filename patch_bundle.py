"""
tooling/patch_bundle.py
───────────────────────
Cryptographically-verifiable *diff archive*.

The bundle is a single **tar file** that contains

    ├── manifest.json      (metadata + SHA-256 of the diff)
    └── patch.diff         (unified diff)

Goals
─────
1. **Integrity**   – SHA-256 hash in manifest must match `patch.diff`.
2. **Idempotency** – Applying the *same* bundle twice is a no-op.
3. **CLI usability** – `python -m tooling.patch_bundle make/apply/verify`.
4. **Security**    – Cryptographic hash verification to prevent tampering.

No 3rd-party dependencies – only `tarfile`, `hashlib`, `json`, `subprocess`,
and `pathlib`.
"""

from __future__ import annotations

import json
import hashlib
import logging
import subprocess
import tarfile
import time
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("patch_bundle")

# internal constants
BUNDLE_DIR = Path(".triangulum") / "bundles"
APPLIED_REG = BUNDLE_DIR / "applied.json"
HASH_LOG_FILE = BUNDLE_DIR / "hash_verification.log"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────────
class BundleError(RuntimeError):
    """Raised on IO/hash/verification failure."""


class IntegrityError(BundleError):
    """Raised when hash verification fails (indicates tampering)."""


class HashAlgorithm:
    """
    Secure hash algorithm options for patch verification.
    
    Currently supports:
    - SHA-256 (default)
    - SHA-384
    - SHA-512
    """
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    
    @staticmethod
    def get_algorithm(name: str) -> str:
        """Get algorithm name if valid, or default to SHA256."""
        if name.lower() in {HashAlgorithm.SHA256, HashAlgorithm.SHA384, HashAlgorithm.SHA512}:
            return name.lower()
        
        logger.warning(f"Invalid hash algorithm '{name}', defaulting to SHA-256")
        return HashAlgorithm.SHA256
    
    @staticmethod
    def get_all_algorithms() -> list[str]:
        """Get list of all supported hash algorithms."""
        return [HashAlgorithm.SHA256, HashAlgorithm.SHA384, HashAlgorithm.SHA512]


def calculate_hash(content: Union[str, bytes], algorithm: str = HashAlgorithm.SHA256) -> str:
    """
    Calculate a secure hash for content.
    
    Args:
        content: Content to hash (string or bytes)
        algorithm: Hash algorithm to use (sha256, sha384, sha512)
        
    Returns:
        Hexadecimal hash digest
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
        
    hash_obj = hashlib.new(HashAlgorithm.get_algorithm(algorithm))
    hash_obj.update(content)
    return hash_obj.hexdigest()


def verify_content_hash(
    content: Union[str, bytes], 
    expected_hash: str,
    algorithm: str = HashAlgorithm.SHA256
) -> bool:
    """
    Verify that content matches its expected hash.
    
    Args:
        content: Content to verify (string or bytes)
        expected_hash: Expected hash digest
        algorithm: Hash algorithm to use (sha256, sha384, sha512)
        
    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = calculate_hash(content, algorithm)
    match = actual_hash == expected_hash
    
    # Log verification attempt
    if HASH_LOG_FILE.parent.exists():
        timestamp = datetime.now().isoformat()
        result = "✓ verified" if match else "✗ FAILED"
        with open(HASH_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {result} - algorithm={algorithm}\n")
    
    return match

# ───────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ───────────────────────────────────────────────────────────────────────────────
def _load_applied() -> Dict[str, str]:
    if APPLIED_REG.exists():
        return json.loads(APPLIED_REG.read_text(encoding="utf-8"))
    return {}


def _save_applied(reg: Dict[str, str]) -> None:
    APPLIED_REG.write_text(json.dumps(reg, indent=2), encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
def create_bundle(
    bug_id: str, 
    diff_text: str,
    algorithm: str = HashAlgorithm.SHA256,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Package *diff_text* into `<bug_id>.tri.tgz` in BUNDLE_DIR.
    
    Args:
        bug_id: Unique identifier for the bug
        diff_text: Unified diff content
        algorithm: Hash algorithm to use (sha256, sha384, sha512)
        metadata: Optional additional metadata to include in manifest
        
    Returns:
        Path to the created bundle file
        
    Raises:
        BundleError: If bundle already exists or creation fails
    """
    # Calculate secure hash of diff content
    diff_hash = calculate_hash(diff_text, algorithm)
    
    # Create manifest with metadata
    manifest = {
        "bug_id": bug_id,
        "hash_algorithm": HashAlgorithm.get_algorithm(algorithm),
        "content_hash": diff_hash,
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    
    # Add additional metadata if provided
    if metadata:
        manifest["metadata"] = metadata
    
    # Create bundle path
    bundle_path = BUNDLE_DIR / f"{bug_id}.tri.tgz"
    if bundle_path.exists():
        raise BundleError(f"Bundle already exists: {bundle_path}")
    
    logger.info(f"Creating bundle for bug {bug_id} with {algorithm} hash")
    
    try:
        # Create bundle with manifest and diff
        with tarfile.open(bundle_path, "w:gz") as tf:
            # Add manifest
            manifest_data = json.dumps(manifest, indent=2).encode("utf-8")
            manifest_info = _make_info("manifest.json", manifest_data)
            tf.addfile(manifest_info, io.BytesIO(manifest_data))
            
            # Add diff
            diff_bytes = diff_text.encode("utf-8")
            diff_info = _make_info("patch.diff", diff_bytes)
            tf.addfile(diff_info, io.BytesIO(diff_bytes))
        
        logger.info(f"Created bundle: {bundle_path}")
        return bundle_path
        
    except Exception as e:
        # Clean up if creation fails
        if bundle_path.exists():
            try:
                bundle_path.unlink()
            except:
                pass
        
        raise BundleError(f"Failed to create bundle: {e}") from e


def verify_bundle(bundle_path: Path, verify_hash: bool = True) -> Dict:
    """
    Validate archive structure and verify content hash.
    
    Args:
        bundle_path: Path to the bundle file
        verify_hash: Whether to verify content hash (default: True)
        
    Returns:
        The manifest dict on success
        
    Raises:
        BundleError: If bundle doesn't exist or is invalid
        IntegrityError: If content hash verification fails
    """
    if not bundle_path.exists():
        raise BundleError(f"No such bundle: {bundle_path}")
    
    try:
        # Extract files from bundle
        with tarfile.open(bundle_path, "r:gz") as tf:
            try:
                manifest_bytes = tf.extractfile("manifest.json").read()
                patch_bytes = tf.extractfile("patch.diff").read()
            except KeyError as e:
                raise BundleError("Bundle missing required members") from e
            except (tarfile.TarError, IOError) as e:
                raise BundleError(f"Failed to extract bundle contents: {e}") from e
        
        # Parse manifest
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise BundleError(f"Invalid manifest JSON: {e}") from e
        
        # Get hash details from manifest
        content_hash = manifest.get("content_hash")
        if not content_hash:
            # Try legacy format
            content_hash = manifest.get("sha256")
            algorithm = HashAlgorithm.SHA256
            if not content_hash:
                raise BundleError("Manifest missing content hash")
        else:
            algorithm = manifest.get("hash_algorithm", HashAlgorithm.SHA256)
        
        # Verify hash if requested
        if verify_hash:
            if not verify_content_hash(patch_bytes, content_hash, algorithm):
                logger.error(f"Hash verification failed for {bundle_path}")
                raise IntegrityError(
                    f"Content hash verification failed for bundle {bundle_path.name} - "
                    f"possible tampering detected"
                )
            
            logger.info(f"Verified bundle integrity for {bundle_path.name}")
        
        return manifest
        
    except (tarfile.TarError, IOError) as e:
        raise BundleError(f"Failed to open bundle: {e}") from e


def apply_bundle(bundle_path: Path, skip_hash_verify: bool = False) -> Tuple[bool, str]:
    """
    Idempotently apply `patch.diff` inside bundle via `git apply`.
    On success, records hash in `applied.json`.
    
    Args:
        bundle_path: Path to the bundle file
        skip_hash_verify: Skip hash verification (not recommended)
        
    Returns:
        Tuple of (success, message)
        
    Raises:
        BundleError: If bundle verification or application fails
        IntegrityError: If content hash verification fails
    """
    try:
        # Verify bundle
        manifest = verify_bundle(bundle_path, verify_hash=not skip_hash_verify)
        
        # Get hash (support both new and legacy formats)
        content_hash = manifest.get("content_hash", manifest.get("sha256"))
        if not content_hash:
            raise BundleError("Manifest missing content hash")
        
        # Check if already applied
        reg = _load_applied()
        if content_hash in reg:
            msg = f"✓ Bundle already applied on {reg[content_hash]}"
            logger.info(msg)
            print(msg)
            return True, msg
        
        # Log application attempt
        bug_id = manifest.get("bug_id", "unknown")
        logger.info(f"Applying bundle for bug {bug_id}")
        
        # Dry-run first
        _git_apply(["--check", str(bundle_path)])
        
        # Real apply
        _git_apply([str(bundle_path)])
        
        # Record application
        applied_time = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        reg[content_hash] = applied_time
        _save_applied(reg)
        
        msg = f"✓ Patch for bug {bug_id} applied and recorded"
        logger.info(msg)
        print(msg)
        return True, msg
        
    except IntegrityError as e:
        # Specific handling for integrity errors
        msg = f"✗ {e}"
        logger.error(msg)
        print(msg, file=sys.stderr)
        return False, msg
        
    except BundleError as e:
        # Handling for other bundle errors
        msg = f"✗ {e}"
        logger.error(msg)
        print(msg, file=sys.stderr)
        return False, msg


# ───────────────────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────────────────
def _make_info(name: str, data: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = time.time()
    return info


def _git_apply(args: list[str]) -> None:
    """
    Wrapper around `git apply` that accepts either '--check diff' or 'bundle.tgz'
    (we untar patch.diff to stdin if bundle path detected).
    """
    if args and args[-1].endswith(".tgz"):
        bundle = Path(args[-1])
        with tarfile.open(bundle, "r:gz") as tf:
            patch_bytes = tf.extractfile("patch.diff").read()
        cmd = ["git", "apply"] + args[:-1]
        proc = subprocess.run(
            cmd, input=patch_bytes, text=True, capture_output=True, check=False
        )
    else:
        cmd = ["git", "apply"] + args
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode:
        raise BundleError(
            f"`{' '.join(cmd)}` failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


# ───────────────────────────────────────────────────────────────────────────────
# CLI wrapper
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys
    import io

    parser = argparse.ArgumentParser(description="Triangulum patch bundle utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Create bundle command
    mk = sub.add_parser("make", help="Create bundle")
    mk.add_argument("bug_id", help="Bug identifier")
    mk.add_argument("diff_file", help="Path to diff file")
    mk.add_argument("--algorithm", choices=HashAlgorithm.get_all_algorithms(), 
                   default=HashAlgorithm.SHA256, help="Hash algorithm (default: sha256)")
    mk.add_argument("--metadata", help="JSON string with additional metadata")

    # Verify bundle command
    ver = sub.add_parser("verify", help="Verify bundle")
    ver.add_argument("bundle", help="Path to bundle file")
    ver.add_argument("--skip-hash", action="store_true", help="Skip hash verification")

    # Apply bundle command
    ap = sub.add_parser("apply", help="Apply bundle idempotently")
    ap.add_argument("bundle", help="Path to bundle file")
    ap.add_argument("--skip-hash", action="store_true", help="Skip hash verification (not recommended)")

    # Hash command
    hash_cmd = sub.add_parser("hash", help="Calculate hash of a file")
    hash_cmd.add_argument("file", help="File to hash")
    hash_cmd.add_argument("--algorithm", choices=HashAlgorithm.get_all_algorithms(), 
                        default=HashAlgorithm.SHA256, help="Hash algorithm (default: sha256)")

    ns = parser.parse_args()

    try:
        if ns.cmd == "make":
            # Parse metadata if provided
            metadata = None
            if ns.metadata:
                try:
                    metadata = json.loads(ns.metadata)
                except json.JSONDecodeError as e:
                    print(f"✗ Invalid metadata JSON: {e}", file=sys.stderr)
                    sys.exit(1)
            
            # Read diff and create bundle
            diff = Path(ns.diff_file).read_text(encoding="utf-8")
            path = create_bundle(ns.bug_id, diff, ns.algorithm, metadata)
            print(f"✓ Bundle created: {path}")
            print(f"  Hash algorithm: {ns.algorithm}")
            print(f"  Content hash: {calculate_hash(diff, ns.algorithm)}")
            
        elif ns.cmd == "verify":
            m = verify_bundle(Path(ns.bundle), verify_hash=not ns.skip_hash)
            print("✓ Bundle verification successful")
            print(json.dumps(m, indent=2))
            
        elif ns.cmd == "apply":
            success, message = apply_bundle(Path(ns.bundle), skip_hash_verify=ns.skip_hash)
            if not success:
                sys.exit(1)
                
        elif ns.cmd == "hash":
            content = Path(ns.file).read_bytes()
            file_hash = calculate_hash(content, ns.algorithm)
            print(f"✓ {ns.algorithm} hash of {ns.file}:")
            print(file_hash)
            
    except IntegrityError as e:
        print("✗", e, file=sys.stderr)
        print("  This error indicates possible tampering with the patch content.", file=sys.stderr)
        sys.exit(2)
        
    except BundleError as e:
        print("✗", e, file=sys.stderr)
        sys.exit(1)
