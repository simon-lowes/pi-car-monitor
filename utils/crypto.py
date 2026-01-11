"""
Cryptographic Utilities
=======================
Secure hashing functions for privacy-sensitive data.
"""

import hashlib
import os
import secrets
from pathlib import Path
from typing import Optional


def generate_salt(length: int = 32) -> bytes:
    """Generate a cryptographically secure random salt."""
    return os.urandom(length)


def load_or_create_salt(salt_file: str) -> bytes:
    """
    Load existing salt from file or create a new one.

    Args:
        salt_file: Path to salt file

    Returns:
        Salt bytes
    """
    salt_path = Path(salt_file)
    salt_path.parent.mkdir(parents=True, exist_ok=True)

    if salt_path.exists():
        with open(salt_path, "rb") as f:
            return f.read()
    else:
        salt = generate_salt()
        with open(salt_path, "wb") as f:
            f.write(salt)
        # Secure file permissions (owner read/write only)
        os.chmod(salt_path, 0o600)
        return salt


def hash_sha256(data: str, salt: Optional[bytes] = None) -> str:
    """
    Create a SHA-256 hash of data with optional salt.

    Args:
        data: String to hash
        salt: Optional salt bytes

    Returns:
        Hex-encoded hash string
    """
    if salt:
        combined = salt + data.encode('utf-8')
    else:
        combined = data.encode('utf-8')

    return hashlib.sha256(combined).hexdigest()


def hash_plate(plate: str, salt: bytes) -> str:
    """
    Securely hash a number plate.

    Normalises the plate first (uppercase, no spaces).

    Args:
        plate: Number plate string
        salt: Device-specific salt

    Returns:
        Hex-encoded hash
    """
    # Normalise: uppercase, remove spaces and special chars
    normalised = ''.join(c for c in plate.upper() if c.isalnum())
    return hash_sha256(normalised, salt)


def secure_compare(a: str, b: str) -> bool:
    """
    Constant-time string comparison to prevent timing attacks.

    Args:
        a: First string
        b: Second string

    Returns:
        True if strings are equal
    """
    return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return secrets.token_hex(8)
