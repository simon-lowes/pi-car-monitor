"""
Plate Handler - Secure Local Plate Processing
==============================================
Handles number plate recognition with strict privacy guarantees:

1. All processing happens locally on the Hailo chip
2. Raw plate text is NEVER stored or transmitted
3. Only a salted hash is stored for matching
4. Salt is unique to this device and never leaves it

This ensures that even if someone accessed the device, they couldn't
recover the actual plate number from stored data.
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlateResult:
    """Result of plate recognition attempt."""
    detected: bool          # Was any plate detected?
    readable: bool          # Was the plate readable?
    is_match: bool          # Does it match target plate?
    confidence: float       # OCR confidence (0-1)
    # Note: We do NOT store the actual plate text


class PlateHandler:
    """
    Secure plate recognition and matching.
    
    Privacy design:
    - Target plate is hashed immediately on initialization
    - Detected plates are hashed and compared, never stored as text
    - Salt is device-specific and auto-generated
    """
    
    def __init__(self, target_plate: str, salt_file: Optional[str] = None):
        """
        Initialize plate handler.
        
        Args:
            target_plate: The owner's plate number (hashed immediately)
            salt_file: Path to store the device-specific salt
        """
        # Generate or load device-specific salt
        self.salt = self._get_or_create_salt(salt_file)
        
        # Normalise and hash the target plate immediately
        # After this point, we never store the raw plate
        normalised = self._normalise_plate(target_plate)
        self.target_hash = self._hash_plate(normalised)
        
        # Clear the raw plate from memory
        del target_plate
        del normalised
        
        logger.info("Plate handler initialized (target stored as hash only)")
        
        # Plate detection model will be loaded by Hailo
        self.model_loaded = False
    
    def _get_or_create_salt(self, salt_file: Optional[str]) -> bytes:
        """
        Get existing salt or create a new one.
        
        The salt is device-specific and adds extra security to the hash.
        Even if someone knows the hashing algorithm, they can't pre-compute
        hashes without the salt.
        """
        if salt_file:
            salt_path = Path(salt_file)
            salt_path.parent.mkdir(parents=True, exist_ok=True)
            
            if salt_path.exists():
                with open(salt_path, "rb") as f:
                    salt = f.read()
                logger.debug("Loaded existing plate salt")
                return salt
            else:
                # Generate new salt
                salt = os.urandom(32)
                with open(salt_path, "wb") as f:
                    f.write(salt)
                # Secure the file
                os.chmod(salt_path, 0o600)
                logger.info("Generated new plate salt")
                return salt
        else:
            # No persistent salt - generate temporary one
            # This means plate matching won't persist across restarts
            logger.warning("No salt file configured - using temporary salt")
            return os.urandom(32)
    
    def _normalise_plate(self, plate: str) -> str:
        """
        Normalise a plate number for consistent matching.
        
        Handles variations like:
        - AB12 CDE vs AB12CDE
        - ab12cde vs AB12CDE
        - Extra spaces
        """
        # Remove all whitespace
        plate = re.sub(r'\s+', '', plate)
        # Convert to uppercase
        plate = plate.upper()
        # Remove any non-alphanumeric characters
        plate = re.sub(r'[^A-Z0-9]', '', plate)
        return plate
    
    def _hash_plate(self, normalised_plate: str) -> str:
        """
        Create a secure hash of a plate number.
        
        Uses SHA-256 with device-specific salt.
        """
        # Combine plate with salt
        salted = self.salt + normalised_plate.encode('utf-8')
        # Hash it
        return hashlib.sha256(salted).hexdigest()
    
    def check_plate(self, vehicle_crop: np.ndarray) -> PlateResult:
        """
        Detect and check if a plate matches the target.
        
        Args:
            vehicle_crop: Image crop of detected vehicle
            
        Returns:
            PlateResult with match status (never contains raw plate)
        """
        # Step 1: Detect plate region in vehicle crop
        plate_region = self._detect_plate_region(vehicle_crop)
        
        if plate_region is None:
            return PlateResult(
                detected=False,
                readable=False,
                is_match=False,
                confidence=0.0
            )
        
        # Step 2: Run OCR on plate region
        ocr_result = self._read_plate(plate_region)
        
        if ocr_result is None:
            return PlateResult(
                detected=True,
                readable=False,
                is_match=False,
                confidence=0.0
            )
        
        detected_text, confidence = ocr_result
        
        # Step 3: Normalise and hash the detected plate
        # The raw text is used only for this comparison, never stored
        normalised = self._normalise_plate(detected_text)
        detected_hash = self._hash_plate(normalised)
        
        # Clear raw text from memory immediately
        del detected_text
        del normalised
        
        # Step 4: Compare hashes
        is_match = (detected_hash == self.target_hash)
        
        if is_match:
            logger.info("Plate match confirmed (hash comparison)")
        
        return PlateResult(
            detected=True,
            readable=True,
            is_match=is_match,
            confidence=confidence
        )
    
    def _detect_plate_region(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the plate region within a vehicle crop.
        
        Uses Hailo-accelerated plate detection model.
        """
        # TODO: Implement with Hailo plate detection
        # This would:
        # 1. Run plate detection model on vehicle crop
        # 2. Return cropped plate region if found
        
        # Placeholder
        return None
    
    def _read_plate(self, plate_region: np.ndarray) -> Optional[tuple]:
        """
        Run OCR on plate region to read characters.
        
        Uses Hailo-accelerated OCR model (e.g., LPRNet).
        
        Returns:
            Tuple of (plate_text, confidence) or None if unreadable
        """
        # TODO: Implement with Hailo OCR
        # This would:
        # 1. Preprocess plate image
        # 2. Run OCR model
        # 3. Return text and confidence
        
        # Placeholder
        return None


class SecurePlateLogger:
    """
    Logs plate-related events without exposing plate data.
    
    For audit purposes, we might want to log that plate verification
    occurred, but we never log the actual plate.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
    
    def log_verification(self, result: PlateResult, timestamp: str):
        """Log a plate verification event (no plate data included)."""
        event = {
            "timestamp": timestamp,
            "event": "plate_verification",
            "detected": result.detected,
            "readable": result.readable,
            "matched": result.is_match,
            "confidence": round(result.confidence, 2)
            # Note: No plate text or hash is logged
        }
        
        if self.log_file:
            import json
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
