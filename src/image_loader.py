"""
Image Loading, Preprocessing, and SHA-256 Data Integrity Module for CosmoAlign.

Handles reading image files from disk, verifying file integrity, computing SHA-256
file checksums, and converting images to single-channel grayscale format.
"""

import os
import hashlib
from typing import Tuple
import cv2
import numpy as np


def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """
    Computes the SHA-256 cryptographic hash checksum of a file on disk.

    Used for raw scientific data integrity verification to ensure input TIFFs
    are immutable and uncorrupted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot compute SHA-256 hash. File not found: {file_path}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified file path using OpenCV.

    Args:
        image_path (str): Path to target image file.

    Returns:
        np.ndarray: Loaded BGR image array.

    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If file cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Failed to load image from '{image_path}'. File may be corrupt or an unsupported format."
        )

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts multi-channel BGR/RGB image to single-channel uint8 grayscale.
    """
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")
