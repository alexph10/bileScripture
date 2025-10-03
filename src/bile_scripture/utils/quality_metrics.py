"""
Quality Metrics for Texture Analysis

This module provides functions to analyze texture quality, detect artifacts,
and assess the suitability of generated textures for PBR workflows.
"""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


def calculate_texture_quality(image: Image.Image) -> dict[str, float]:
    """
    Calculate comprehensive quality metrics for a texture.

    This helps us understand if our generated textures are good enough for use.

    Args:
        image: Input PIL Image to analyze

    Returns:
        Dictionary with quality metrics:
        - 'sharpness': How sharp/blurry the texture is (higher = sharper)
        - 'contrast': Dynamic range of the texture (higher = more contrast)
        - 'detail_level': Amount of fine detail present
        - 'uniformity': How evenly distributed features are
        - 'overall_score': Combined quality score (0-1)
    """
    # Convert to grayscale for analysis
    gray = np.array(image.convert("L"), dtype=np.float32)
    gray_uint8 = gray.astype(np.uint8)

    # 1. Sharpness - using Laplacian variance
    # Sharp images have high variance in their Laplacian (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.var(laplacian.astype(np.float64)))

    # 2. Contrast - using standard deviation
    # High contrast images have wide range of pixel values
    contrast = float(np.std(gray.astype(np.float64)))

    # 3. Detail level - using high-frequency content
    # More detail = more high-frequency components
    detail_level = _analyze_detail_content(gray_uint8)

    # 4. Uniformity - how evenly features are distributed
    uniformity = _calculate_uniformity(gray_uint8)

    # Normalize scores to 0-1 range
    sharpness_norm = min(1.0, sharpness / 1000)  # Typical range: 0-2000
    contrast_norm = min(1.0, contrast / 50)  # Typical range: 0-100
    detail_norm = min(1.0, detail_level)  # Already 0-1
    uniformity_norm = uniformity  # Already 0-1

    # Calculate overall score (weighted average)
    overall_score = (
        sharpness_norm * 0.25
        + contrast_norm * 0.25
        + detail_norm * 0.3
        + uniformity_norm * 0.2
    )

    return {
        "sharpness": float(sharpness_norm),
        "contrast": float(contrast_norm),
        "detail_level": float(detail_norm),
        "uniformity": float(uniformity_norm),
        "overall_score": float(overall_score),
    }


def analyze_frequency_content(image: Image.Image) -> dict[str, float]:
    """
    Analyze the frequency content of a texture using FFT.

    This tells us about the scale of features in the texture:
    - High frequency = fine details
    - Low frequency = large features

    Args:
        image: Input PIL Image

    Returns:
        Dictionary with frequency analysis:
        - 'low_freq_energy': Energy in low frequencies (large features)
        - 'mid_freq_energy': Energy in mid frequencies (medium features)
        - 'high_freq_energy': Energy in high frequencies (fine details)
        - 'dominant_frequency': The most prominent frequency
    """
    # Convert to grayscale
    gray = np.array(image.convert("L"), dtype=np.float32)

    # Apply FFT (Fast Fourier Transform)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)  # Center the zero frequency
    magnitude = np.abs(fft_shifted)

    # Create frequency masks for different frequency bands
    center_y, center_x = np.array(magnitude.shape) // 2
    y, x = np.ogrid[: magnitude.shape[0], : magnitude.shape[1]]

    # Calculate distance from center (frequency)
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x**2 + center_y**2)

    # Define frequency bands (normalized 0-1)
    low_freq_mask = distance < (max_distance * 0.1)  # Inner 10%
    mid_freq_mask = (distance >= (max_distance * 0.1)) & (
        distance < (max_distance * 0.4)
    )
    high_freq_mask = distance >= (max_distance * 0.4)  # Outer 60%

    # Calculate energy in each band
    total_energy = np.sum(magnitude**2)
    low_freq_energy = np.sum((magnitude * low_freq_mask) ** 2) / total_energy
    mid_freq_energy = np.sum((magnitude * mid_freq_mask) ** 2) / total_energy
    high_freq_energy = np.sum((magnitude * high_freq_mask) ** 2) / total_energy

    # Find dominant frequency
    dominant_freq = np.unravel_index(
        np.argmax(magnitude[1:, 1:]), magnitude[1:, 1:].shape
    )
    dominant_freq_dist = (
        np.sqrt(dominant_freq[0] ** 2 + dominant_freq[1] ** 2) / max_distance
    )

    return {
        "low_freq_energy": float(low_freq_energy),
        "mid_freq_energy": float(mid_freq_energy),
        "high_freq_energy": float(high_freq_energy),
        "dominant_frequency": float(dominant_freq_dist),
    }


def detect_artifacts(image: Image.Image) -> dict[str, Any]:
    """
    Detect common artifacts in generated textures.

    Common problems in procedural textures:
    - Repeating patterns (not random enough)
    - Banding (visible steps in gradients)
    - Aliasing (jagged edges)
    - Dead pixels or extreme values

    Args:
        image: Input PIL Image

    Returns:
        Dictionary with artifact detection results
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    gray_uint8 = gray.astype(np.uint8)

    artifacts = {
        "has_banding": _detect_banding(gray_uint8),
        "has_aliasing": _detect_aliasing(gray_uint8),
        "has_dead_pixels": _detect_dead_pixels(gray_uint8),
        "pattern_repetition": _detect_pattern_repetition(gray_uint8),
        "dynamic_range": _check_dynamic_range(gray_uint8),
    }

    # Overall artifact score (0 = no artifacts, 1 = many artifacts)
    artifact_count = sum(
        [
            artifacts["has_banding"],
            artifacts["has_aliasing"],
            artifacts["has_dead_pixels"],
            artifacts["pattern_repetition"] > 0.8,  # High repetition is bad
            artifacts["dynamic_range"] < 0.3,  # Low dynamic range is bad
        ]
    )

    artifacts["artifact_score"] = artifact_count / 5.0
    rating: str = (
        "excellent"
        if artifact_count == 0
        else (
            "good" if artifact_count <= 1 else "fair" if artifact_count <= 2 else "poor"
        )
    )
    artifacts["quality_rating"] = rating  # type: ignore[assignment]

    return artifacts


def _analyze_detail_content(gray_array: NDArray[np.uint8]) -> float:
    """Analyze the amount of fine detail in the image."""
    # Use high-pass filter to isolate fine details
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    high_pass = cv2.filter2D(gray_array.astype(np.float32), -1, kernel)

    # Measure the energy in high-frequency content
    detail_energy = float(np.std(high_pass.astype(np.float64)))

    # Normalize to 0-1 range
    return min(1.0, detail_energy / 20.0)


def _calculate_uniformity(gray_array: NDArray[np.uint8]) -> float:
    """Calculate how uniformly features are distributed across the image."""
    # Divide image into blocks and analyze variance between blocks
    block_size = 32
    height, width = gray_array.shape

    block_means = []
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            block = gray_array[y : y + block_size, x : x + block_size]
            block_means.append(float(np.mean(block)))

    # Low variance between blocks = high uniformity
    if len(block_means) < 2:
        return 1.0

    variance = float(np.var(np.array(block_means)))
    # Normalize: lower variance = higher uniformity
    uniformity = 1.0 / (1.0 + variance / 100.0)

    return uniformity


def _detect_banding(gray_array: NDArray[np.uint8]) -> bool:
    """Detect banding artifacts (visible steps in gradients)."""
    # Look for horizontal and vertical lines with similar values
    # This is a simplified detector - real banding detection is complex

    # Check for repeated values in gradients
    grad_x = np.diff(gray_array.astype(np.int16), axis=1)
    grad_y = np.diff(gray_array.astype(np.int16), axis=0)

    # Count zero gradients (flat areas that might indicate banding)
    zero_grad_x = float(np.sum(np.abs(grad_x) < 1)) / float(grad_x.size)
    zero_grad_y = float(np.sum(np.abs(grad_y) < 1)) / float(grad_y.size)

    # High percentage of zero gradients suggests banding
    return bool((zero_grad_x > 0.3) or (zero_grad_y > 0.3))


def _detect_aliasing(gray_array: NDArray[np.uint8]) -> bool:
    """Detect aliasing artifacts (jagged edges, moiré patterns)."""
    # Use Laplacian to detect sharp edges
    laplacian = cv2.Laplacian(gray_array, cv2.CV_32F)

    # Count pixels with very high Laplacian values (sharp transitions)
    high_edge_pixels = float(np.sum(np.abs(laplacian) > 50)) / float(laplacian.size)

    # Too many sharp transitions suggest aliasing
    return bool(high_edge_pixels > 0.1)


def _detect_dead_pixels(gray_array: NDArray[np.uint8]) -> bool:
    """Detect completely black or white pixels that might be errors."""
    # Count pixels at extremes
    dead_black = int(np.sum(gray_array == 0))
    dead_white = int(np.sum(gray_array == 255))
    total_pixels = int(gray_array.size)

    # More than 1% dead pixels is suspicious
    dead_ratio = float(dead_black + dead_white) / float(total_pixels)
    return bool(dead_ratio > 0.01)


def _detect_pattern_repetition(gray_array: NDArray[np.uint8]) -> float:
    """Detect how much the pattern repeats (0 = no repetition, 1 = perfect repetition)."""
    # This is a simplified version - real pattern detection is very complex
    # Use autocorrelation to find self-similarity

    # Downsample for speed
    small = cv2.resize(gray_array, (64, 64))

    # Calculate autocorrelation
    correlation = cv2.matchTemplate(small, small, cv2.TM_CCOEFF_NORMED)

    # Remove the center peak (perfect self-match)
    center = correlation.shape[0] // 2, correlation.shape[1] // 2
    correlation[center[0] - 2 : center[0] + 3, center[1] - 2 : center[1] + 3] = 0

    # Find the maximum correlation (indicates repetition)
    max_correlation = float(np.max(correlation))

    return max_correlation


def _check_dynamic_range(gray_array: NDArray[np.uint8]) -> float:
    """Check the dynamic range of the image (0 = no range, 1 = full range)."""
    min_val = float(np.min(gray_array))
    max_val = float(np.max(gray_array))

    # Dynamic range as fraction of full 0-255 range
    return (max_val - min_val) / 255.0
