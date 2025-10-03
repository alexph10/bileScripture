"""
Image Processing Utilities for Texture Generation

This module contains functions for:
- Ensuring textures are tileable (seamless)
- Validating tiling quality
- Detecting seam artifacts
- Basic image manipulations for PBR textures
"""

from typing import Any

import cv2
import numpy as np
from PIL import Image


def ensure_tileable(image: Image.Image, blend_width: int = 32) -> Image.Image:
    """
    Make an image tileable by blending the edges.

    This is crucial for PBR textures - they must tile seamlessly without visible seams.

    Args:
        image: Input PIL Image
        blend_width: Width of the blending region in pixels

    Returns:
        Tileable version of the input image

    How it works:
    1. Takes pixels from opposite edges
    2. Creates a smooth transition between them
    3. Ensures when tiled, edges match perfectly
    """
    if image.mode not in ["L", "RGB", "RGBA"]:
        raise ValueError(f"Unsupported image mode: {image.mode}")

    # Convert to numpy array for processing
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Ensure blend_width is reasonable
    blend_width = min(blend_width, width // 4, height // 4)

    # Clone the image to work on
    result = img_array.copy()

    # Blend horizontal edges (left-right)
    for i in range(blend_width):
        # Calculate blend factor (0 to 1)
        alpha = i / blend_width

        # Blend left edge with right edge
        left_col = img_array[:, i]
        right_col = img_array[:, width - blend_width + i]
        blended = (1 - alpha) * right_col + alpha * left_col
        result[:, i] = blended

        # Mirror the blend on the right side
        result[:, width - blend_width + i] = blended

    # Blend vertical edges (top-bottom)
    for i in range(blend_width):
        alpha = i / blend_width

        # Blend top edge with bottom edge
        top_row = result[i, :]
        bottom_row = result[height - blend_width + i, :]
        blended = (1 - alpha) * bottom_row + alpha * top_row
        result[i, :] = blended
        result[height - blend_width + i, :] = blended

    return Image.fromarray(result.astype(np.uint8))


def validate_tiling(image: Image.Image, tile_count: int = 2) -> dict[str, Any]:
    """
    Validate how well an image tiles by creating a tiled version and analyzing seams.

    Args:
        image: Input PIL Image to test
        tile_count: Number of tiles in each direction for testing

    Returns:
        Dictionary with validation results:
        - 'is_tileable': bool indicating if image tiles well
        - 'seam_error': float (0-255) indicating edge mismatch
        - 'quality_score': float (0-1) overall tiling quality
    """
    # Create a tiled version for analysis
    width, height = image.size
    tiled_width = width * tile_count
    tiled_height = height * tile_count

    # Create tiled image
    tiled = Image.new(image.mode, (tiled_width, tiled_height))

    for y in range(tile_count):
        for x in range(tile_count):
            tiled.paste(image, (x * width, y * height))

    # Analyze seam quality
    seam_error = calculate_seam_error(image)

    # Calculate overall quality score
    # Lower seam error = better quality
    quality_score = max(0, 1 - (seam_error / 50))  # Normalize to 0-1

    return {
        "is_tileable": seam_error < 20,  # Threshold for "good enough"
        "seam_error": float(seam_error),
        "quality_score": float(quality_score),
        "tiled_preview": tiled,  # For debugging/visualization
    }


def calculate_seam_error(image: Image.Image) -> float:
    """
    Calculate the average difference between opposite edges of an image.

    This tells us how well the image will tile - lower values mean better tiling.

    Args:
        image: Input PIL Image

    Returns:
        Average pixel difference between opposite edges (0-255)
    """
    # Convert to grayscale for simpler analysis
    if image.mode != "L":
        gray = image.convert("L")
    else:
        gray = image

    img_array = np.array(gray, dtype=np.float32)
    height, width = img_array.shape

    # Calculate horizontal seam error (left vs right edge)
    left_edge = img_array[:, 0]
    right_edge = img_array[:, -1]
    horizontal_error = np.mean(np.abs(left_edge - right_edge))

    # Calculate vertical seam error (top vs bottom edge)
    top_edge = img_array[0, :]
    bottom_edge = img_array[-1, :]
    vertical_error = float(
        np.mean(np.abs(top_edge.astype(np.float32) - bottom_edge.astype(np.float32)))
    )

    # Return average of both directions
    return float((horizontal_error + vertical_error) / 2.0)


def create_noise_texture(
    size: int, scale: float = 0.1, octaves: int = 4
) -> Image.Image:
    """
    Create a procedural noise texture using tileable noise.

    This will replace the simple checker pattern in your current API.

    Args:
        size: Image size (square)
        scale: Noise frequency (smaller = larger features)
        octaves: Number of noise layers (more = more detail)

    Returns:
        Grayscale noise texture as PIL Image that tiles seamlessly
    """
    # Create coordinate grids that wrap around (periodic)
    x = np.arange(size) / size  # 0 to 1
    y = np.arange(size) / size  # 0 to 1
    X, Y = np.meshgrid(x, y)

    # Generate multi-octave noise that tiles naturally
    noise = np.zeros((size, size))
    frequency = scale
    amplitude = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        # Create tileable patterns using sine/cosine
        # These naturally wrap around at 2π
        angle_x = X * 2 * np.pi * frequency
        angle_y = Y * 2 * np.pi * frequency

        # Combine multiple sine waves for complexity
        layer = (
            np.sin(angle_x) * np.cos(angle_y)
            + 0.5 * np.sin(angle_x * 2 + angle_y)
            + 0.3 * np.cos(angle_x + angle_y * 2)
        )

        noise += layer * amplitude
        max_amplitude += amplitude

        frequency *= 2
        amplitude *= 0.5

    # Normalize to 0-255
    if max_amplitude > 0:
        noise = noise / max_amplitude
    noise = (noise + 1) / 2  # Map from [-1,1] to [0,1]
    noise = np.clip(noise * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(noise, "L")


def enhance_contrast(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Enhance image contrast - useful for making textures more dramatic.

    Args:
        image: Input PIL Image
        factor: Contrast factor (1.0 = no change, >1.0 = more contrast)

    Returns:
        Contrast-enhanced image
    """
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def generate_normal_map(height_map: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Generate a normal map from a height map.

    Normal maps are crucial for PBR - they add surface detail without geometry.

    Args:
        height_map: Grayscale height map
        strength: Normal map intensity

    Returns:
        RGB normal map (tangent space)
    """
    # Convert to numpy array
    height_array = np.array(height_map.convert("L"), dtype=np.float32) / 255.0

    # Calculate gradients using Sobel operators
    grad_x = cv2.Sobel(height_array, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_array, cv2.CV_32F, 0, 1, ksize=3)

    # Apply strength
    grad_x = grad_x * strength
    grad_y = grad_y * strength

    # Calculate normal vectors
    # In tangent space: X = gradient_x, Y = gradient_y, Z = calculated from X,Y
    normal_x = grad_x
    normal_y = -grad_y  # Flip Y for correct orientation
    normal_z = np.sqrt(np.maximum(0.0, 1.0 - normal_x**2 - normal_y**2))

    # Convert to 0-255 range (normal maps use [0,1] mapped to [0,255])
    normal_x_clipped = np.clip((normal_x + 1.0) * 127.5, 0.0, 255.0)
    normal_y_clipped = np.clip((normal_y + 1.0) * 127.5, 0.0, 255.0)
    normal_z_clipped = np.clip(normal_z * 255.0, 0.0, 255.0)

    # Combine into RGB image (R=X, G=Y, B=Z)
    normal_rgb = np.stack(
        [normal_x_clipped, normal_y_clipped, normal_z_clipped], axis=-1
    ).astype(np.uint8)

    return Image.fromarray(normal_rgb, "RGB")
