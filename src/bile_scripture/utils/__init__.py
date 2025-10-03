"""
Bile Scripture Utilities Package

This package contains core utilities for texture processing, validation, and quality assessment.
"""

__version__ = "0.1.0"

from .config_utils import get_export_path, load_config, validate_environment

# Import main utility functions for easy access
from .image_utils import (
    calculate_seam_error,
    create_noise_texture,
    enhance_contrast,
    ensure_tileable,
    generate_normal_map,
    validate_tiling,
)
from .quality_metrics import (
    analyze_frequency_content,
    calculate_texture_quality,
    detect_artifacts,
)

__all__ = [
    "ensure_tileable",
    "validate_tiling",
    "calculate_seam_error",
    "create_noise_texture",
    "enhance_contrast",
    "generate_normal_map",
    "calculate_texture_quality",
    "analyze_frequency_content",
    "detect_artifacts",
    "load_config",
    "validate_environment",
    "get_export_path",
]
