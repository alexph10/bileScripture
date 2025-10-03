"""
Test suite for bile_scripture utilities.

This demonstrates how to use the utilities we just created and ensures they work correctly.
"""

import os
import tempfile

import numpy as np
from PIL import Image

from bile_scripture.utils import (
    analyze_frequency_content,
    calculate_seam_error,
    calculate_texture_quality,
    create_noise_texture,
    detect_artifacts,
    ensure_tileable,
    generate_normal_map,
    get_export_path,
    load_config,
    validate_environment,
    validate_tiling,
)


class TestImageUtils:
    """Test image processing utilities."""

    def test_create_noise_texture(self) -> None:
        """Test noise texture generation."""
        texture = create_noise_texture(size=128, scale=0.1, octaves=4)

        assert texture.size == (128, 128)
        assert texture.mode == "L"  # Grayscale

        # Check that it's not all the same value
        texture_array = np.array(texture)
        assert np.std(texture_array) > 5  # Should have some variation

    def test_ensure_tileable(self) -> None:
        """Test making an image tileable."""
        # Create a test image with distinct edges
        test_img = Image.new("RGB", (64, 64), "white")
        # Add some pattern that would create seams
        pixels = test_img.load()
        for x in range(64):
            pixels[x, 0] = (255, 0, 0)  # Red top edge
            pixels[0, x] = (0, 255, 0)  # Green left edge
            pixels[x, 63] = (0, 0, 255)  # Blue bottom edge
            pixels[63, x] = (255, 255, 0)  # Yellow right edge

        # Make it tileable
        tileable = ensure_tileable(test_img, blend_width=8)

        assert tileable.size == test_img.size
        assert tileable.mode == test_img.mode

        # Test that seam error is reduced
        original_error = calculate_seam_error(test_img)
        tileable_error = calculate_seam_error(tileable)
        assert tileable_error < original_error

    def test_validate_tiling(self) -> None:
        """Test tiling validation."""
        # Create a simple tileable pattern
        tileable_img = Image.new("L", (32, 32), 128)  # Gray image

        result = validate_tiling(tileable_img, tile_count=2)

        assert "is_tileable" in result
        assert "seam_error" in result
        assert "quality_score" in result
        assert "tiled_preview" in result

        # Should be considered tileable (low seam error)
        assert result["seam_error"] < 5
        assert result["is_tileable"] is True

    def test_generate_normal_map(self) -> None:
        """Test normal map generation from height map."""
        # Create a simple gradient as height map
        height_map = Image.new("L", (64, 64))
        pixels = height_map.load()
        for y in range(64):
            for x in range(64):
                pixels[x, y] = int(x * 4)  # Simple gradient

        normal_map = generate_normal_map(height_map, strength=1.0)

        assert normal_map.size == height_map.size
        assert normal_map.mode == "RGB"

        # Check that we have variation in the normal map
        normal_array = np.array(normal_map)
        assert np.std(normal_array) > 10


class TestQualityMetrics:
    """Test quality assessment utilities."""

    def test_calculate_texture_quality(self) -> None:
        """Test texture quality calculation."""
        # Create a test texture
        texture = create_noise_texture(128)

        quality = calculate_texture_quality(texture)

        # Check all expected metrics are present
        expected_keys = [
            "sharpness",
            "contrast",
            "detail_level",
            "uniformity",
            "overall_score",
        ]
        for key in expected_keys:
            assert key in quality
            assert 0 <= quality[key] <= 1  # All scores should be normalized

    def test_analyze_frequency_content(self) -> None:
        """Test frequency analysis."""
        texture = create_noise_texture(128)

        freq_analysis = analyze_frequency_content(texture)

        expected_keys = [
            "low_freq_energy",
            "mid_freq_energy",
            "high_freq_energy",
            "dominant_frequency",
        ]
        for key in expected_keys:
            assert key in freq_analysis
            assert freq_analysis[key] >= 0

        # Energy should sum to approximately 1
        total_energy = (
            freq_analysis["low_freq_energy"]
            + freq_analysis["mid_freq_energy"]
            + freq_analysis["high_freq_energy"]
        )
        assert 0.8 <= total_energy <= 1.2  # Allow some numerical error

    def test_detect_artifacts(self) -> None:
        """Test artifact detection."""
        # Create a clean texture
        clean_texture = create_noise_texture(128)

        artifacts = detect_artifacts(clean_texture)

        expected_keys = [
            "has_banding",
            "has_aliasing",
            "has_dead_pixels",
            "pattern_repetition",
            "dynamic_range",
            "artifact_score",
            "quality_rating",
        ]
        for key in expected_keys:
            assert key in artifacts

        # Clean texture should have low artifact score
        assert artifacts["artifact_score"] < 0.5
        assert artifacts["quality_rating"] in ["excellent", "good", "fair", "poor"]


class TestConfigUtils:
    """Test configuration management."""

    def test_load_config_defaults(self) -> None:
        """Test loading default configuration."""
        config = load_config()

        # Check that all required attributes exist
        assert hasattr(config, "export_dir")
        assert hasattr(config, "model_path")
        assert hasattr(config, "default_texture_size")

        # Check reasonable defaults
        assert config.default_texture_size == 512
        assert config.blend_width == 32

    def test_validate_environment(self) -> None:
        """Test environment validation."""
        validation = validate_environment()

        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert "info" in validation

        # Should be able to create export directory
        assert "export_dir" in validation["info"]

    def test_get_export_path(self) -> None:
        """Test export path creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set temporary export directory
            os.environ["BS_EXPORT_DIR"] = temp_dir

            try:
                export_path = get_export_path("test_texture")

                assert export_path.exists()
                assert export_path.is_dir()
                assert export_path.name == "test_texture"
                assert export_path.parent.samefile(temp_dir)
            finally:
                # Clean up environment variable
                if "BS_EXPORT_DIR" in os.environ:
                    del os.environ["BS_EXPORT_DIR"]


def test_integration_example() -> None:
    """
    Integration test showing how to use multiple utilities together.

    This demonstrates a complete workflow:
    1. Generate a texture
    2. Validate its tiling quality
    3. Analyze its quality
    4. Generate PBR maps
    """
    # Step 1: Generate base texture with good parameters for tiling
    albedo = create_noise_texture(size=256, scale=2.0, octaves=4)

    # Step 2: Validate tiling (should already be good since we made it tileable)
    tiling_result = validate_tiling(albedo)
    print(f"Original texture - Seam error: {tiling_result['seam_error']:.2f}")

    # If it's not tileable enough, apply our tiling fix
    if not tiling_result["is_tileable"]:
        print("Applying tiling fix...")
        albedo = ensure_tileable(albedo, blend_width=16)
        tiling_result = validate_tiling(albedo)
        print(f"After fix - Seam error: {tiling_result['seam_error']:.2f}")

    # Step 3: Analyze quality
    quality = calculate_texture_quality(albedo)
    assert quality["overall_score"] > 0.2  # Reasonable threshold

    # Step 4: Generate normal map
    generate_normal_map(albedo, strength=0.8)

    # Step 5: Check for artifacts
    artifacts = detect_artifacts(albedo)

    # Should be a decent texture
    assert artifacts["artifact_score"] < 0.8  # More lenient threshold

    print(f"✅ Generated texture quality: {quality['overall_score']:.2f}")
    print(f"✅ Tiling error: {tiling_result['seam_error']:.2f}")
    print(f"✅ Artifact score: {artifacts['artifact_score']:.2f}")
    print("✅ All texture generation steps completed successfully!")


if __name__ == "__main__":
    # Run the integration test if script is executed directly
    test_integration_example()
    print("All tests passed! 🎉")
