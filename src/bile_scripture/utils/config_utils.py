"""
Configuration and Environment Management Utilities

This module handles:
- Loading and validating configuration
- Environment variable management
- Path resolution
- Settings validation
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BileScriptureConfig:
    """
    Configuration class for Bile Scripture.

    This centralizes all configuration in one place and provides type hints.
    """

    # Paths
    export_dir: Path
    model_path: Path
    data_dir: Path

    # Generation settings
    default_texture_size: int = 512
    default_tile_size: int = 64
    blend_width: int = 32

    # Quality settings
    min_quality_score: float = 0.6
    max_seam_error: float = 20.0

    # Model settings
    model_enabled: bool = False
    gpu_enabled: bool = False
    batch_size: int = 1

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug_mode: bool = False


def load_config(config_path: str | Path | None = None) -> BileScriptureConfig:
    """
    Load configuration from environment variables and optional config file.

    Priority order:
    1. Environment variables (highest priority)
    2. Config file (if provided)
    3. Default values (lowest priority)

    Args:
        config_path: Optional path to JSON config file

    Returns:
        BileScriptureConfig object with all settings
    """
    # Start with defaults
    config_dict = _get_default_config()

    # Load from config file if provided
    if config_path:
        file_config = _load_config_file(config_path)
        config_dict.update(file_config)

    # Override with environment variables
    env_config = _load_env_config()
    config_dict.update(env_config)

    # Convert paths to Path objects
    config_dict["export_dir"] = Path(config_dict["export_dir"])
    config_dict["model_path"] = Path(config_dict["model_path"])
    config_dict["data_dir"] = Path(config_dict["data_dir"])

    # Create and validate config
    config = BileScriptureConfig(**config_dict)
    _validate_config(config)

    return config


def validate_environment() -> dict[str, Any]:
    """
    Validate that the environment is properly set up for Bile Scripture.

    Checks:
    - Required directories exist or can be created
    - Required packages are available
    - GPU availability (if requested)
    - File permissions

    Returns:
        Dictionary with validation results
    """
    results: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }

    config = load_config()

    # Check directory access
    try:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        results["info"]["export_dir"] = str(config.export_dir.absolute())
    except PermissionError:
        results["errors"].append(f"Cannot create export directory: {config.export_dir}")
        results["valid"] = False

    # Check model availability
    if config.model_enabled:
        if not config.model_path.exists():
            results["warnings"].append(f"Model file not found: {config.model_path}")
        else:
            results["info"]["model_available"] = True

    # Check GPU availability
    gpu_available = _check_gpu_availability()
    results["info"]["gpu_available"] = gpu_available

    if config.gpu_enabled and not gpu_available:
        results["warnings"].append(
            "GPU requested but not available, falling back to CPU"
        )

    # Check required packages
    missing_packages = _check_required_packages()
    if missing_packages:
        results["errors"].extend(
            [f"Missing package: {pkg}" for pkg in missing_packages]
        )
        results["valid"] = False

    return results


def get_export_path(name: str, config: BileScriptureConfig | None = None) -> Path:
    """
    Get the full export path for a given texture pack name.

    Args:
        name: Name of the texture pack
        config: Optional config object (will load if not provided)

    Returns:
        Full path to the export directory for this texture pack
    """
    if config is None:
        config = load_config()

    export_path = config.export_dir / name
    export_path.mkdir(parents=True, exist_ok=True)

    return export_path


def setup_logging(debug: bool = False) -> None:
    """
    Set up logging for the application.

    Args:
        debug: Enable debug-level logging
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("bile_scripture.log"),
        ],
    )

    # Suppress some noisy loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _get_default_config() -> dict[str, Any]:
    """Get default configuration values."""
    cwd = Path.cwd()

    return {
        "export_dir": cwd / "export",
        "model_path": cwd / "models" / "unet.onnx",
        "data_dir": cwd / "data",
        "default_texture_size": 512,
        "default_tile_size": 64,
        "blend_width": 32,
        "min_quality_score": 0.6,
        "max_seam_error": 20.0,
        "model_enabled": False,
        "gpu_enabled": False,
        "batch_size": 1,
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "debug_mode": False,
    }


def _load_config_file(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path) as f:
            data = json.load(f)
            # Ensure we return a dictionary
            if isinstance(data, dict):
                return data
            else:
                logging.error(
                    f"Config file {config_path} does not contain a JSON object"
                )
                return {}
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file {config_path}: {e}")
        return {}


def _load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}

    # Map environment variable names to config keys
    env_mapping = {
        "BS_EXPORT_DIR": "export_dir",
        "BS_MODEL_PATH": "model_path",
        "BS_DATA_DIR": "data_dir",
        "BS_TEXTURE_SIZE": ("default_texture_size", int),
        "BS_TILE_SIZE": ("default_tile_size", int),
        "BS_BLEND_WIDTH": ("blend_width", int),
        "BS_MIN_QUALITY": ("min_quality_score", float),
        "BS_MAX_SEAM_ERROR": ("max_seam_error", float),
        "BS_MODEL_ENABLED": ("model_enabled", lambda x: x.lower() == "true"),
        "BS_GPU_ENABLED": ("gpu_enabled", lambda x: x.lower() == "true"),
        "BS_BATCH_SIZE": ("batch_size", int),
        "BS_API_HOST": "api_host",
        "BS_API_PORT": ("api_port", int),
        "BS_DEBUG": ("debug_mode", lambda x: x.lower() == "true"),
    }

    for env_var, config_key in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            if isinstance(config_key, tuple):
                key, converter = config_key
                try:
                    config[key] = converter(value)
                except (ValueError, TypeError):
                    logging.warning(f"Invalid value for {env_var}: {value}")
            else:
                config[config_key] = value

    return config


def _validate_config(config: BileScriptureConfig) -> None:
    """Validate configuration values."""
    # Validate texture size
    if config.default_texture_size < 64 or config.default_texture_size > 4096:
        raise ValueError(f"Invalid texture size: {config.default_texture_size}")

    # Validate quality thresholds
    if not 0 <= config.min_quality_score <= 1:
        raise ValueError(f"Invalid min quality score: {config.min_quality_score}")

    if config.max_seam_error < 0:
        raise ValueError(f"Invalid max seam error: {config.max_seam_error}")

    # Validate API settings
    if not 1 <= config.api_port <= 65535:
        raise ValueError(f"Invalid API port: {config.api_port}")


def _check_gpu_availability() -> bool:
    """Check if GPU is available for computation."""
    try:
        import torch

        available = torch.cuda.is_available()
        return bool(available)
    except ImportError:
        return False


def _check_required_packages() -> list[str]:
    """Check for required packages and return list of missing ones."""
    required_packages = [
        "numpy",
        "PIL",
        "cv2",
        "scipy",
        "fastapi",
        "uvicorn",
        "pydantic",
    ]

    missing = []
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL  # noqa: F401
            elif package == "cv2":
                import cv2  # noqa: F401
            else:
                __import__(package)
        except ImportError:
            missing.append(package)

    return missing


def create_sample_config() -> None:
    """Create a sample configuration file for reference."""
    sample_config = {
        "export_dir": "./export",
        "model_path": "./models/unet.onnx",
        "data_dir": "./data",
        "default_texture_size": 512,
        "default_tile_size": 64,
        "blend_width": 32,
        "min_quality_score": 0.6,
        "max_seam_error": 20.0,
        "model_enabled": False,
        "gpu_enabled": False,
        "batch_size": 1,
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "debug_mode": False,
    }

    config_path = Path("config.json")
    with open(config_path, "w") as f:
        json.dump(sample_config, f, indent=2)

    print(f"Sample configuration written to {config_path}")


if __name__ == "__main__":
    # If run as script, create sample config and validate environment
    create_sample_config()

    config = load_config()
    print(f"Configuration loaded: {config}")

    validation = validate_environment()
    print(f"Environment validation: {validation}")
