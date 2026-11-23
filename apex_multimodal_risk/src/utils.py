"""
Utility functions for APEX Multimodal Risk Overlay.
Device management, configuration loading, and DuckDB interface.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_logger(name: str = "apex_risk", level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_device() -> torch.device:
    """
    Selects the optimal hardware acceleration device:
    1. Apple Silicon MPS (Metal Performance Shaders) if available.
    2. NVIDIA CUDA if available.
    3. CPU fallback.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads pipeline configuration from YAML file."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "apex_multimodal_risk" / "configs" / "pipeline.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_duckdb_connection(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Returns an in-memory or file-backed DuckDB connection."""
    if db_path is None:
        return duckdb.connect(database=":memory:")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return duckdb.connect(database=db_path)
