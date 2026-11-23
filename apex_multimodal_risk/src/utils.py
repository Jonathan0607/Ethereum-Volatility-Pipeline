"""
Utility functions for APEX Multimodal Risk Overlay.
Device management, configuration loading, UTC timezone enforcement, and DuckDB interface.
"""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_env_root = os.environ.get("APEX_PROJECT_ROOT")
PROJECT_ROOT = Path(_env_root).resolve() if _env_root else Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


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


def get_device():
    """
    Selects the optimal hardware acceleration device:
    1. Apple Silicon MPS (Metal Performance Shaders) if available.
    2. NVIDIA CUDA if available.
    3. CPU fallback.
    """
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Loads pipeline configuration from YAML file."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "apex_multimodal_risk" / "configs" / "pipeline.yaml")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_duckdb_connection(db_path: str | None = None) -> Any:
    """Returns an in-memory or file-backed DuckDB connection with UTC timezone enforced."""
    import duckdb

    if db_path is None:
        con = duckdb.connect(database=":memory:")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        con = duckdb.connect(database=db_path)
    con.execute("SET TimeZone = 'UTC'")
    return con


def enforce_utc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all datetime and datetimetz columns in a DataFrame to UTC (+00:00)."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
            else:
                df[col] = df[col].dt.tz_convert("UTC")
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    return df


def read_parquet_utc(
    path: str | Path, con: Any = None, order_by: str | None = None
) -> pd.DataFrame:
    """Reads Parquet via DuckDB and guarantees all datetime columns are strictly UTC."""
    p = Path(path)
    if not p.is_absolute():
        if (PROJECT_ROOT / p).exists():
            p = PROJECT_ROOT / p
        elif (PACKAGE_ROOT / p).exists():
            p = PACKAGE_ROOT / p
    if con is None:
        con = get_duckdb_connection()
    sql = f"SELECT * FROM '{p}'"
    if order_by:
        sql += f" ORDER BY {order_by}"
    df = con.execute(sql).df()
    return enforce_utc_columns(df)
