"""
Data ingestion, noise filtration, and causal alignment modules.
"""

from .ingest_news import ingest_historical_news
from .noise_filter import apply_4stage_noise_funnel
from .align_targets import align_causal_targets

__all__ = [
    "ingest_historical_news",
    "apply_4stage_noise_funnel",
    "align_causal_targets",
]
