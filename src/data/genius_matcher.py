"""
Compatibility wrapper that delegates lyrics matching to `fetch_lyrics_lrclib`
in `src.data.lrclib_fetcher`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.data.lrclib_fetcher import DEFAULT_RAW_DIR, fetch_lyrics_lrclib


def match_spotify_to_genius(
    spotify_df: pd.DataFrame,
    cache_path: Optional[Path] = DEFAULT_RAW_DIR / "lrclib_lyrics_cache.parquet",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Delegate directly to the fast LRCLIB fetcher."""
    return fetch_lyrics_lrclib(spotify_df=spotify_df, cache_path=cache_path)
