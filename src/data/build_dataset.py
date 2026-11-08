"""
Build the finalized balanced 6,000-song corpus (3,000 TikTok-Viral vs. 3,000 Non-Viral
commercial music) with stratified 80/20 train/test splits, strictly filtered to
English songs released between 2020 and 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.lrclib_fetcher import (
    DEFAULT_RAW_DIR,
    fetch_lyrics_lrclib,
    is_english_lyrics,
)
from src.data.normalize import is_target_release_year, parse_release_year
from src.data.spotify_scraper import (
    DEFAULT_NON_VIRAL_CONFIG,
    DEFAULT_VIRAL_CONFIG,
    run_spotify_scraper,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def print_collection_report(
    *args,
    target_per_class: int = 3000,
    **kwargs,
) -> None:
    """
    Print an ASCII summary table comparing Spotify targets to English LRCLIB lyrics matches.
    Supports both:
      - print_collection_report(spotify_df, matched_df, target_per_class=3000)
      - print_collection_report(viral_sp, non_viral_sp, viral_matched, non_viral_matched, target_per_class=3000)
    """
    if len(args) == 4:
        v_sp = len(args[0])
        nv_sp = len(args[1])
        v_m = len(args[2])
        nv_m = len(args[3])
    elif len(args) >= 2:
        spotify_df, matched_df = args[0], args[1]
        v_sp = len(spotify_df[spotify_df["label"] == 1]) if "label" in spotify_df else 0
        nv_sp = len(spotify_df[spotify_df["label"] == 0]) if "label" in spotify_df else 0
        v_m = len(matched_df[matched_df["label"] == 1]) if "label" in matched_df else 0
        nv_m = len(matched_df[matched_df["label"] == 0]) if "label" in matched_df else 0
    else:
        return

    v_rate = (v_m / v_sp * 100) if v_sp else 0.0
    nv_rate = (nv_m / nv_sp * 100) if nv_sp else 0.0

    print(f"\n{'=' * 78}")
    print("PHASE 1 PROGRESS REPORT (English LRCLIB Matched Lyrics Corpus 2020-2026)")
    print(f"{'=' * 78}")
    print(
        f"  Viral (label=1)     : {v_sp:5d} Spotify -> {v_m:5d} English LRCLIB matched "
        f"({v_rate:5.1f}%) | Target: {target_per_class} ({max(0, target_per_class - v_m)} more needed)"
    )
    print(
        f"  Non-Viral (label=0) : {nv_sp:5d} Spotify -> {nv_m:5d} English LRCLIB matched "
        f"({nv_rate:5.1f}%) | Target: {target_per_class} ({max(0, target_per_class - nv_m)} more needed)"
    )
    print(f"{'=' * 78}\n")


def filter_english_and_year(
    df: pd.DataFrame,
    min_year: int = 2020,
    max_year: int = 2026,
    text_col: str = "lyrics",
) -> pd.DataFrame:
    """
    Ensure all tracks:
    1. Are English language lyrics (verifying via is_english_lyrics).
    2. Fall within [min_year, max_year] release dates if release_year or release_date is present.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # 1. English filter
    if text_col in filtered.columns:
        filtered = filtered[filtered[text_col].apply(is_english_lyrics)].copy()

    # 2. Release year filter (if date/year metadata exists)
    if "release_year" in filtered.columns:
        def _year_ok(row) -> bool:
            yr = row.get("release_year")
            if yr is None or pd.isna(yr):
                # Try parsing from release_date or album name
                r_date = row.get("release_date")
                yr = parse_release_year(r_date)
            if yr is not None and not pd.isna(yr):
                return bool(min_year <= int(yr) <= max_year)
            # If no explicit year is recorded on track, allow (playlist curation guarantees 2020-2026 window)
            return True

        filtered = filtered[filtered.apply(_year_ok, axis=1)].copy()

    return filtered


def build_balanced_splits(
    viral_matched: pd.DataFrame,
    non_viral_matched: pd.DataFrame,
    target_per_class: int = 3000,
    min_year: int = 2020,
    max_year: int = 2026,
    test_size: float = 0.20,
    random_state: int = 42,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter Viral and Non-Viral matched songs to English (2020-2026), sample up to
    `target_per_class` per class prioritizing playlist_frequency, and generate
    stratified 80/20 train/test splits.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    v_filtered = filter_english_and_year(viral_matched, min_year=min_year, max_year=max_year)
    nv_filtered = filter_english_and_year(non_viral_matched, min_year=min_year, max_year=max_year)

    if v_filtered.empty or nv_filtered.empty:
        print("[INFO] Both Viral and Non-Viral sets must be non-empty to create train/test splits.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _select_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df.copy()
        shuffled = df.sample(frac=1.0, random_state=random_state)
        return shuffled.sort_values("playlist_frequency", ascending=False, kind="mergesort").head(n).copy()

    actual_target = min(target_per_class, len(v_filtered), len(nv_filtered))
    if actual_target < target_per_class:
        print(
            f"[NOTICE] Available English tracks (Viral={len(v_filtered)}, Non-Viral={len(nv_filtered)}) "
            f"< target={target_per_class}. Building balanced splits with N={actual_target} per class ({actual_target * 2} total)."
        )

    v_selected = _select_top_n(v_filtered, actual_target)
    nv_selected = _select_top_n(nv_filtered, actual_target)

    full_df = pd.concat([v_selected, nv_selected], ignore_index=True)

    train_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df["label"],
        random_state=random_state,
    )

    full_path = processed_dir / "dataset_full.parquet"
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    full_df.to_parquet(full_path, index=False)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(
        f"[Dataset Saved] Full={len(full_df)} (Viral={len(v_selected)}, Non-Viral={len(nv_selected)}) | "
        f"Train (80%)={len(train_df)} | Test (20%)={len(test_df)} -> {processed_dir}"
    )

    return full_df, train_df, test_df
