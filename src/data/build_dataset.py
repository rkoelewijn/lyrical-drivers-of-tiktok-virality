"""
Phase 1 Pipeline Entrypoint:
1. Scrape curated Viral (label=1) and Non-Viral (label=0) Spotify playlists (2020-2025).
2. Fetch plain lyrics concurrently via the LRCLIB API with strict exact matching on `(norm_artist, norm_title)`.
3. Log progress toward the 3,000 Viral / 3,000 Non-Viral target.
4. Build the balanced dataset and create the stratified 80/20 train-test split.

Usage:
    python -m src.data.build_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.lrclib_fetcher import fetch_lyrics_lrclib
from src.data.spotify_scraper import (
    DEFAULT_NON_VIRAL_CONFIG,
    DEFAULT_RAW_DIR,
    DEFAULT_VIRAL_CONFIG,
    PROJECT_ROOT,
    run_spotify_scraper,
)

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def print_collection_report(
    viral_spotify: pd.DataFrame,
    non_viral_spotify: pd.DataFrame,
    viral_matched: pd.DataFrame,
    non_viral_matched: pd.DataFrame,
    target_per_class: int = 3000,
) -> None:
    """Print a clear progress report toward the target 3,000 Viral / 3,000 Non-Viral dataset."""
    v_sp = len(viral_spotify)
    nv_sp = len(non_viral_spotify)
    v_m = len(viral_matched)
    nv_m = len(non_viral_matched)

    v_rate = (v_m / v_sp * 100.0) if v_sp else 0.0
    nv_rate = (nv_m / nv_sp * 100.0) if nv_sp else 0.0

    print(f"\n{'=' * 78}")
    print("PHASE 1 PROGRESS REPORT (LRCLIB Matched Lyrics Corpus)")
    print(f"{'=' * 78}")
    print(
        f"  Viral (label=1)     : {v_sp:5d} Spotify -> {v_m:5d} LRCLIB matched "
        f"({v_rate:5.1f}%) | Target: {target_per_class} ({max(0, target_per_class - v_m)} more needed)"
    )
    print(
        f"  Non-Viral (label=0) : {nv_sp:5d} Spotify -> {nv_m:5d} LRCLIB matched "
        f"({nv_rate:5.1f}%) | Target: {target_per_class} ({max(0, target_per_class - nv_m)} more needed)"
    )
    print(f"{'=' * 78}\n")


def build_balanced_splits(
    viral_matched: pd.DataFrame,
    non_viral_matched: pd.DataFrame,
    target_per_class: int = 3000,
    test_size: float = 0.20,
    random_state: int = 42,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sample up to `target_per_class` songs per class (prioritizing tracks that appeared
    most frequently across curated playlists, then random tie-breaking), and perform a
    stratified 80/20 train-test split.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    if viral_matched.empty or non_viral_matched.empty:
        print("[INFO] Both Viral and Non-Viral matched sets must be non-empty to create train/test splits.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _select_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df.copy()
        shuffled = df.sample(frac=1.0, random_state=random_state)
        return shuffled.sort_values("playlist_frequency", ascending=False, kind="mergesort").head(n).copy()

    v_selected = _select_top_n(viral_matched, target_per_class)
    nv_selected = _select_top_n(non_viral_matched, target_per_class)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Viral vs. Non-Viral lyrics dataset via LRCLIB (Phase 1).")
    parser.add_argument("--target", type=int, default=3000, help="Target number of songs per class (default: 3000)")
    parser.add_argument("--min-year", type=int, default=2020, help="Minimum release year (default: 2020)")
    parser.add_argument("--max-year", type=int, default=2025, help="Maximum release year (default: 2025)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent LRCLIB worker threads (default: 8)")
    parser.add_argument(
        "--skip-spotify",
        action="store_true",
        help="Reuse already-scraped parquet files in data/raw/ instead of re-scraping playlists",
    )
    args = parser.parse_args()

    if args.skip_spotify:
        v_path = DEFAULT_RAW_DIR / "spotify_viral_2020_2025.parquet"
        nv_path = DEFAULT_RAW_DIR / "spotify_non_viral_2020_2025.parquet"
        viral_sp = pd.read_parquet(v_path) if v_path.exists() else pd.DataFrame()
        non_viral_sp = pd.read_parquet(nv_path) if nv_path.exists() else pd.DataFrame()
    else:
        viral_sp, non_viral_sp = run_spotify_scraper(
            viral_config=DEFAULT_VIRAL_CONFIG,
            non_viral_config=DEFAULT_NON_VIRAL_CONFIG,
            output_dir=DEFAULT_RAW_DIR,
            min_year=args.min_year,
            max_year=args.max_year,
            target_per_class=args.target,
        )

    combined_sp = pd.concat([viral_sp, non_viral_sp], ignore_index=True)
    if combined_sp.empty:
        print(
            "\n[NEXT STEP] Add Spotify playlist URLs/IDs to:\n"
            f"  1. {DEFAULT_VIRAL_CONFIG}\n"
            f"  2. {DEFAULT_NON_VIRAL_CONFIG}\n"
            "Then re-run: python -m src.data.build_dataset"
        )
        return

    matched_all, unmatched_all = fetch_lyrics_lrclib(
        spotify_df=combined_sp,
        max_workers=args.workers,
    )

    if not unmatched_all.empty:
        unmatched_out = DEFAULT_RAW_DIR / "unmatched_spotify_tracks.csv"
        unmatched_all.to_csv(unmatched_out, index=False, encoding="utf-8")
        print(f"[Diagnostics] Saved {len(unmatched_all)} unmatched Spotify tracks to {unmatched_out}")

    viral_matched = matched_all[matched_all["label"] == 1].copy() if not matched_all.empty else pd.DataFrame()
    non_viral_matched = matched_all[matched_all["label"] == 0].copy() if not matched_all.empty else pd.DataFrame()

    print_collection_report(
        viral_spotify=viral_sp,
        non_viral_spotify=non_viral_sp,
        viral_matched=viral_matched,
        non_viral_matched=non_viral_matched,
        target_per_class=args.target,
    )

    build_balanced_splits(
        viral_matched=viral_matched,
        non_viral_matched=non_viral_matched,
        target_per_class=args.target,
    )


if __name__ == "__main__":
    main()
