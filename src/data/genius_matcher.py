"""
Strict exact matcher linking scraped Spotify tracks (2020-2025) to the
Hugging Face `theelderemo/genius-lyrics-cleaned` dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from src.data.normalize import is_target_release_year, make_match_key
from src.data.spotify_scraper import _load_env_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
HF_DATASET_ID = "theelderemo/genius-lyrics-cleaned"

# Candidate column names to auto-detect schema in the Hugging Face dataset
_TITLE_CANDIDATES = ("title", "song", "track_name", "name", "song_title")
_ARTIST_CANDIDATES = ("artist", "primary_artist", "artist_name", "artists", "performer")
_LYRICS_CANDIDATES = ("lyrics", "cleaned_lyrics", "text", "lyric", "body")
_YEAR_CANDIDATES = ("year", "release_year", "release_date", "date")
_LANG_CANDIDATES = ("language", "lang", "language_cld3", "language_ft")


def _detect_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def match_spotify_to_genius(
    spotify_df: pd.DataFrame,
    hf_dataset_id: str = HF_DATASET_ID,
    cache_path: Optional[Path] = DEFAULT_RAW_DIR / "genius_matched_cache.parquet",
    streaming: bool = True,
    enforce_genius_year: bool = False,
    min_year: int = 2020,
    max_year: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match a DataFrame of scraped Spotify tracks (containing `match_key`) against
    the Hugging Face Genius lyrics dataset using strict exact matching on
    `match_key = norm_artist::norm_title`.

    Maintains an incremental local cache at `cache_path` so previously matched
    songs do not need to be re-streamed when you add new playlists.

    Returns:
        (matched_df, unmatched_df)
    """
    if spotify_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    _load_env_file(PROJECT_ROOT / ".env")
    hf_token = os.getenv("HF_TOKEN") or None

    needed_keys: Set[str] = set(spotify_df["match_key"].dropna().unique())
    cached_matches: Dict[str, Dict] = {}

    if cache_path and cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        for row in cached_df.to_dict(orient="records"):
            cached_matches[row["match_key"]] = row
        print(f"[Genius Cache] Loaded {len(cached_matches)} previously matched songs from {cache_path.name}")

    missing_keys = needed_keys - set(cached_matches.keys())

    if missing_keys:
        from datasets import load_dataset

        print(
            f"[Genius Matcher] Scanning '{hf_dataset_id}' (streaming={streaming}) "
            f"for {len(missing_keys)} uncached Spotify songs..."
        )
        ds = load_dataset(hf_dataset_id, split="train", streaming=streaming, token=hf_token)

        first_row = next(iter(ds))
        cols = list(first_row.keys())
        title_col = _detect_column(cols, _TITLE_CANDIDATES)
        artist_col = _detect_column(cols, _ARTIST_CANDIDATES)
        lyrics_col = _detect_column(cols, _LYRICS_CANDIDATES)
        year_col = _detect_column(cols, _YEAR_CANDIDATES)
        lang_col = _detect_column(cols, _LANG_CANDIDATES)

        if not title_col or not artist_col or not lyrics_col:
            raise ValueError(
                f"Could not auto-detect title/artist/lyrics columns in {hf_dataset_id}. "
                f"Available columns: {cols}"
            )

        print(
            f"[Genius Schema] Detected columns -> artist='{artist_col}', "
            f"title='{title_col}', lyrics='{lyrics_col}', year='{year_col}', lang='{lang_col}'"
        )

        newly_found = 0
        for row in tqdm(ds, desc="Scanning Genius corpus", unit="songs"):
            if lang_col and row.get(lang_col):
                lang_val = str(row[lang_col]).strip().lower()
                if lang_val not in ("en", "eng", "english"):
                    continue

            if enforce_genius_year and year_col and row.get(year_col) is not None:
                if not is_target_release_year(row[year_col], min_year=min_year, max_year=max_year):
                    continue

            raw_artist = str(row.get(artist_col) or "")
            raw_title = str(row.get(title_col) or "")
            mkey = make_match_key(raw_artist, raw_title)

            if not mkey or mkey not in missing_keys:
                continue

            lyrics_text = str(row.get(lyrics_col) or "").strip()
            if not lyrics_text or len(lyrics_text.split()) < 10:
                continue

            cached_matches[mkey] = {
                "match_key": mkey,
                "genius_artist": raw_artist,
                "genius_title": raw_title,
                "genius_year": row.get(year_col) if year_col else None,
                "lyrics": lyrics_text,
            }
            missing_keys.remove(mkey)
            newly_found += 1

            if not missing_keys:
                print("[Genius Matcher] All requested Spotify tracks matched early!")
                break

        print(f"[Genius Matcher] Found {newly_found} new exact matches in '{hf_dataset_id}'.")

        if cache_path and cached_matches:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(list(cached_matches.values())).to_parquet(cache_path, index=False)
            print(f"[Genius Cache] Updated cache with {len(cached_matches)} total matches -> {cache_path}")

    genius_df = pd.DataFrame(list(cached_matches.values()))
    if genius_df.empty:
        return pd.DataFrame(), spotify_df.copy()

    merged = spotify_df.merge(genius_df, on="match_key", how="left")
    matched_df = merged[merged["lyrics"].notna()].copy()
    unmatched_df = merged[merged["lyrics"].isna()].drop(
        columns=["genius_artist", "genius_title", "genius_year", "lyrics"], errors="ignore"
    ).copy()

    return matched_df, unmatched_df
