"""
Fast, concurrent lyrics fetcher using the open LRCLIB API (https://lrclib.net)
with strict exact matching on `match_key = norm_artist::norm_title` and
persistent local caching.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

from src.data.normalize import make_match_key, normalize_artist, normalize_title

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
LRCLIB_BASE_URL = "https://lrclib.net/api"
USER_AGENT = "TxMM-Lyrics-Research/1.0 (Radboud University Data Science Project)"

_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    """Return a thread-local requests.Session with automatic retry."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session


def _clean_whitespace(text: str) -> str:
    """Replace non-breaking spaces and zero-width characters with standard spaces."""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text or "")
    text = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
    return text


def _query_single_track(
    match_key: str,
    raw_artist: str,
    raw_title: str,
) -> Dict:
    """
    Query LRCLIB for a single song using a thread-local session and verify
    strict normalized key match.
    """
    session = _get_thread_session()
    artist_std = _clean_whitespace(raw_artist)
    title_std = _clean_whitespace(raw_title)

    primary_artist_display = re.split(r"[,;&/]", artist_std)[0].strip() or artist_std
    clean_title = normalize_title(title_std)

    # 1. Fast path: single /api/get call with primary artist and track title
    try:
        resp = session.get(
            f"{LRCLIB_BASE_URL}/get",
            params={"artist_name": primary_artist_display, "track_name": title_std},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            plain = (data.get("plainLyrics") or "").strip()
            if not data.get("instrumental") and len(plain.split()) >= 10:
                return {
                    "match_key": match_key,
                    "lrclib_id": data.get("id"),
                    "lrclib_artist": data.get("artistName"),
                    "lrclib_title": data.get("trackName"),
                    "lrclib_album": data.get("albumName"),
                    "lyrics": plain,
                    "status": "matched",
                }
    except Exception:
        pass

    # 2. Fallback: single /api/search call with strict exact match on normalized (artist, title)
    try:
        resp = session.get(
            f"{LRCLIB_BASE_URL}/search",
            params={"q": f"{primary_artist_display} {clean_title}"},
            timeout=10,
        )
        if resp.status_code == 200:
            candidates = resp.json() or []
            for cand in candidates:
                if cand.get("instrumental"):
                    continue
                cand_key = make_match_key(cand.get("artistName"), cand.get("trackName"))
                plain = (cand.get("plainLyrics") or "").strip()
                if cand_key == match_key and len(plain.split()) >= 10:
                    return {
                        "match_key": match_key,
                        "lrclib_id": cand.get("id"),
                        "lrclib_artist": cand.get("artistName"),
                        "lrclib_title": cand.get("trackName"),
                        "lrclib_album": cand.get("albumName"),
                        "lyrics": plain,
                        "status": "matched",
                    }
    except Exception:
        pass

    return {
        "match_key": match_key,
        "lrclib_id": None,
        "lrclib_artist": None,
        "lrclib_title": None,
        "lrclib_album": None,
        "lyrics": None,
        "status": "not_found",
    }


def fetch_lyrics_lrclib(
    spotify_df: pd.DataFrame,
    cache_path: Optional[Path] = DEFAULT_RAW_DIR / "lrclib_lyrics_cache.parquet",
    max_workers: int = 8,
    retry_not_found: bool = False,
    target_per_class: int = 3000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch plain English lyrics from LRCLIB for `spotify_df`, prioritizing songs
    with the highest `playlist_frequency` and stopping per class once `target_per_class`
    matched songs are collected.

    Maintains a persistent local cache at `cache_path` (saving incrementally) so
    already-fetched songs are returned immediately on subsequent runs.

    Returns:
        (matched_df, unmatched_df)
    """
    if spotify_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cached_records: Dict[str, Dict] = {}
    if cache_path and cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        for row in cached_df.to_dict(orient="records"):
            if row.get("status") != "matched" or not row.get("lyrics"):
                continue
            cached_records[row["match_key"]] = row
        print(
            f"[LRCLIB Cache] Loaded {len(cached_records)} previously matched songs from {cache_path.name}"
        )

    sort_cols = ["playlist_frequency"] if "playlist_frequency" in spotify_df.columns else []
    ordered_df = (
        spotify_df.sort_values(sort_cols, ascending=False, kind="mergesort")
        if sort_cols
        else spotify_df
    )

    labels = ordered_df["label"].dropna().unique() if "label" in ordered_df.columns else [None]

    for lbl in labels:
        class_df = ordered_df[ordered_df["label"] == lbl] if lbl is not None else ordered_df
        class_name = f"label={lbl}" if lbl is not None else "all"

        unique_tracks = (
            class_df[["match_key", "primary_artist", "track_name"]]
            .dropna(subset=["match_key"])
            .drop_duplicates(subset=["match_key"])
        )

        already_matched_count = sum(
            1 for mkey in unique_tracks["match_key"] if mkey in cached_records
        )
        needed_for_class = max(0, target_per_class - already_matched_count)

        if needed_for_class == 0:
            print(
                f"[LRCLIB Fetcher] Class ({class_name}) already has {already_matched_count} >= {target_per_class} matched lyrics in cache!"
            )
            continue

        to_fetch = [
            row
            for row in unique_tracks.to_dict(orient="records")
            if row["match_key"] not in cached_records
        ]

        if not to_fetch:
            continue

        print(
            f"[LRCLIB Fetcher] Class ({class_name}): have {already_matched_count}/{target_per_class} matched; "
            f"scanning up to {len(to_fetch)} candidate tracks (workers={max_workers})..."
        )
        newly_matched = 0
        batch_size = 200

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            with tqdm(total=needed_for_class, desc=f"Matching LRCLIB ({class_name})", unit="matched") as pbar:
                for start_idx in range(0, len(to_fetch), batch_size):
                    if newly_matched >= needed_for_class:
                        break
                    batch = to_fetch[start_idx : start_idx + batch_size]
                    futures = {
                        pool.submit(
                            _query_single_track,
                            item["match_key"],
                            item["primary_artist"],
                            item["track_name"],
                        ): item["match_key"]
                        for item in batch
                    }
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res.get("status") == "matched" and res.get("lyrics"):
                            cached_records[res["match_key"]] = res
                            newly_matched += 1
                            if pbar.n < needed_for_class:
                                pbar.update(1)

                    if cache_path and cached_records:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(list(cached_records.values())).to_parquet(cache_path, index=False)

        print(
            f"[LRCLIB Fetcher] Class ({class_name}): found {newly_matched} new lyrics "
            f"(Total matched for class: {already_matched_count + newly_matched}/{target_per_class})."
        )

    cache_df = pd.DataFrame(list(cached_records.values()))
    if cache_df.empty:
        return pd.DataFrame(), spotify_df.copy()

    merged = spotify_df.merge(
        cache_df[["match_key", "lrclib_id", "lrclib_artist", "lrclib_title", "lrclib_album", "lyrics"]],
        on="match_key",
        how="left",
    )
    matched_df = merged[merged["lyrics"].notna()].copy()
    unmatched_df = merged[merged["lyrics"].isna()].drop(
        columns=["lrclib_id", "lrclib_artist", "lrclib_title", "lrclib_album", "lyrics"],
        errors="ignore",
    ).copy()

    return matched_df, unmatched_df
