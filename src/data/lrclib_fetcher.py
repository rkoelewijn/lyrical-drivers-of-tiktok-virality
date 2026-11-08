"""
Fast, concurrent lyrics fetcher using the open LRCLIB API (https://lrclib.net)
with strict exact matching on `match_key = norm_artist::norm_title`,
English-only language verification, and persistent local caching.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

# High-frequency English function words used for robust, fast English lyrics validation
ENGLISH_STOPWORDS: Set[str] = {
    "the", "you", "i", "me", "my", "and", "to", "a", "in", "it", "is", "that",
    "of", "for", "on", "with", "your", "im", "dont", "all", "we", "be", "so",
    "just", "like", "but", "what", "yeah", "no", "up", "out", "this", "know",
    "can", "cause", "get", "got", "was", "when", "if", "at", "go", "see", "now",
    "they", "she", "he", "have", "are", "do", "will", "from", "one", "tell",
}


def is_english_lyrics(text: str, min_words: int = 30, min_stopword_matches: int = 5) -> bool:
    """
    Verify that lyric text is predominantly English by checking for section structure,
    sufficient word volume, and high-frequency English function words.
    Filters out non-English tracks (Spanish, French, Portuguese phonk, K-pop) and instrumentals.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    # Strip bracketed section indicators like [Chorus], [Verse 1]
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", cleaned.lower())
    if len(tokens) < min_words:
        return False
    unique_tokens = set(tokens)
    overlap = len(unique_tokens & ENGLISH_STOPWORDS)
    return overlap >= min_stopword_matches


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
    english_only: bool = True,
) -> Dict:
    """
    Query LRCLIB for a single song using a thread-local session, verify
    strict normalized key match, and enforce English language filtering.
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
            lyrics = data.get("plainLyrics") or data.get("syncedLyrics")
            if lyrics and not data.get("instrumental"):
                resp_key = make_match_key(data.get("artistName"), data.get("trackName"))
                if resp_key == match_key or (
                    normalize_artist(data.get("artistName")) == normalize_artist(primary_artist_display)
                    and normalize_title(data.get("trackName")) == clean_title
                ):
                    if not english_only or is_english_lyrics(lyrics):
                        return {
                            "match_key": match_key,
                            "lrclib_id": data.get("id"),
                            "lrclib_artist": data.get("artistName"),
                            "lrclib_title": data.get("trackName"),
                            "lrclib_album": data.get("albumName") or "",
                            "lyrics": lyrics,
                            "is_english": True,
                            "status": "matched",
                        }
    except Exception:
        pass

    # 2. Search fallback: query LRCLIB search endpoint and check top results
    try:
        resp = session.get(
            f"{LRCLIB_BASE_URL}/search",
            params={"track_name": clean_title, "artist_name": primary_artist_display},
            timeout=10,
        )
        if resp.status_code == 200:
            hits = resp.json()
            for cand in hits:
                lyrics = cand.get("plainLyrics") or cand.get("syncedLyrics")
                if not lyrics or cand.get("instrumental"):
                    continue
                cand_key = make_match_key(cand.get("artistName"), cand.get("trackName"))
                if cand_key == match_key:
                    if not english_only or is_english_lyrics(lyrics):
                        return {
                            "match_key": match_key,
                            "lrclib_id": cand.get("id"),
                            "lrclib_artist": cand.get("artistName"),
                            "lrclib_title": cand.get("trackName"),
                            "lrclib_album": cand.get("albumName") or "",
                            "lyrics": lyrics,
                            "is_english": True,
                            "status": "matched",
                        }
    except Exception:
        pass

    return {"match_key": match_key, "lyrics": None, "status": "not_found", "is_english": False}


def fetch_lyrics_lrclib(
    spotify_df: pd.DataFrame,
    target_per_class: int = 3000,
    max_workers: int = 8,
    cache_path: Optional[Path] = DEFAULT_RAW_DIR / "lrclib_lyrics_cache.parquet",
    english_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch lyrics for Spotify tracks using LRCLIB, caching to parquet.
    Filters exclusively for English-language lyrics and prioritizes songs
    by playlist_frequency descending.
    """
    if spotify_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cached_records: Dict[str, Dict] = {}
    if cache_path and cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        for row in cached_df.to_dict(orient="records"):
            if row.get("status") != "matched" or not row.get("lyrics"):
                continue
            # Annotate English status if missing
            if "is_english" not in row:
                row["is_english"] = is_english_lyrics(row["lyrics"])
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

        def _is_valid_cached(mkey: str) -> bool:
            if mkey not in cached_records:
                return False
            rec = cached_records[mkey]
            if not rec.get("lyrics"):
                return False
            return bool(rec.get("is_english", True)) if english_only else True

        already_matched_count = sum(1 for mkey in unique_tracks["match_key"] if _is_valid_cached(mkey))
        needed_for_class = max(0, target_per_class - already_matched_count)

        if needed_for_class == 0:
            print(
                f"[LRCLIB Fetcher] Class ({class_name}) already has {already_matched_count} >= {target_per_class} valid English matched lyrics in cache!"
            )
            continue

        to_fetch = [
            row
            for row in unique_tracks.to_dict(orient="records")
            if not _is_valid_cached(row["match_key"])
        ]

        if not to_fetch:
            continue

        print(
            f"[LRCLIB Fetcher] Class ({class_name}): have {already_matched_count}/{target_per_class} English matched; "
            f"scanning up to {len(to_fetch)} candidate tracks (workers={max_workers})..."
        )
        newly_matched = 0
        batch_size = 200

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            with tqdm(total=needed_for_class, desc=f"Matching English LRCLIB ({class_name})", unit="matched") as pbar:
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
                            english_only,
                        ): item["match_key"]
                        for item in batch
                    }
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res.get("status") == "matched" and res.get("lyrics"):
                            if not english_only or res.get("is_english"):
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

    if english_only and "is_english" in cache_df.columns:
        cache_df = cache_df[cache_df["is_english"] == True].copy()

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
