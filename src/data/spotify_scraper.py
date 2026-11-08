"""
Spotify playlist scraper for curating Viral (label=1) and Non-Viral (label=0)
track lists (2020-2025).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import requests
import pandas as pd

from src.data.normalize import (
    is_target_release_year,
    make_match_key,
    normalize_artist,
    normalize_title,
    parse_release_year,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIRAL_CONFIG = PROJECT_ROOT / "config" / "playlists_viral.txt"
DEFAULT_NON_VIRAL_CONFIG = PROJECT_ROOT / "config" / "playlists_non_viral.txt"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

_PLAYLIST_URL_RE = re.compile(r"playlist[/:]([a-zA-Z0-9]{22})")
_BARE_ID_RE = re.compile(r"^([a-zA-Z0-9]{22})$")


def fetch_playlist_tracks_embed(playlist_id: str) -> Tuple[str, List[Dict]]:
    """
    Fetch track items directly from Spotify's public embed interface.
    Works for both user-created and official Spotify editorial playlists without
    triggering Spotify's 403/404 Web API restrictions.
    """
    url = f"https://open.spotify.com/embed/playlist/{playlist_id}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                break
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", 3 * (attempt + 1)))
                time.sleep(retry_after)
                continue
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if r is None or r.status_code != 200:
        return "", []

    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', r.text)
    if not match:
        return "", []

    try:
        data = json.loads(match.group(1))
        entity = data.get("props", {}).get("pageProps", {}).get("state", {}).get("data", {}).get("entity", {})
        playlist_name = entity.get("title") or playlist_id
        tracklist = entity.get("trackList", [])
    except Exception:
        return "", []

    tracks: List[Dict] = []
    for item in tracklist:
        uri = item.get("uri", "")
        tid = uri.split(":")[-1] if uri else ""
        track_name = item.get("title") or ""
        primary_artist = item.get("subtitle") or ""

        norm_artist = normalize_artist(primary_artist, primary_only=True)
        norm_title = normalize_title(track_name)
        match_key = make_match_key(primary_artist, track_name)

        if not match_key:
            continue

        tracks.append(
            {
                "spotify_track_id": tid,
                "track_name": track_name,
                "primary_artist": primary_artist,
                "all_artists": primary_artist,
                "album_name": "",
                "release_date": None,
                "release_year": None,
                "spotify_popularity": 0,
                "norm_artist": norm_artist,
                "norm_title": norm_title,
                "match_key": match_key,
                "playlist_id": playlist_id,
                "playlist_name": playlist_name,
                "playlist_followers": 0,
            }
        )

    return playlist_name, tracks


def _load_env_file(env_path: Path) -> None:
    """Load key=value pairs from `.env` into os.environ without requiring python-dotenv."""
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ImportError:
        pass

    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def parse_playlist_id(raw_entry: str) -> Optional[str]:
    """
    Extract a 22-character Spotify playlist ID from a URL, URI, or raw ID string.
    """
    cleaned = raw_entry.split("#", 1)[0].strip()
    if not cleaned:
        return None
    url_match = _PLAYLIST_URL_RE.search(cleaned)
    if url_match:
        return url_match.group(1)
    bare_match = _BARE_ID_RE.match(cleaned)
    if bare_match:
        return bare_match.group(1)
    return None


def load_playlist_ids(config_path: Path) -> List[str]:
    """
    Read a playlist configuration file and return a deduplicated list of playlist IDs.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Playlist config file not found: {config_path}")

    playlist_ids: List[str] = []
    seen: Set[str] = set()
    for line_num, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pid = parse_playlist_id(stripped)
        if pid is None:
            print(f"[WARN] Could not parse Spotify playlist ID on line {line_num} of {config_path.name}: {line}")
            continue
        if pid not in seen:
            seen.add(pid)
            playlist_ids.append(pid)
    return playlist_ids


def get_spotify_client():
    """
    Initialize and return an authenticated Spotipy client.
    
    Uses `SpotifyOAuth` with redirect_uri='http://127.0.0.1:8888/callback' so that
    playlist track items can be fetched (as Spotify's 2024 Web API update requires
    user authorization for playlist items). Falls back to `SpotifyClientCredentials`
    if user authorization is not available.
    """
    _load_env_file(PROJECT_ROOT / ".env")
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    if not client_id or not client_secret or client_id == "your_spotify_client_id_here":
        raise RuntimeError(
            "Missing Spotify API credentials. Copy `.env.example` to `.env` and fill in "
            "SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET from https://developer.spotify.com/dashboard."
        )

    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

    try:
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="playlist-read-private playlist-read-collaborative",
            cache_path=str(PROJECT_ROOT / ".spotify_cache"),
            open_browser=True,
        )
        return spotipy.Spotify(auth_manager=auth_manager, requests_timeout=20, retries=3)
    except Exception:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        return spotipy.Spotify(auth_manager=auth_manager, requests_timeout=20, retries=3)


def fetch_playlist_tracks(sp, playlist_id: str) -> Tuple[str, List[Dict]]:
    """
    Fetch metadata and all track items for a given Spotify playlist ID, handling pagination.
    """
    playlist_meta = sp.playlist(playlist_id, fields="name,followers.total,owner.display_name")
    playlist_name = playlist_meta.get("name", playlist_id)
    followers = playlist_meta.get("followers", {}).get("total", 0)

    tracks: List[Dict] = []
    results = sp.playlist_items(
        playlist_id,
        fields="items(track(id,name,popularity,is_local,artists(name),album(name,release_date))),next",
        additional_types=("track",),
        limit=100,
    )

    while results:
        for item in results.get("items", []):
            track = item.get("track")
            if not track or track.get("is_local") or not track.get("id"):
                continue

            artists = [a.get("name", "") for a in (track.get("artists") or []) if a.get("name")]
            primary_artist = artists[0] if artists else ""
            track_name = track.get("name") or ""
            album = track.get("album") or {}
            release_date = album.get("release_date") or ""
            release_year = parse_release_year(release_date)

            norm_artist = normalize_artist(primary_artist, primary_only=True)
            norm_title = normalize_title(track_name)
            match_key = make_match_key(primary_artist, track_name)

            if not match_key:
                continue

            tracks.append(
                {
                    "spotify_track_id": track["id"],
                    "track_name": track_name,
                    "primary_artist": primary_artist,
                    "all_artists": ", ".join(artists),
                    "album_name": album.get("name", ""),
                    "release_date": release_date,
                    "release_year": release_year,
                    "spotify_popularity": track.get("popularity", 0),
                    "norm_artist": norm_artist,
                    "norm_title": norm_title,
                    "match_key": match_key,
                    "playlist_id": playlist_id,
                    "playlist_name": playlist_name,
                    "playlist_followers": followers,
                }
            )

        if results.get("next"):
            results = sp.next(results)
        else:
            break

    return playlist_name, tracks


def scrape_playlist_group(
    sp,
    playlist_ids: List[str],
    label: int,
    label_name: str,
    min_year: int = 2020,
    max_year: int = 2025,
    target_count: int = 3000,
    exclude_match_keys: Optional[Set[str]] = None,
    exclude_track_ids: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    """
    Scrape all playlists for a class (Viral or Non-Viral), filter to [min_year, max_year],
    deduplicate on `match_key`, and log progress toward `target_count`.

    Returns:
        (filtered_df, all_seen_match_keys, all_seen_track_ids)
    """
    exclude_keys = exclude_match_keys or set()
    exclude_ids = exclude_track_ids or set()

    all_seen_keys: Set[str] = set()
    all_seen_ids: Set[str] = set()
    kept_records: Dict[str, Dict] = {}

    print(f"\n{'=' * 78}")
    print(f"Scraping {label_name} Playlists (label={label}, years={min_year}-{max_year}, target={target_count})")
    print(f"{'=' * 78}")

    if not playlist_ids:
        print(f"[WARN] No playlist IDs found for {label_name}. Add URLs/IDs to the config file.")
        return pd.DataFrame(), all_seen_keys, all_seen_ids

    # Stop scraping once we have ~1.65x target_count unique songs (to account for ~65% lyrics match rate)
    raw_unique_goal = int(target_count * 1.65)

    for idx, pid in enumerate(playlist_ids, start=1):
        # 1. Try public embed first (bypasses 403/404 Web API restrictions)
        if len(kept_records) >= raw_unique_goal:
            print(
                f"  -> Reached {len(kept_records)} unique Spotify tracks (>= {raw_unique_goal} buffer for {target_count} target). Stopping early!"
            )
            break
        pname, raw_tracks = fetch_playlist_tracks_embed(pid)

        # 2. Fall back to Spotipy Web API if embed returned nothing
        if not raw_tracks and sp is not None:
            try:
                pname, raw_tracks = fetch_playlist_tracks(sp, pid)
            except Exception as exc:
                print(f"  [{idx}/{len(playlist_ids)}] ERROR fetching playlist {pid}: {exc}")
                continue

        time.sleep(0.15)
        if not raw_tracks:
            print(f"  [{idx}/{len(playlist_ids)}] Could not retrieve tracks for playlist {pid}")
            print(f"  [{idx:03d}/{len(playlist_ids):03d}] Could not retrieve tracks for playlist {pid}")
            continue

        in_year_count = 0
        added_count = 0
        excluded_overlap = 0

        for rec in raw_tracks:
            mkey = rec["match_key"]
            tid = rec["spotify_track_id"]
            all_seen_keys.add(mkey)
            all_seen_ids.add(tid)

            # Exclude any track that ever appeared in the Viral playlists when building Non-Viral
            if mkey in exclude_keys or tid in exclude_ids:
                excluded_overlap += 1
                continue

            # If release year is present on Spotify, check boundary; otherwise Genius matcher validates year
            if rec["release_year"] is not None:
                if not is_target_release_year(rec["release_year"], min_year=min_year, max_year=max_year):
                    continue

            in_year_count += 1

            if mkey not in kept_records:
                rec_copy = dict(rec)
                rec_copy["label"] = label
                rec_copy["label_name"] = label_name
                rec_copy["playlist_frequency"] = 1
                kept_records[mkey] = rec_copy
                added_count += 1
            else:
                kept_records[mkey]["playlist_frequency"] += 1

        total_so_far = len(kept_records)
        overlap_msg = f", {excluded_overlap} viral overlaps skipped" if exclude_keys else ""
        year_info = f"{in_year_count} in {min_year}-{max_year}" if any(r["release_year"] for r in raw_tracks) else f"{len(raw_tracks)} tracks"
        print(
            f"  [{idx:02d}/{len(playlist_ids):02d}] '{pname[:36]}' ({pid}) -> "
            f"{len(raw_tracks)} raw | {year_info} | "
            f"+{added_count} new unique{overlap_msg} | Total: {total_so_far}/{target_count}"
        )

    df = pd.DataFrame(list(kept_records.values()))
    print(
        f"-> Finished {label_name}: {len(df)} unique {min_year}-{max_year} Spotify songs collected "
        f"({max(0, target_count - len(df))} more needed before lyrics matching)."
    )
    return df, all_seen_keys, all_seen_ids


def run_spotify_scraper(
    viral_config: Path = DEFAULT_VIRAL_CONFIG,
    non_viral_config: Path = DEFAULT_NON_VIRAL_CONFIG,
    output_dir: Path = DEFAULT_RAW_DIR,
    min_year: int = 2020,
    max_year: int = 2025,
    target_per_class: int = 3000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrape both Viral and Non-Viral playlist configs, save raw deduplicated tables
    to `data/raw/`, and return `(viral_df, non_viral_df)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viral_ids = load_playlist_ids(viral_config)
    non_viral_ids = load_playlist_ids(non_viral_config)

    try:
        sp = get_spotify_client()
    except Exception:
        sp = None

    viral_df, all_viral_keys, all_viral_track_ids = scrape_playlist_group(
        sp=sp,
        playlist_ids=viral_ids,
        label=1,
        label_name="Viral",
        min_year=min_year,
        max_year=max_year,
        target_count=target_per_class,
    )
    raw_goal = int(target_per_class * 1.65)
    viral_out = output_dir / "spotify_viral_2020_2025.parquet"
    non_viral_out = output_dir / "spotify_non_viral_2020_2025.parquet"

    non_viral_df, _, _ = scrape_playlist_group(
        sp=sp,
        playlist_ids=non_viral_ids,
        label=0,
        label_name="Non-Viral",
        min_year=min_year,
        max_year=max_year,
        target_count=target_per_class,
        exclude_match_keys=all_viral_keys,
        exclude_track_ids=all_viral_track_ids,
    )
    if viral_out.exists():
        existing_v = pd.read_parquet(viral_out)
        if len(existing_v) >= raw_goal:
            print(f"[Spotify Cache] Reusing {len(existing_v)} already-scraped Viral tracks from {viral_out.name}")
            viral_df = existing_v
            all_viral_keys = set(viral_df["match_key"].dropna())
            all_viral_track_ids = set(viral_df["spotify_track_id"].dropna())
        else:
            viral_df, all_viral_keys, all_viral_track_ids = scrape_playlist_group(
                sp=sp,
                playlist_ids=viral_ids,
                label=1,
                label_name="Viral",
                min_year=min_year,
                max_year=max_year,
                target_count=target_per_class,
            )
    else:
        viral_df, all_viral_keys, all_viral_track_ids = scrape_playlist_group(
            sp=sp,
            playlist_ids=viral_ids,
            label=1,
            label_name="Viral",
            min_year=min_year,
            max_year=max_year,
            target_count=target_per_class,
        )

    if non_viral_out.exists():
        existing_nv = pd.read_parquet(non_viral_out)
        if len(existing_nv) >= raw_goal:
            print(f"[Spotify Cache] Reusing {len(existing_nv)} already-scraped Non-Viral tracks from {non_viral_out.name}")
            non_viral_df = existing_nv
        else:
            non_viral_df, _, _ = scrape_playlist_group(
                sp=sp,
                playlist_ids=non_viral_ids,
                label=0,
                label_name="Non-Viral",
                min_year=min_year,
                max_year=max_year,
                target_count=target_per_class,
                exclude_match_keys=all_viral_keys,
                exclude_track_ids=all_viral_track_ids,
            )
    else:
        non_viral_df, _, _ = scrape_playlist_group(
            sp=sp,
            playlist_ids=non_viral_ids,
            label=0,
            label_name="Non-Viral",
            min_year=min_year,
            max_year=max_year,
            target_count=target_per_class,
            exclude_match_keys=all_viral_keys,
            exclude_track_ids=all_viral_track_ids,
        )

    if not viral_df.empty:
        viral_out = output_dir / "spotify_viral_2020_2025.parquet"
        viral_df.to_parquet(viral_out, index=False)
        print(f"Saved Viral Spotify tracks to {viral_out}")

    if not non_viral_df.empty:
        non_viral_out = output_dir / "spotify_non_viral_2020_2025.parquet"
        non_viral_df.to_parquet(non_viral_out, index=False)
        print(f"Saved Non-Viral Spotify tracks to {non_viral_out}")

    return viral_df, non_viral_df


if __name__ == "__main__":
    run_spotify_scraper()
