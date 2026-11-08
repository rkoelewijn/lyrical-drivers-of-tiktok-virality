"""
Verification script for Phase 1 Data Ingestion:
1. Tests Spotify playlist embed extraction.
2. Tests LRCLIB API lyrics lookup.
3. Verifies playlist config files in `config/`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lrclib_fetcher import _query_single_track
from src.data.spotify_scraper import fetch_playlist_tracks_embed, load_playlist_ids

print("=" * 60)
print("PHASE 1 INGESTION PRE-FLIGHT TEST")
print("=" * 60)

# 1. Spotify Playlist Scraper test
print("[1/3] Testing Spotify playlist extraction...")
try:
    pname, tracks = fetch_playlist_tracks_embed("37i9dQZF1DXcBWIGoYBM5M")
    print(f"      SUCCESS: Extracted {len(tracks)} tracks from '{pname}'!")
    if tracks:
        print(f"      Sample track: '{tracks[0]['track_name']}' by {tracks[0]['primary_artist']}")
except Exception as exc:
    print(f"      FAILED: Spotify extraction error: {exc}")

# 2. LRCLIB Lyrics API test
print("\n[2/3] Testing LRCLIB lyrics API...")
try:
    res = _query_single_track("billie eilish::birds of a feather", "Billie Eilish", "BIRDS OF A FEATHER")
    if res["status"] == "matched":
        word_count = len((res["lyrics"] or "").split())
        print(f"      SUCCESS: Connected to LRCLIB API!")
        print(f"      Sample match : '{res['lrclib_title']}' by '{res['lrclib_artist']}' ({word_count} words)")
    else:
        print("      WARN: LRCLIB reachable but sample track not matched.")
except Exception as exc:
    print(f"      FAILED: LRCLIB access error: {exc}")

# 3. Playlist Config files check
print("\n[3/3] Checking curated playlist configs:")
viral_cfg = PROJECT_ROOT / "config" / "playlists_viral.txt"
non_viral_cfg = PROJECT_ROOT / "config" / "playlists_non_viral.txt"

viral_ids = load_playlist_ids(viral_cfg) if viral_cfg.exists() else []
non_viral_ids = load_playlist_ids(non_viral_cfg) if non_viral_cfg.exists() else []

print(f"      - {viral_cfg.name:<25} : {len(viral_ids)} playlist IDs loaded")
print(f"      - {non_viral_cfg.name:<25} : {len(non_viral_ids)} playlist IDs loaded")
print("=" * 60)

