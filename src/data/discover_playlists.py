"""
Discover public Spotify playlists for Viral (TikTok 2020-2025) and Non-Viral
(Billboard / Top Hits / Mainstream 2020-2025) using Spotify Search API and
append them to `config/playlists_viral.txt` and `config/playlists_non_viral.txt`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spotify_scraper import (
    DEFAULT_NON_VIRAL_CONFIG,
    DEFAULT_VIRAL_CONFIG,
    get_spotify_client,
    load_playlist_ids,
)

VIRAL_QUERIES = [
    "tiktok viral 2020",
    "tiktok hits 2020",
    "tiktok songs 2020",
    "tiktok viral 2021",
    "tiktok hits 2021",
    "tiktok songs 2021",
    "tiktok viral 2022",
    "tiktok hits 2022",
    "tiktok songs 2022",
    "tiktok viral 2023",
    "tiktok hits 2023",
    "tiktok songs 2023",
    "tiktok viral 2024",
    "tiktok hits 2024",
    "tiktok songs 2024",
    "tiktok viral 2025",
    "tiktok hits 2025",
    "tiktok songs 2025",
    "tiktok trending songs",
    "viral tiktok songs",
    "tiktokcharts",
    "tiktok pop hits",
    "tiktok viral hip hop",
    "tiktok r&b viral",
    "tiktok indie viral",
    "tiktok dance hits",
    "tiktok throwback 2020 2021",
    "tiktok songs everyone knows",
    "blown up on tiktok",
    "songs from tiktok",
    "popular tiktok songs",
    "top tiktok songs",
    "tiktok top 100",
    "tiktok top 50",
    "tiktok mashup songs",
    "tiktok audio hits",
    "tiktok viral hits",
    "tik tok hits",
    "tik tok viral",
]

NON_VIRAL_QUERIES = [
    "billboard hot 100 2020",
    "billboard hot 100 2021",
    "billboard hot 100 2022",
    "billboard hot 100 2023",
    "billboard hot 100 2024",
    "billboard hot 100 2025",
    "top hits 2020",
    "top hits 2021",
    "top hits 2022",
    "top hits 2023",
    "top hits 2024",
    "top hits 2025",
    "pop hits 2020 2025",
    "mainstream pop hits 2022",
    "uk top 40 2021",
    "uk top 40 2022",
    "uk top 40 2023",
    "uk top 40 2024",
    "radio hits 2020 2024",
    "biggest songs 2020 2024",
    "hot hits 2021",
    "hot hits 2022",
    "hot hits 2023",
    "hot hits 2024",
    "new music friday 2023",
    "new music friday 2024",
]


def discover_and_append(sp, queries, config_path: Path, is_viral: bool, max_per_query: int = 10, pages: int = 2):
    existing = set(load_playlist_ids(config_path))
    added = []

    with config_path.open("a", encoding="utf-8") as f:
        f.write("\n# --- Auto-discovered playlists via Spotify Search ---\n")
        for q in queries:
            for page in range(pages):
                try:
                    res = sp.search(q=q, type="playlist", limit=max_per_query, offset=page * max_per_query)
                    items = res.get("playlists", {}).get("items", []) or []
                except Exception as exc:
                    print(f"[WARN] Search failed for '{q}' (page {page}): {exc}")
                    break

                for item in items:
                    if not item or not item.get("id"):
                        continue
                    pid = item["id"]
                    name = (item.get("name") or "").replace("\n", " ").strip()
                    name_lower = name.lower()

                    # Keep Non-Viral playlists strictly free of "tiktok" in their title
                    if not is_viral and ("tiktok" in name_lower or "tik tok" in name_lower):
                        continue
                    # Ensure Viral playlists actually mention tiktok or viral in their title
                    if is_viral and not any(k in name_lower for k in ("tiktok", "tik tok", "viral")):
                        continue

                    if pid not in existing:
                        existing.add(pid)
                        added.append((pid, name))
                        f.write(f"https://open.spotify.com/playlist/{pid}  # {name}\n")

    print(f"Added {len(added)} new playlists to {config_path.name} (Total: {len(existing)})")


if __name__ == "__main__":
    sp = get_spotify_client()
    discover_and_append(sp, VIRAL_QUERIES, DEFAULT_VIRAL_CONFIG, is_viral=True, max_per_query=10)
    discover_and_append(sp, NON_VIRAL_QUERIES, DEFAULT_NON_VIRAL_CONFIG, is_viral=False, max_per_query=10)

