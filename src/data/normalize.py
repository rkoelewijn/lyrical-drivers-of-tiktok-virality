"""
Text normalization utilities for strict exact entity resolution between
Spotify track metadata and the Hugging Face Genius lyrics corpus.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Parenthetical/bracketed suffixes that indicate remixes, features, or audio edits
_BRACKET_SUFFIX_RE = re.compile(
    r"[\(\[\{]"
    r"[^)\]\}]*?"
    r"\b(?:feat\.?|ft\.?|featuring|with|prod\.?|produced by|remix|mix|edit|"
    r"version|ver\.?|sped up|slowed|reverb|remaster(?:ed)?|live|acoustic|"
    r"instrumental|bonus|deluxe| soundtrack|from\s+[\"']?)"
    r"[^)\]\}]*?"
    r"[\)\]\}]",
    flags=re.IGNORECASE,
)

# Trailing dash suffixes: e.g. "Song Name - Sped Up", "Song Name - 2021 Remaster", "Song Name - Twin Ver."
_DASH_SUFFIX_RE = re.compile(
    r"\s+[-–—]\s+.*"
    r"\b(?:feat\.?|ft\.?|featuring|with|remix|mix|edit|ver(?:sion)?|sped up|slowed|"
    r"reverb|remaster(?:ed)?|live|acoustic|instrumental|mono|stereo|radio)\b.*$",
    flags=re.IGNORECASE,
)

# Secondary artist splitters in artist strings (e.g., "Artist A feat. Artist B", "Artist A & Artist B")
_ARTIST_SPLIT_RE = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring|with|vs\.?|x)\s+|\s*[,;&/]\s*",
    flags=re.IGNORECASE,
)

# Non-alphanumeric characters (keep spaces and basic ASCII alphanumeric)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _ascii_fold(text: str) -> str:
    """Fold Unicode accents/diacritics and normalize smart quotes to ASCII."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_artist(artist: Optional[str], primary_only: bool = True) -> str:
    """
    Normalize an artist name for strict exact matching.

    Examples:
        "Doja Cat (feat. SZA)" -> "doja cat"
        "ROSALÍA" -> "rosalia"
        "The Weeknd & Ariana Grande" -> "the weeknd" (if primary_only=True)
    """
    if not artist or not isinstance(artist, str):
        return ""

    text = _ascii_fold(artist.strip().lower())
    # Remove any parenthetical feature tags in the artist field
    text = _BRACKET_SUFFIX_RE.sub(" ", text)

    if primary_only:
        parts = _ARTIST_SPLIT_RE.split(text)
        if parts and parts[0].strip():
            text = parts[0]

    # Replace "&" or "+" with "and" if not split
    text = text.replace("&", " and ")
    text = _NON_ALNUM_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_title(title: Optional[str]) -> str:
    """
    Normalize a track title for strict exact matching while collapsing
    remixes, sped-up edits, and "(feat. ...)" suffixes to the canonical song title.

    Examples:
        "Say So (feat. Nicki Minaj)" -> "say so"
        "Cupid - Twin Ver. (Sped Up)" -> "cupid - twin ver" -> "cupid"
        "Industry Baby (with Jack Harlow)" -> "industry baby"
    """
    if not title or not isinstance(title, str):
        return ""

    text = _ascii_fold(title.strip().lower())

    # Strip bracketed/parenthetical feature, remix, sped-up, or version tags
    text = _BRACKET_SUFFIX_RE.sub(" ", text)
    # Strip trailing dash suffixes (e.g. "- Sped Up", "- Remix", "- feat. X")
    text = _DASH_SUFFIX_RE.sub(" ", text)

    # Also strip trailing "feat. ..." without parentheses
    text = re.sub(r"\s+\b(?:feat\.?|ft\.?|featuring)\b.*$", "", text, flags=re.IGNORECASE)

    text = text.replace("&", " and ")
    text = _NON_ALNUM_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def make_match_key(artist: Optional[str], title: Optional[str]) -> str:
    """
    Create a deterministic composite key `norm_artist::norm_title` for strict exact matching.
    Returns an empty string if either normalized component is empty.
    """
    norm_a = normalize_artist(artist, primary_only=True)
    norm_t = normalize_title(title)
    if not norm_a or not norm_t:
        return ""
    return f"{norm_a}::{norm_t}"


def parse_release_year(release_date: Optional[str | int | float]) -> Optional[int]:
    """
    Extract a 4-digit year from a Spotify release_date ('YYYY-MM-DD', 'YYYY') or numeric year.
    """
    if release_date is None:
        return None
    if isinstance(release_date, (int, float)):
        try:
            year = int(release_date)
            return year if 1900 <= year <= 2100 else None
        except (ValueError, OverflowError):
            return None
    if isinstance(release_date, str):
        match = re.search(r"\b(19\d{2}|20\d{2})\b", release_date.strip())
        if match:
            return int(match.group(1))
    return None


def is_target_release_year(
    release_date: Optional[str | int | float],
    min_year: int = 2020,
    max_year: int = 2025,
) -> bool:
    """Return True if the parsed release year falls strictly within [min_year, max_year]."""
    year = parse_release_year(release_date)
    return year is not None and (min_year <= year <= max_year)
