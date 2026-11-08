"""Text preprocessing utilities for lyric tokenization and line segmentation."""

from __future__ import annotations

import re
from typing import List

# Matches structural annotations like [Chorus], [Verse 1: Artist], (x2), [Bridge]
_SECTION_HEADER_RE = re.compile(r"^\s*[\[\(][^\]\)]*[\]\)]\s*$")
_INLINE_BRACKET_RE = re.compile(r"\[[^\]]*\]")
# Word tokenizer keeping internal apostrophes (e.g., don't, i'm, walkin')
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def clean_lyric_text(text: str) -> str:
    """Normalize whitespace, apostrophes, and strip bracketed section headers."""
    if not isinstance(text, str) or not text:
        return ""
    # Normalize curly apostrophes and non-breaking spaces
    cleaned = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("`", "'")
        .replace("\u00a0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    cleaned = _INLINE_BRACKET_RE.sub(" ", cleaned)
    return cleaned


def get_lyric_lines(text: str) -> List[str]:
    """Return non-empty lyric lines with section headers removed."""
    cleaned = clean_lyric_text(text)
    if not cleaned:
        return []
    lines: List[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line or _SECTION_HEADER_RE.match(line):
            continue
        # Normalize internal spaces
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines


def get_word_tokens(text: str, lowercase: bool = True) -> List[str]:
    """Extract alphabetic word tokens (preserving contractions like don't, i'm)."""
    cleaned = clean_lyric_text(text)
    if not cleaned:
        return []
    if lowercase:
        cleaned = cleaned.lower()
    return _WORD_TOKEN_RE.findall(cleaned)

