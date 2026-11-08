"""Group 2: Repetition & Information Compressibility Features."""

from __future__ import annotations

from collections import Counter
import re
from typing import Dict, List
import zlib

from src.features.preprocess import clean_lyric_text, get_lyric_lines, get_word_tokens

REPETITION_FEATURES: List[str] = [
    "compression_ratio",
    "duplicate_line_ratio",
    "adjacent_line_repeat_ratio",
    "max_line_repetition_share",
    "repeated_bigram_ratio",
    "repeated_trigram_ratio",
]

_LINE_NORM_RE = re.compile(r"[^a-z0-9\s']+ ")


def _normalize_line_for_comparison(line: str) -> str:
    """Lowercase and strip punctuation from a line for structural repetition checks."""
    norm = re.sub(r"[^a-z0-9\s']", "", line.lower())
    return re.sub(r"\s+", " ", norm).strip()


def extract_repetition_features(text: str) -> Dict[str, float]:
    """Extract Group 2 (Repetition & Compressibility) features for a single lyric text."""
    cleaned = clean_lyric_text(text).strip()
    if not cleaned:
        return {feat: 0.0 for feat in REPETITION_FEATURES}

    # 1. Zlib compressibility: 1 - (compressed_bytes / raw_bytes)
    # Higher values mean greater redundancy/repetitiveness (Parada-Cabaleiro et al., 2024)
    raw_bytes = cleaned.lower().encode("utf-8")
    if len(raw_bytes) > 0:
        compressed_bytes = zlib.compress(raw_bytes, level=9)
        compression_ratio = max(0.0, 1.0 - (len(compressed_bytes) / len(raw_bytes)))
    else:
        compression_ratio = 0.0

    # 2. Line-level repetition (chorus/hook repetition)
    lines = get_lyric_lines(text)
    norm_lines = [_normalize_line_for_comparison(ln) for ln in lines]
    norm_lines = [ln for ln in norm_lines if ln]
    n_lines = len(norm_lines)

    if n_lines > 1:
        line_counts = Counter(norm_lines)
        duplicate_line_ratio = 1.0 - (len(line_counts) / n_lines)
        max_line_repetition_share = max(line_counts.values()) / n_lines
        adjacent_repeats = sum(
            1 for i in range(n_lines - 1) if norm_lines[i] == norm_lines[i + 1]
        )
        adjacent_line_repeat_ratio = adjacent_repeats / (n_lines - 1)
    else:
        duplicate_line_ratio = 0.0
        max_line_repetition_share = 0.0
        adjacent_line_repeat_ratio = 0.0

    # 3. Token n-gram repetition
    tokens = get_word_tokens(text, lowercase=True)
    n_tokens = len(tokens)

    if n_tokens >= 2:
        bigrams = [ (tokens[i], tokens[i + 1]) for i in range(n_tokens - 1) ]
        repeated_bigram_ratio = 1.0 - (len(set(bigrams)) / len(bigrams))
    else:
        repeated_bigram_ratio = 0.0

    if n_tokens >= 3:
        trigrams = [
            (tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(n_tokens - 2)
        ]
        repeated_trigram_ratio = 1.0 - (len(set(trigrams)) / len(trigrams))
    else:
        repeated_trigram_ratio = 0.0

    return {
        "compression_ratio": float(compression_ratio),
        "duplicate_line_ratio": float(duplicate_line_ratio),
        "adjacent_line_repeat_ratio": float(adjacent_line_repeat_ratio),
        "max_line_repetition_share": float(max_line_repetition_share),
        "repeated_bigram_ratio": float(repeated_bigram_ratio),
        "repeated_trigram_ratio": float(repeated_trigram_ratio),
    }

