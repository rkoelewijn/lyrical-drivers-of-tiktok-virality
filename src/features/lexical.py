"""Group 1: Aggregated Lexical Diversity & Readability Features."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re
from typing import Dict, List

import pronouncing

from src.features.preprocess import get_lyric_lines, get_word_tokens

_VOWEL_RUN_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

LEXICAL_FEATURES: List[str] = [
    "ttr",
    "mattr_50",
    "hapax_ratio",
    "mean_word_length",
    "mean_syllables_per_word",
    "polysyllable_ratio",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
]


@lru_cache(maxsize=65536)
def count_word_syllables(word: str) -> int:
    """Count syllables in a word using CMU Pronouncing Dictionary with vowel-run fallback."""
    w = word.lower().strip("'")
    if not w:
        return 1
    phones_list = pronouncing.phones_for_word(w)
    if phones_list:
        return max(1, pronouncing.syllable_count(phones_list[0]))
    # Fallback heuristic for OOV / slang / contractions
    clean_w = w.replace("'", "")
    if len(clean_w) <= 3:
        return 1
    # Strip silent trailing 'e' (unless ending in 'le')
    if clean_w.endswith("e") and not clean_w.endswith("le"):
        clean_w = clean_w[:-1]
    runs = _VOWEL_RUN_RE.findall(clean_w)
    return max(1, len(runs))


def compute_mattr(tokens: List[str], window: int = 50) -> float:
    """Compute Moving-Average Type-Token Ratio (MATTR) with O(N) sliding window."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n

    counts: Counter = Counter(tokens[:window])
    unique_counts_sum = len(counts)
    num_windows = n - window + 1

    for i in range(window, n):
        out_tok = tokens[i - window]
        in_tok = tokens[i]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
        counts[in_tok] += 1
        unique_counts_sum += len(counts)

    return unique_counts_sum / (num_windows * window)


def extract_lexical_features(text: str) -> Dict[str, float]:
    """Extract Group 1 (Lexical Diversity & Readability) features for a single lyric text."""
    tokens = get_word_tokens(text, lowercase=True)
    lines = get_lyric_lines(text)
    n_tokens = len(tokens)
    n_lines = max(1, len(lines))

    if n_tokens == 0:
        return {feat: 0.0 for feat in LEXICAL_FEATURES}

    counts = Counter(tokens)
    n_types = len(counts)
    n_hapax = sum(1 for c in counts.values() if c == 1)

    ttr = n_types / n_tokens
    mattr_50 = compute_mattr(tokens, window=50)
    hapax_ratio = n_hapax / n_tokens

    total_chars = sum(len(t.replace("'", "")) for t in tokens)
    mean_word_length = total_chars / n_tokens

    syllables = [count_word_syllables(t) for t in tokens]
    total_syllables = sum(syllables)
    mean_syllables = total_syllables / n_tokens
    polysyllable_ratio = sum(1 for s in syllables if s >= 3) / n_tokens

    words_per_line = n_tokens / n_lines
    flesch_reading_ease = 206.835 - 1.015 * words_per_line - 84.6 * mean_syllables
    flesch_kincaid_grade = 0.39 * words_per_line + 11.8 * mean_syllables - 15.59

    return {
        "ttr": float(ttr),
        "mattr_50": float(mattr_50),
        "hapax_ratio": float(hapax_ratio),
        "mean_word_length": float(mean_word_length),
        "mean_syllables_per_word": float(mean_syllables),
        "polysyllable_ratio": float(polysyllable_ratio),
        "flesch_reading_ease": float(flesch_reading_ease),
        "flesch_kincaid_grade": float(flesch_kincaid_grade),
    }

