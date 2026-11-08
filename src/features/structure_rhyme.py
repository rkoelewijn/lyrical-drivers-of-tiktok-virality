"""Group 5: Lyric Structure & Phonetic Rhyme Features."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Dict, List, Set

import pronouncing

from src.features.preprocess import get_lyric_lines, get_word_tokens

STRUCTURE_RHYME_FEATURES: List[str] = [
    "token_count",
    "line_count",
    "mean_words_per_line",
    "std_words_per_line",
    "short_line_ratio",
    "end_rhyme_density",
    "couplet_rhyme_ratio",
    "identical_line_end_ratio",
]


@lru_cache(maxsize=65536)
def get_word_rhyme_parts(word: str) -> Set[str]:
    """Return the set of CMU phonetic rhyming parts for a word (handling informal -in' endings)."""
    w = word.lower().strip("'")
    if not w:
        return set()

    candidates = [w]
    if w.endswith("in") and len(w) > 3:
        candidates.append(w + "g")

    rhyme_parts: Set[str] = set()
    for cand in candidates:
        phones = pronouncing.phones_for_word(cand)
        for p in phones:
            rp = pronouncing.rhyming_part(p)
            if rp:
                rhyme_parts.add(rp)
        if rhyme_parts:
            break

    # Fallback suffix rhyme for slang/OOV words of length >= 3
    if not rhyme_parts and len(w) >= 3:
        rhyme_parts.add(f"__suffix_{w[-3:]}")
    return rhyme_parts


def words_phonetically_rhyme(w1: str, w2: str) -> bool:
    """Return True if w1 and w2 are distinct words that share a phonetic rhyming part."""
    w1_norm = w1.lower().strip("'")
    w2_norm = w2.lower().strip("'")
    if not w1_norm or not w2_norm or w1_norm == w2_norm:
        return False
    rp1 = get_word_rhyme_parts(w1_norm)
    rp2 = get_word_rhyme_parts(w2_norm)
    return bool(rp1 and rp2 and (rp1 & rp2))


def extract_structure_rhyme_features(text: str, window: int = 4) -> Dict[str, float]:
    """Extract Group 5 (Lyric Structure & Phonetic Rhyme) features for a single lyric text."""
    lines = get_lyric_lines(text)
    all_tokens = get_word_tokens(text, lowercase=True)
    n_tokens = len(all_tokens)
    n_lines = len(lines)

    if n_lines == 0 or n_tokens == 0:
        return {feat: 0.0 for feat in STRUCTURE_RHYME_FEATURES}

    line_token_lists = [get_word_tokens(ln, lowercase=True) for ln in lines]
    line_token_lists = [toks for toks in line_token_lists if toks]
    n_valid_lines = len(line_token_lists)

    if n_valid_lines == 0:
        return {feat: 0.0 for feat in STRUCTURE_RHYME_FEATURES}

    line_lengths = [len(toks) for toks in line_token_lists]
    mean_wpl = sum(line_lengths) / n_valid_lines
    if n_valid_lines > 1:
        variance = sum((x - mean_wpl) ** 2 for x in line_lengths) / (n_valid_lines - 1)
        std_wpl = math.sqrt(variance)
    else:
        std_wpl = 0.0

    short_line_ratio = sum(1 for x in line_lengths if x <= 4) / n_valid_lines

    # Line-ending words
    end_words = [toks[-1] for toks in line_token_lists]

    if n_valid_lines > 1:
        rhyming_lines = 0
        identical_end_lines = 0
        for i, w_i in enumerate(end_words):
            start_j = max(0, i - window)
            end_j = min(n_valid_lines, i + window + 1)
            has_rhyme = False
            has_identical = False
            for j in range(start_j, end_j):
                if i == j:
                    continue
                w_j = end_words[j]
                if w_i == w_j:
                    has_identical = True
                elif words_phonetically_rhyme(w_i, w_j):
                    has_rhyme = True
            if has_rhyme:
                rhyming_lines += 1
            if has_identical:
                identical_end_lines += 1

        end_rhyme_density = rhyming_lines / n_valid_lines
        identical_line_end_ratio = identical_end_lines / n_valid_lines

        couplets = sum(
            1
            for i in range(n_valid_lines - 1)
            if words_phonetically_rhyme(end_words[i], end_words[i + 1])
        )
        couplet_rhyme_ratio = couplets / (n_valid_lines - 1)
    else:
        end_rhyme_density = 0.0
        identical_line_end_ratio = 0.0
        couplet_rhyme_ratio = 0.0

    return {
        "token_count": float(n_tokens),
        "line_count": float(n_valid_lines),
        "mean_words_per_line": float(mean_wpl),
        "std_words_per_line": float(std_wpl),
        "short_line_ratio": float(short_line_ratio),
        "end_rhyme_density": float(end_rhyme_density),
        "couplet_rhyme_ratio": float(couplet_rhyme_ratio),
        "identical_line_end_ratio": float(identical_line_end_ratio),
    }

