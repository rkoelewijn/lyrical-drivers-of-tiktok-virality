"""Group 4: Syntactic / Part-of-Speech (POS) Tag Proportions via spaCy."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import spacy
from spacy.language import Language
from tqdm.auto import tqdm

from src.features.preprocess import clean_lyric_text

SYNTACTIC_FEATURES: List[str] = [
    "pos_noun_ratio",
    "pos_verb_ratio",
    "pos_adj_ratio",
    "pos_adv_ratio",
    "pos_pron_ratio",
    "pos_intj_ratio",
    "pos_adp_det_ratio",
    "verb_to_noun_ratio",
]

_NLP_INSTANCE: Optional[Language] = None


def get_spacy_pos_pipeline() -> Language:
    """Load spaCy en_core_web_sm with parser, ner, and lemmatizer disabled for fast POS tagging."""
    global _NLP_INSTANCE
    if _NLP_INSTANCE is None:
        _NLP_INSTANCE = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner", "lemmatizer"],
        )
    return _NLP_INSTANCE


def _doc_to_pos_features(doc) -> Dict[str, float]:
    """Compute Universal POS ratios from a processed spaCy Doc."""
    pos_tags = [
        tok.pos_
        for tok in doc
        if not tok.is_space and not tok.is_punct and tok.is_alpha
    ]
    n_tokens = len(pos_tags)
    if n_tokens == 0:
        return {feat: 0.0 for feat in SYNTACTIC_FEATURES}

    counts = Counter(pos_tags)
    nouns = counts.get("NOUN", 0) + counts.get("PROPN", 0)
    verbs = counts.get("VERB", 0) + counts.get("AUX", 0)
    adjs = counts.get("ADJ", 0)
    advs = counts.get("ADV", 0)
    prons = counts.get("PRON", 0)
    intjs = counts.get("INTJ", 0)
    adp_dets = counts.get("ADP", 0) + counts.get("DET", 0)

    return {
        "pos_noun_ratio": float(nouns / n_tokens),
        "pos_verb_ratio": float(verbs / n_tokens),
        "pos_adj_ratio": float(adjs / n_tokens),
        "pos_adv_ratio": float(advs / n_tokens),
        "pos_pron_ratio": float(prons / n_tokens),
        "pos_intj_ratio": float(intjs / n_tokens),
        "pos_adp_det_ratio": float(adp_dets / n_tokens),
        "verb_to_noun_ratio": float(verbs / (nouns + 1.0)),
    }


def extract_syntactic_features(
    text: str, nlp: Optional[Language] = None
) -> Dict[str, float]:
    """Extract Group 4 (Syntactic / POS) features for a single lyric text."""
    if nlp is None:
        nlp = get_spacy_pos_pipeline()
    cleaned = clean_lyric_text(text)
    if not cleaned:
        return {feat: 0.0 for feat in SYNTACTIC_FEATURES}
    doc = nlp(cleaned)
    return _doc_to_pos_features(doc)


def extract_syntactic_features_batch(
    texts: Iterable[str],
    batch_size: int = 256,
    show_progress: bool = True,
) -> List[Dict[str, float]]:
    """Extract Group 4 (Syntactic / POS) features in batches via nlp.pipe."""
    nlp = get_spacy_pos_pipeline()
    cleaned_texts = [clean_lyric_text(t) for t in texts]
    results: List[Dict[str, float]] = []

    iterator = nlp.pipe(cleaned_texts, batch_size=batch_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(cleaned_texts),
            desc="Extracting POS features (spaCy)",
        )

    for doc in iterator:
        results.append(_doc_to_pos_features(doc))
    return results

