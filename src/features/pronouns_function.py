"""Group 3: Function Words, Pronouns & Conversational Markers."""

from __future__ import annotations

from typing import Dict, List, Set

from src.features.preprocess import get_word_tokens

PRONOUN_FUNCTION_FEATURES: List[str] = [
    "function_word_ratio",
    "first_person_sg_ratio",
    "first_person_pl_ratio",
    "second_person_ratio",
    "third_person_ratio",
    "interjection_adlib_ratio",
]

FIRST_PERSON_SG: Set[str] = {
    "i", "me", "my", "mine", "myself",
    "i'm", "im", "i've", "ive", "i'll", "ill", "i'd", "id",
}

FIRST_PERSON_PL: Set[str] = {
    "we", "us", "our", "ours", "ourselves",
    "we're", "were", "we've", "weve", "we'll", "well", "we'd",
}

SECOND_PERSON: Set[str] = {
    "you", "your", "yours", "yourself", "yourselves",
    "u", "ya", "y'all", "yall", "ur",
    "you're", "youre", "you've", "youve", "you'll", "youll", "you'd", "youd",
}

THIRD_PERSON: Set[str] = {
    "he", "him", "his", "himself", "he's", "hes", "he'll", "he'd",
    "she", "her", "hers", "herself", "she's", "shes", "she'll", "she'd",
    "they", "them", "their", "theirs", "themselves",
    "they're", "theyre", "they've", "they'll", "they'd",
}

INTERJECTIONS_ADLIBS: Set[str] = {
    "yeah", "yea", "yep", "yup", "oh", "ooh", "oo", "uh", "uhh",
    "ah", "ahh", "ha", "haha", "hey", "ay", "ayy", "aye", "yo",
    "baby", "babe", "woah", "whoa", "wow", "na", "la", "mm", "mmm",
    "hmm", "huh", "shh", "damn", "god", "boy", "girl",
}

# Standard closed-class English function words (determiners, prepositions, auxiliaries, conjunctions, pronouns, negations)
FUNCTION_WORDS: Set[str] = (
    FIRST_PERSON_SG
    | FIRST_PERSON_PL
    | SECOND_PERSON
    | THIRD_PERSON
    | {
        "a", "an", "the", "this", "that", "these", "those",
        "it", "its", "it's", "itself",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "ain't", "aint", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
        "won't", "wont", "wouldn't", "can't", "cant", "couldn't", "shouldn't",
        "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
        "not", "no", "never", "nothing", "nobody", "none",
        "and", "or", "but", "if", "because", "cause", "'cause", "cuz", "cos",
        "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "only", "own", "same", "so",
        "than", "too", "very", "just", "now", "who", "whom", "whose", "which", "what",
    }
)


def extract_pronoun_function_features(text: str) -> Dict[str, float]:
    """Extract Group 3 (Function Words & Pronouns) features for a single lyric text."""
    tokens = get_word_tokens(text, lowercase=True)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return {feat: 0.0 for feat in PRONOUN_FUNCTION_FEATURES}

    fn_count = sum(1 for t in tokens if t in FUNCTION_WORDS)
    fp_sg_count = sum(1 for t in tokens if t in FIRST_PERSON_SG)
    fp_pl_count = sum(1 for t in tokens if t in FIRST_PERSON_PL)
    sp_count = sum(1 for t in tokens if t in SECOND_PERSON)
    tp_count = sum(1 for t in tokens if t in THIRD_PERSON)
    intj_count = sum(1 for t in tokens if t in INTERJECTIONS_ADLIBS)

    return {
        "function_word_ratio": float(fn_count / n_tokens),
        "first_person_sg_ratio": float(fp_sg_count / n_tokens),
        "first_person_pl_ratio": float(fp_pl_count / n_tokens),
        "second_person_ratio": float(sp_count / n_tokens),
        "third_person_ratio": float(tp_count / n_tokens),
        "interjection_adlib_ratio": float(intj_count / n_tokens),
    }

