"""Master Phase 2 pipeline: Extract all 5 interpretable feature groups and save train/test matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from src.features.lexical import LEXICAL_FEATURES, extract_lexical_features
from src.features.pronouns_function import (
    PRONOUN_FUNCTION_FEATURES,
    extract_pronoun_function_features,
)
from src.features.repetition import REPETITION_FEATURES, extract_repetition_features
from src.features.structure_rhyme import (
    STRUCTURE_RHYME_FEATURES,
    extract_structure_rhyme_features,
)
from src.features.syntactic import (
    SYNTACTIC_FEATURES,
    extract_syntactic_features_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_GROUPS: Dict[str, List[str]] = {
    "1_Lexical_Diversity_Readability": LEXICAL_FEATURES,
    "2_Repetition_Compressibility": REPETITION_FEATURES,
    "3_Function_Words_Pronouns": PRONOUN_FUNCTION_FEATURES,
    "4_Syntactic_POS": SYNTACTIC_FEATURES,
    "5_Structure_Rhyme": STRUCTURE_RHYME_FEATURES,
}

ALL_FEATURE_COLUMNS: List[str] = [
    feat for group_feats in FEATURE_GROUPS.values() for feat in group_feats
]


def extract_all_features_df(
    df: pd.DataFrame,
    text_col: str = "lyrics",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract all 5 stylometric feature groups for a DataFrame of songs."""
    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)

    non_spacy_rows: List[Dict[str, float]] = []
    iterator = range(n)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=n,
            desc="Extracting Lexical, Repetition, Pronoun & Rhyme features",
        )

    for idx in iterator:
        txt = texts[idx]
        row_feats: Dict[str, float] = {}
        row_feats.update(extract_lexical_features(txt))
        row_feats.update(extract_repetition_features(txt))
        row_feats.update(extract_pronoun_function_features(txt))
        row_feats.update(extract_structure_rhyme_features(txt))
        non_spacy_rows.append(row_feats)

    spacy_rows = extract_syntactic_features_batch(
        texts,
        batch_size=256,
        show_progress=show_progress,
    )

    feats_df = pd.DataFrame(non_spacy_rows, index=df.index)
    spacy_df = pd.DataFrame(spacy_rows, index=df.index)
    combined_feats = pd.concat([feats_df, spacy_df], axis=1)[ALL_FEATURE_COLUMNS]

    # Drop any existing feature columns in df before concatenating
    base_cols = [c for c in df.columns if c not in ALL_FEATURE_COLUMNS]
    return pd.concat([df[base_cols], combined_feats], axis=1)


def run_feature_pipeline(
    processed_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all 34 interpretable stylometric features on dataset_full.parquet,
    align with train.parquet and test.parquet, and save features_*.parquet.
    """
    proc_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    full_out = proc_dir / "features_full.parquet"
    train_out = proc_dir / "features_train.parquet"
    test_out = proc_dir / "features_test.parquet"

    if (
        not force_recompute
        and full_out.exists()
        and train_out.exists()
        and test_out.exists()
    ):
        print(f"[Phase 2 Cache] Loading precomputed features from {proc_dir}...")
        return {
            "full": pd.read_parquet(full_out),
            "train": pd.read_parquet(train_out),
            "test": pd.read_parquet(test_out),
        }

    full_df = pd.read_parquet(proc_dir / "dataset_full.parquet")
    train_df = pd.read_parquet(proc_dir / "train.parquet")
    test_df = pd.read_parquet(proc_dir / "test.parquet")

    print(
        f"[Phase 2] Extracting {len(ALL_FEATURE_COLUMNS)} features across 5 groups for {len(full_df)} tracks..."
    )
    features_full = extract_all_features_df(full_df, text_col="lyrics", show_progress=True)

    # Map extracted features onto train and test splits by match_key (avoiding duplicate computation)
    feat_lookup = features_full.set_index("match_key")[ALL_FEATURE_COLUMNS]
    features_train = train_df.join(feat_lookup, on="match_key")
    features_test = test_df.join(feat_lookup, on="match_key")

    features_full.to_parquet(full_out, index=False)
    features_train.to_parquet(train_out, index=False)
    features_test.to_parquet(test_out, index=False)

    print(
        f"[Phase 2 Saved] Features ({len(ALL_FEATURE_COLUMNS)} columns) saved to:\n"
        f"  - {full_out} ({len(features_full)} rows)\n"
        f"  - {train_out} ({len(features_train)} rows)\n"
        f"  - {test_out} ({len(features_test)} rows)"
    )

    return {
        "full": features_full,
        "train": features_train,
        "test": features_test,
    }


if __name__ == "__main__":
    run_feature_pipeline(force_recompute=True)

