"""Statistical validation utilities: 95% Confidence Intervals and Chi-Square (X^2) vocabulary tests."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.features.preprocess import get_word_tokens


def compute_feature_confidence_intervals(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Compute 95% Confidence Intervals (mean +/- 1.96 * SE), Cohen's d, and Welch's t-test
    comparing Viral (label=1) vs. Non-Viral (label=0) across all stylometric feature groups.
    """
    viral_df = df[df[label_col] == 1]
    non_viral_df = df[df[label_col] == 0]

    rows = []
    for group_name, feat_list in feature_groups.items():
        for feat in feat_list:
            v_vals = viral_df[feat].dropna().to_numpy(dtype=float)
            nv_vals = non_viral_df[feat].dropna().to_numpy(dtype=float)

            n1, n0 = len(v_vals), len(nv_vals)
            if n1 < 2 or n0 < 2:
                continue

            m1, s1 = float(np.mean(v_vals)), float(np.std(v_vals, ddof=1))
            m0, s0 = float(np.mean(nv_vals)), float(np.std(nv_vals, ddof=1))

            se1 = s1 / np.sqrt(n1)
            se0 = s0 / np.sqrt(n0)

            ci1_low, ci1_high = m1 - 1.96 * se1, m1 + 1.96 * se1
            ci0_low, ci0_high = m0 - 1.96 * se0, m0 + 1.96 * se0

            diff = m1 - m0
            se_diff = float(np.sqrt(se1**2 + se0**2))
            diff_ci_low, diff_ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

            pooled_sd = float(
                np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
            )
            cohens_d = diff / pooled_sd if pooled_sd > 0 else 0.0

            t_stat, p_val = stats.ttest_ind(v_vals, nv_vals, equal_var=False)
            u_stat, mw_pval = stats.mannwhitneyu(
                v_vals, nv_vals, alternative="two-sided"
            )

            ci_non_overlapping = (ci1_low > ci0_high) or (ci0_low > ci1_high)
            diff_significant_95 = (diff_ci_low > 0) or (diff_ci_high < 0)

            rows.append(
                {
                    "feature_group": group_name,
                    "feature": feat,
                    "viral_mean": m1,
                    "viral_std": s1,
                    "viral_ci95_low": ci1_low,
                    "viral_ci95_high": ci1_high,
                    "non_viral_mean": m0,
                    "non_viral_std": s0,
                    "non_viral_ci95_low": ci0_low,
                    "non_viral_ci95_high": ci0_high,
                    "diff_mean": diff,
                    "diff_ci95_low": diff_ci_low,
                    "diff_ci95_high": diff_ci_high,
                    "cohens_d": cohens_d,
                    "abs_cohens_d": abs(cohens_d),
                    "welch_t": float(t_stat),
                    "p_value": float(p_val),
                    "mannwhitney_p": float(mw_pval),
                    "ci_non_overlapping": bool(ci_non_overlapping),
                    "diff_95ci_excludes_zero": bool(diff_significant_95),
                }
            )

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values("abs_cohens_d", ascending=False).reset_index(
            drop=True
        )
    return res_df


def compute_chi_square_vocabulary(
    df: pd.DataFrame,
    text_col: str = "lyrics",
    label_col: str = "label",
    min_doc_freq: int = 30,
    top_k: Optional[int] = 50,
) -> pd.DataFrame:
    """
    Compute 2x2 Chi-Square (X^2) test of independence (Lecture 6) across document presence
    for vocabulary terms in Viral (label=1) vs. Non-Viral (label=0) songs.

    2x2 Contingency table for term t:
      A = # Viral songs containing t
      B = # Non-Viral songs containing t
      C = # Viral songs NOT containing t (N_1 - A)
      D = # Non-Viral songs NOT containing t (N_0 - B)
      X^2 = N * (A*D - B*C)^2 / ((A+C)*(B+D)*(A+B)*(C+D))
    """
    viral_mask = df[label_col] == 1
    n1 = int(viral_mask.sum())
    n0 = int((~viral_mask).sum())
    n_total = n1 + n0

    viral_df_counts: Counter = Counter()
    non_viral_df_counts: Counter = Counter()

    for text, is_viral in zip(df[text_col], viral_mask):
        unique_tokens = {
            t for t in get_word_tokens(str(text), lowercase=True) if len(t) >= 2
        }
        if is_viral:
            viral_df_counts.update(unique_tokens)
        else:
            non_viral_df_counts.update(unique_tokens)

    all_vocab = set(viral_df_counts.keys()) | set(non_viral_df_counts.keys())

    rows = []
    for term in all_vocab:
        a = viral_df_counts.get(term, 0)
        b = non_viral_df_counts.get(term, 0)
        doc_freq = a + b
        if doc_freq < min_doc_freq or doc_freq >= n_total:
            continue

        c = n1 - a
        d = n0 - b

        denom = (a + c) * (b + d) * (a + b) * (c + d)
        if denom == 0:
            continue

        ad_bc = (a * d) - (b * c)
        chi2 = (n_total * (ad_bc**2)) / denom
        p_val = float(stats.chi2.sf(chi2, df=1))

        viral_rate = a / n1
        non_viral_rate = b / n0
        direction = "Viral (label=1)" if viral_rate > non_viral_rate else "Non-Viral (label=0)"

        rows.append(
            {
                "term": term,
                "chi2": float(chi2),
                "p_value": p_val,
                "direction": direction,
                "viral_doc_count": a,
                "non_viral_doc_count": b,
                "viral_doc_pct": 100.0 * viral_rate,
                "non_viral_doc_pct": 100.0 * non_viral_rate,
                "rate_ratio": (viral_rate + 1e-6) / (non_viral_rate + 1e-6),
            }
        )

    res_df = pd.DataFrame(rows).sort_values("chi2", ascending=False).reset_index(drop=True)
    if top_k is not None and not res_df.empty:
        return res_df.head(top_k)
    return res_df

