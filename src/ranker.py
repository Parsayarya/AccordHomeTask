"""
Ambiguity Ranker

- Builds ambiguity features per message.
- Ranks messages by engineered ambiguity_target score.
- Flags ambiguous messages: multi-topic AND score >= per-community quantile.
- Writes outputs/top_ambiguous_ranked.csv.
"""

import pathlib
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RANKED = OUTPUT_DIR / "top_ambiguous_ranked.csv"


def _safe_entropy_from_array(scores: np.ndarray) -> float:
    """Normalized entropy in [0,1]. If <2 positive probs, return 0.0."""
    arr = np.asarray(scores, dtype=float)
    total = arr.sum()
    if total <= 0.0:
        return 0.0
    probs = arr / total
    p = probs[probs > 0]
    k = p.size
    if k <= 1:
        return 0.0
    ent = -(p * np.log(p)).sum()
    return float(ent / math.log(k))


def _top2_margin(scores: np.ndarray) -> float:
    """Margin = top1 - top2. If <2 scores, treat as unambiguous by margin."""
    arr = np.sort(np.asarray(scores, dtype=float))[::-1]
    if arr.size == 0:
        return 1.0
    if arr.size == 1:
        return float(arr[0])
    return float(arr[0] - arr[1])


def _clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df[c] = s
    return df



def build_ambiguity_frame(df: pd.DataFrame, messages_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    d.columns = [c.lower().strip() for c in d.columns]
    messages_df = messages_df.copy()
    messages_df.columns = [c.lower().strip() for c in messages_df.columns]

    req = {"message_id", "topic_id", "community_id", "confidence_score"}
    if not req.issubset(d.columns):
        missing = req - set(d.columns)
        raise ValueError(f"df missing required columns: {missing}")
    if not {"message_id", "content"}.issubset(messages_df.columns):
        raise ValueError("messages_df must contain: message_id, content")

    grp = d.groupby("message_id")

    g = grp.agg(
        community_id=("community_id", "first"),
        guild_id=("guild_id", "first"),
        channel_id=("channel_id", "first"),
        topic_count=("topic_id", "nunique"),
        max_conf=("confidence_score", "max"),
        sum_conf=("confidence_score", "sum"),
        mean_conf=("confidence_score", "mean"),
        std_conf=("confidence_score", "std"),
    ).reset_index()

    margin = grp["confidence_score"].apply(lambda s: _top2_margin(s.values)).rename("margin")
    g = g.merge(margin, on="message_id", how="left")

    entropy = grp["confidence_score"].apply(lambda s: _safe_entropy_from_array(s.values)).rename("entropy")
    g = g.merge(entropy, on="message_id", how="left")

    txt = messages_df[["message_id", "content"]].copy()
    txt["content"] = txt["content"].astype(str)
    txt["len_chars"] = txt["content"].str.len().clip(upper=10000)
    txt["len_words"] = txt["content"].str.split().map(len).clip(upper=2000)
    g = g.merge(txt, on="message_id", how="left")

    g["has_multi"] = (g["topic_count"] > 1).astype(int)
    g["std_conf"] = g["std_conf"].fillna(0.0)
    g["margin_clamped"] = g["margin"].clip(lower=0.0, upper=1.0)
    g["inv_margin"] = 1.0 - g["margin_clamped"]

    feature_cols = [
        "topic_count",
        "max_conf",
        "mean_conf",
        "std_conf",
        "margin_clamped",
        "inv_margin",
        "entropy",
        "len_chars",
        "len_words",
        "has_multi",
    ]
    g = _clean_numeric(g, feature_cols)

    # Continuous target in [0,1] - this becomes our rank_score
    g["ambiguity_target"] = (
        0.50 * g["entropy"] +
        0.35 * g["inv_margin"] +
        0.15 * (g["topic_count"].clip(upper=5) - 1) / 4.0
    ).clip(lower=0.0, upper=1.0)

    g["ambiguity_target"] = pd.to_numeric(g["ambiguity_target"], errors="coerce") \
        .replace([np.inf, -np.inf], np.nan) \
        .fillna(0.0)

    g["community_id_str"] = g["community_id"].astype(str)
    return g, feature_cols



def score_and_flag(
    full_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    top_quantile: float = 0.9,
    top_n_per_community: int = 200
) -> pd.DataFrame:
    """
    Score messages using the ambiguity_target directly (no ML model).
    Flag ambiguous messages using the same logic as before.
    """
    df_sorted = full_df.sort_values(["community_id_str", "message_id"]).reset_index(drop=True)
    
    # Use ambiguity_target directly as rank_score (no ML model)
    df_sorted["rank_score"] = df_sorted["ambiguity_target"]

    # Per-community threshold at given quantile
    comm_thresh = df_sorted.groupby("community_id_str")["rank_score"].quantile(top_quantile)
    df_sorted = df_sorted.merge(comm_thresh.rename("comm_thresh"), on="community_id_str", how="left")

    # Exact same flagging rule
    df_sorted["is_ambiguous"] = (
        (df_sorted["has_multi"] == 1) &
        (df_sorted["rank_score"] >= df_sorted["comm_thresh"])
    ).astype(int)

    df_sorted["rank_within_community"] = (
        df_sorted.groupby("community_id_str")["rank_score"]
        .rank(ascending=False, method="first").astype(int)
    )

    cols = [
        "community_id", "message_id", "rank_score", "rank_within_community",
        "topic_count", "max_conf", "mean_conf", "margin_clamped", "entropy",
        "len_chars", "len_words", "has_multi", "ambiguity_target", "is_ambiguous", "content",
    ]
    out = df_sorted[cols].copy()

    topk = out[out["rank_within_community"] <= top_n_per_community].copy()
    topk.to_csv(OUT_RANKED.as_posix(), index=False)
    return out


def train_and_rank_ambiguity(
    df: pd.DataFrame,
    messages_df: pd.DataFrame,
    *,
    top_quantile: float = 0.9,
    top_n_per_community: int = 200
) -> pd.DataFrame:
    """
    Simplified version: no ML training, just use engineered features directly.
    """
    feats_df, feature_cols = build_ambiguity_frame(df, messages_df)
    # No model training - go straight to scoring
    scored = score_and_flag(
        feats_df,
        feature_cols,
        top_quantile=top_quantile,
        top_n_per_community=top_n_per_community,
    )
    return scored


if __name__ == "__main__":
    df = pd.read_csv("data/message_topic_classifications.csv")
    messages_df = pd.read_csv("data/messages.csv", usecols=["message_id", "content"])
    scored = train_and_rank_ambiguity(df, messages_df, top_quantile=0.9, top_n_per_community=200)
    print(f"Wrote: {OUT_RANKED}")
