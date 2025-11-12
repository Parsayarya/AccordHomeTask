#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMI-based contextual polysemy detector.

This script flags messages whose vocabulary is atypical for their assigned topic
within each community, using token–topic pointwise mutual information (PMI).

Inputs (CSV in ./data):
    - messages.csv (message_id, content, guild_id, date, ...)
    - message_topic_classifications.csv (message_id, topic_id, community_id,
      confidence_score, channel_id, date, ...)
    - topic.csv (id, name, definition, community_id, ...)
    - community.csv (id, name, ...)

Outputs:
    Creates a folder named "PMI-based Contextual polysemy/" containing one CSV
    per (community, topic) with messages flagged as outliers by PMI, and a
    summary CSV per topic.
"""

from __future__ import annotations

import os
import re
import math
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_ROOT = "PMI-based Contextual polysemy"

MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 30
MIN_TOKEN_COUNT_GLOBAL = 5
ADD_K = 1.0
OUTLIER_PERCENTILE = 5.0
Z_SCORE_THRESHOLD = -2.0
MAX_MESSAGES_PER_CSV: Optional[int] = None

STOPWORDS = set(
    """
    a an and are as at be by for from has have if in into is it its of on or our out so than that the their them then there these they this to too up was we were what when where which who will with you your i me my
    u r im ive dont cant wont didnt doesnt isnt wasnt hes shes theyre youre its thats
    lol lmao omg idk btw btw2 tbh gg brb afk
    http https www com net org discord cdn gif png jpg jpeg mp4
    """.split()
)

DOMAIN_NOISE = set(
    """
    pls pm dm
    """.split()
)

STOPWORDS |= DOMAIN_NOISE

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
CODE_RE = re.compile(r"`[^`]*`|```[\s\S]*?```")
MENTION_RE = re.compile(r"<@!?[\d]+>|<#[\d]+>|<@&[\d]+>|@[^\s]+|#[^\s]+")
EMOJI_RE = re.compile(r":[a-zA-Z0-9_~+-]+:")
PUNCT_RE = re.compile(r"[^a-z0-9']+")


def normalize(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = CODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text.lower())
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize(text)
    out: List[str] = []
    for tok in text.split():
        if tok in STOPWORDS:
            continue
        if len(tok) < MIN_TOKEN_LEN or len(tok) > MAX_TOKEN_LEN:
            continue
        tok = tok.strip("'")
        if not tok or tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def slugify(s: Optional[str]) -> str:
    if s is None:
        return "none"
    s2 = unicodedata.normalize("NFKC", s).lower()
    s2 = re.sub(r"[^a-z0-9]+", "-", s2).strip("-")
    return s2 or "none"


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    messages = pd.read_csv(os.path.join(data_dir, "messages.csv"), dtype=str, keep_default_na=False)
    mtc = pd.read_csv(os.path.join(data_dir, "message_topic_classifications.csv"), dtype=str, keep_default_na=False)
    topics = pd.read_csv(os.path.join(data_dir, "topic.csv"), dtype=str, keep_default_na=False)
    communities = pd.read_csv(os.path.join(data_dir, "community.csv"), dtype=str, keep_default_na=False)

    for df, col in [(messages, "message_id"), (mtc, "message_id")]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "confidence_score" in mtc.columns:
        mtc["confidence_score"] = pd.to_numeric(mtc["confidence_score"], errors="coerce")

    for df in (messages, mtc, topics, communities):
        df.replace({"": np.nan}, inplace=True)

    return messages, mtc, topics, communities


def build_token_counts_per_community(
    messages_in_comm: pd.DataFrame,
    mtc_in_comm: pd.DataFrame,
    topic_ids_in_comm: Sequence[str],
) -> Tuple[Counter, Dict[str, Counter], Dict[str, int], int, Dict[str, List[str]]]:
    msg_tokens: Dict[str, List[str]] = {}
    contents = messages_in_comm.set_index("message_id")["content"]
    for mid, content in contents.items():
        msg_tokens[mid] = tokenize(content)

    bg_token_counts: Counter = Counter()
    topic_token_counts: Dict[str, Counter] = {tid: Counter() for tid in topic_ids_in_comm}

    mtc_sorted = mtc_in_comm.copy()
    if "confidence_score" in mtc_sorted.columns:
        mtc_sorted = mtc_sorted.sort_values(
            by=["message_id", "confidence_score"], ascending=[True, False]
        )
    mtc_top = mtc_sorted.drop_duplicates(subset=["message_id"], keep="first")

    for _, row in mtc_top.iterrows():
        mid = row["message_id"]
        tid = row["topic_id"]
        if mid not in msg_tokens or tid not in topic_token_counts:
            continue        # skip messages with no tokens or topics absent
        toks = msg_tokens[mid]
        topic_token_counts[tid].update(toks)
        bg_token_counts.update(toks)

    topic_token_totals: Dict[str, int] = {tid: int(sum(cnt.values())) for tid, cnt in topic_token_counts.items()}
    bg_token_total = int(sum(bg_token_counts.values()))

    return bg_token_counts, topic_token_counts, topic_token_totals, bg_token_total, msg_tokens


def compute_pmi_tables(
    bg_counts: Counter,
    topic_counts: Dict[str, Counter],
    topic_totals: Dict[str, int],
    bg_total: int,
    min_global_count: int = MIN_TOKEN_COUNT_GLOBAL,
    add_k: float = ADD_K,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Set[str], Dict[str, float]]:
    vocab: Set[str] = {t for t, c in bg_counts.items() if c >= min_global_count}
    if bg_total == 0:
        return {tid: {} for tid in topic_counts}, {tid: 0.0 for tid in topic_counts}, set(), {}

    V = len(vocab)
    bg_probs: Dict[str, float] = {tok: (bg_counts.get(tok, 0) + add_k) / (bg_total + add_k * V) for tok in vocab}

    p_topic: Dict[str, float] = {
        tid: (topic_totals.get(tid, 0) + add_k) / (bg_total + add_k * len(topic_totals)) for tid in topic_totals
    }

    pmi: Dict[str, Dict[str, float]] = {tid: {} for tid in topic_counts}
    for tid, cnts in topic_counts.items():
        tot_t = topic_totals.get(tid, 0)
        if tot_t == 0:
            continue
        for tok in vocab:
            p_tok_topic = (cnts.get(tok, 0) + add_k) / (bg_total + add_k * V)
            p_tok = bg_probs[tok]
            p_t = p_topic[tid]
            denom = max(p_tok * p_t, 1e-12)
            ratio = max(p_tok_topic / denom, 1e-12)
            pmi[tid][tok] = math.log(ratio, 2)

    return pmi, p_topic, vocab, bg_probs


def score_messages_for_topic(
    msg_ids: Sequence[str],
    assigned_topic_id: str,
    msg_tokens: Dict[str, List[str]],
    pmi_table: Dict[str, Dict[str, float]],
    vocab: Set[str],
) -> pd.DataFrame:
    rows: List[Tuple[str, float, int]] = []
    pmi_for_topic = pmi_table.get(assigned_topic_id, {})
    for mid in msg_ids:
        toks = msg_tokens.get(mid, [])
        sel = [t for t in toks if t in vocab]
        if not sel:
            score = float("nan")
        else:
            vals = [pmi_for_topic.get(t, 0.0) for t in sel]
            score = float(np.mean(vals)) if vals else float("nan")
        rows.append((mid, score, len(sel)))
    return pd.DataFrame(rows, columns=["message_id", "score_pmi", "token_count_used"])


def flag_outliers(
    scores: pd.Series,
    percentile: float = OUTLIER_PERCENTILE,
    z_thresh: float = Z_SCORE_THRESHOLD,
) -> Tuple[pd.Series, pd.Series, float]:
    s = scores.astype(float)
    mu, sigma = s.mean(skipna=True), s.std(skipna=True, ddof=0)
    z = (s - mu) / (sigma if sigma and sigma > 0 else 1.0)
    cutoff = np.nanpercentile(s, percentile) if s.notna().any() else np.nan
    outlier_mask = (z < z_thresh) | (s <= cutoff)
    return outlier_mask, z, float(cutoff)


os.makedirs(OUTPUT_ROOT, exist_ok=True)
messages, mtc, topics, communities = load_data(DATA_DIR)

messages = messages.loc[messages["message_id"].notna()]
if "content" not in messages.columns:
    raise ValueError("messages.csv must include a 'content' column.")
messages["content"] = messages["content"].fillna("").astype(str)

# Mappings: topic name and NEW: topic definition
topic_name = topics.set_index("id")["name"].to_dict()
topic_def = topics.set_index("id")["definition"].to_dict() if "definition" in topics.columns else {}

comm_name = communities.set_index("id")["name"].to_dict()

mtc = mtc.loc[
    mtc["message_id"].notna() & mtc["topic_id"].notna() & mtc["community_id"].notna()
]
if "confidence_score" in mtc.columns:
    mtc = mtc.sort_values(
        by=["message_id", "community_id", "topic_id", "confidence_score"],
        ascending=[True, True, True, False],
    ).drop_duplicates(subset=["message_id", "community_id"], keep="first")

communities_in_data = sorted(mtc["community_id"].dropna().unique().tolist())

for cid in communities_in_data:
    comm_dir = os.path.join(OUTPUT_ROOT, f"{slugify(comm_name.get(cid, cid))}__{cid}")
    os.makedirs(comm_dir, exist_ok=True)

    mtc_c = mtc[mtc["community_id"] == cid].copy()
    msg_ids_c = set(mtc_c["message_id"].unique().tolist())
    messages_c = messages[messages["message_id"].isin(msg_ids_c)].copy()
    topic_ids_c = sorted(mtc_c["topic_id"].dropna().unique().tolist())

    (
        bg_counts,
        topic_counts,
        topic_totals,
        bg_total,
        msg_tokens,
    ) = build_token_counts_per_community(messages_c, mtc_c, topic_ids_c)

    pmi_tbl, p_topic, vocab, bg_probs = compute_pmi_tables(
        bg_counts,
        topic_counts,
        topic_totals,
        bg_total,
        min_global_count=MIN_TOKEN_COUNT_GLOBAL,
        add_k=ADD_K,
    )

    for tid in topic_ids_c:
        mtc_ct = mtc_c[mtc_c["topic_id"] == tid].copy()
        mids = mtc_ct["message_id"].tolist()

        df_scores = score_messages_for_topic(mids, tid, msg_tokens, pmi_tbl, vocab)
        df_join = mtc_ct.merge(df_scores, on="message_id", how="left")
        df_join = df_join.merge(
            messages_c[["message_id", "content", "channel_id", "date", "guild_id"]],
            on="message_id",
            how="left",
            suffixes=("", "_msg"),
        )

        out_mask, z, cutoff = flag_outliers(df_join["score_pmi"])
        s = df_join["score_pmi"].astype(float)
        ranks = s.rank(pct=True, method="average")

        df_join["score_z"] = z
        df_join["percentile_rank"] = ranks
        df_join["outlier_rule"] = (df_join["score_z"] < Z_SCORE_THRESHOLD) | (
            df_join["score_pmi"] <= cutoff
        )
        df_join["community_name"] = comm_name.get(cid, cid)
        df_join["topic_name"] = topic_name.get(tid, tid)
        df_join["topic_definition"] = topic_def.get(tid, "")  # <-- NEW

        outliers = df_join[df_join["outlier_rule"]].copy()
        outliers = outliers.sort_values(
            by=["score_pmi", "score_z", "percentile_rank"], ascending=[True, True, True]
        )

        cols = [
            "community_id",
            "community_name",
            "topic_id",
            "topic_name",
            "topic_definition",  # <-- NEW
            "message_id",
            "score_pmi",
            "score_z",
            "percentile_rank",
            "token_count_used",
            "confidence_score",
            "content",
            "channel_id",
            "guild_id",
            "date",
        ]
        keep_cols = [c for c in cols if c in outliers.columns]
        outliers = outliers[keep_cols]

        if MAX_MESSAGES_PER_CSV is not None and len(outliers) > MAX_MESSAGES_PER_CSV:
            outliers = outliers.head(MAX_MESSAGES_PER_CSV)

        topic_slug = slugify(topic_name.get(tid, tid))
        out_path = os.path.join(
            comm_dir,
            f"topic_{tid}__{topic_slug}__contextual-outliers.csv",
        )
        outliers.to_csv(out_path, index=False, encoding="utf-8")

        summary = pd.DataFrame(
            {
                "metric": [
                    "community_tokens",
                    "topic_tokens",
                    "vocab_size",
                    "p(topic)",
                ],
                "value": [
                    int(sum(bg_counts.values())),
                    int(topic_totals.get(tid, 0)),
                    int(len(vocab)),
                    float(p_topic.get(tid, 0.0)),
                ],
            }
        )
        summary_path = os.path.join(
            comm_dir,
            f"topic_{tid}__{topic_slug}__summary.csv",
        )
        summary.to_csv(summary_path, index=False, encoding="utf-8")
