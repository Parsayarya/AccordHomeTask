"""
semantic neighbor retrieval + Gemini classification.

Pipeline:
  1) Sample 100 messages that have >1 assigned topic_id.
  2) Merge in community supplementary_context (df2.id -> df.community_id).
  3) For each sampled message, retrieve top-10 nearest single-topic,
     high-confidence examples via GPU cosine similarity over sentence embeddings.
  4) Build a Gemini prompt (includes targets from df3 name/definition, plus examples).
  5) Call Gemini and write outputs/gemini_classification_candidates.csv.
  - df: classifications with columns as provided (message_id, topic_id, community_id,
        confidence_score, guild_id, model_version, classified_at, date, author_id, channel_id)
  - messages_df: at least (message_id, content)
  - df2: community table (id, supplementary_context)
  - df3: topic table (id, name, definition)
"""
import os
import re
import json
import math
import pathlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer

#Conf
RANDOM_STATE = 42
SAMPLE_N = 100
CONF_THRESH = 0.9           # high-confidence single-topic cutoff
MAX_EXAMPLES_PER_MSG = 10    # neighbors to include in prompt
BATCH_SIZE = 2048            # embedding batch size
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RESTRICT_NEIGHBORS_SAME_COMMUNITY = True
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "gemini_classification_candidates.csv"

_WORD_RE = re.compile(r"[a-zA-Z]+")


# Text utils
def normalize_text(s: str) -> str:
    """Lowercase + keep only a-z tokens to stabilize embeddings slightly."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    tokens = _WORD_RE.findall(s)
    return " ".join(tokens)


def truncate(s: str, max_chars: int = 2000) -> str:
    if not isinstance(s, str):
        return ""
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."

    
# Topic metadata
def build_topic_lookup(df3: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    t = df3.copy()
    t.columns = [c.strip().lower() for c in t.columns]
    if "id" in t.columns:
        t = t.rename(columns={"id": "topic_id"})
    for col in ("name", "definition"):
        if col in t.columns:
            t[col] = t[col].astype(str).str.strip()
    return {
        str(row["topic_id"]): {
            "name": row.get("name", ""),
            "definition": row.get("definition", "")
        }
        for _, row in t.iterrows()
    }


def top_two_topics_for_message(df_multi: pd.DataFrame, message_id: int) -> List[Tuple[str, float]]:
    sub = df_multi.loc[df_multi["message_id"] == message_id, ["topic_id", "confidence_score"]]
    sub = sub.sort_values("confidence_score", ascending=False)
    out: List[Tuple[str, float]] = []
    for _, r in sub.head(2).iterrows():
        out.append((str(r["topic_id"]), float(r["confidence_score"])))
    while len(out) < 2:
        out.append(("", float("nan")))
    return out


# GPU embeddings + similarity
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def embed_texts(model: SentenceTransformer, texts: List[str], device: torch.device, batch_size: int) -> Tensor:
    """
    Returns L2-normalized embeddings as a torch.float32 tensor on device.
    """
    embs: List[Tensor] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        chunk = texts[i : i + batch_size]
        # sentence-transformers returns np.ndarray; convert to torch
        arr = model.encode(
            chunk,
            batch_size=min(512, batch_size),   # internal micro-batching by the library
            convert_to_numpy=True,
            normalize_embeddings=False,        # we'll normalize in torch for stability
            device=str(device)
        )
        t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        t = torch.nn.functional.normalize(t, p=2, dim=1)
        embs.append(t)
    return torch.cat(embs, dim=0) if embs else torch.empty(0, model.get_sentence_embedding_dimension(), device=device)


@torch.no_grad()
def topk_similar(
    query_embs: Tensor,
    corpus_embs: Tensor,
    k: int,
    same_comm_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    query_embs: (Q, D), corpus_embs: (N, D) both L2-normalized.
    same_comm_mask: optional (Q, N) where 1 keeps, 0 masks out.
    Returns (topk_scores, topk_indices) with shape (Q, k).
    """
    # Cosine with normalized vectors: just dot-product
    sims = query_embs @ corpus_embs.T  # (Q, N)

    if same_comm_mask is not None:
        # Mask disallowed entries by setting to -inf
        sims = sims.masked_fill(same_comm_mask == 0, float("-inf"))

    # Top-k per row
    scores, idx = torch.topk(sims, k=k, dim=1)
    return scores, idx


# Gemini prompt + call
def make_targets_block(topic_ids: List[str], topic_meta: Dict[str, Dict[str, str]]) -> str:
    lines = []
    for tid in topic_ids:
        meta = topic_meta.get(str(tid), {"name": tid, "definition": ""})
        nm = meta.get("name", str(tid))
        df = meta.get("definition", "")
        lines.append(f"- {nm}  ::  {df}")
    return "\n".join(lines)


def make_examples_block(examples: List[Tuple[int, str, str, float]]) -> str:
    rows = []
    for mid, text, tname, conf in examples:
        rows.append(f"[id={mid}] ({tname}, conf={conf:.3f}) :: {text}")
    return "\n".join(rows)


def build_gemini_prompt(
    message_text: str,
    community_context: str,
    topic_ids_for_prompt: List[str],
    topic_meta: Dict[str, Dict[str, str]],
    example_rows: List[Tuple[int, str, str, float]]
) -> str:
    system_rules = (
        "You are a careful message classifier for gaming Discord data. "
        "Pick exactly ONE topic from the provided target list. "
        "Return ONLY a compact JSON object with keys: predicted_topic_name, predicted_topic_id, confidence."
    )
    output_format = (
        '{ "predicted_topic_name": "<string>", '
        '"predicted_topic_id": "<string>", '
        '"confidence": <float between 0 and 1> }'
    )
    topics_block = make_targets_block(topic_ids_for_prompt, topic_meta)
    examples_block = make_examples_block(example_rows)

    prompt = f"""
{system_rules}

Community supplementary context (if any):
{community_context.strip() if isinstance(community_context, str) else ""}

Message to classify:
{truncate(message_text, 2000)}

Candidate topics (name :: definition):
{topics_block}

Nearest high-confidence single-topic examples:
{examples_block}

Respond with ONLY this JSON:
{output_format}
""".strip()
    return prompt


def try_parse_json_object(s: str) -> Dict[str, str]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                pass
    return {"predicted_topic_name": "", "predicted_topic_id": "", "confidence": None}


def call_gemini(prompt: str) -> Dict[str, str]:
    from google import genai
    client = genai.Client()  # GEMINI_API_KEY must be set in env
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    txt = getattr(resp, "text", "").strip()
    return try_parse_json_object(txt)


def run_pipeline_cuda_semantic(
    df: pd.DataFrame,
    messages_df: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    sample_n: int = SAMPLE_N,
    conf_thresh: float = CONF_THRESH,
    neighbor_k: int = MAX_EXAMPLES_PER_MSG,
    restrict_neighbors_same_community: bool = RESTRICT_NEIGHBORS_SAME_COMMUNITY,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE
) -> pd.DataFrame:

    # Normalize column names
    for d in (df, messages_df, df2, df3):
        d.columns = [c.strip().lower() for c in d.columns]

    # Join community supplementary_context
    if not {"id", "supplementary_context"}.issubset(df2.columns):
        raise ValueError("df2 must contain columns: id, supplementary_context")
    work = df.merge(df2[["id", "supplementary_context"]].rename(columns={"id": "community_id"}),
                    on="community_id", how="left")

    # Attach message content
    if not {"message_id", "content"}.issubset(messages_df.columns):
        raise ValueError("messages_df must contain: message_id, content")
    work = work.merge(messages_df[["message_id", "content"]], on="message_id", how="left")

    # Topic metadata
    topic_meta = build_topic_lookup(df3)

    # Identify multi-topic message IDs
    topics_per_msg = work.groupby("message_id")["topic_id"].nunique()
    multi_ids = topics_per_msg[topics_per_msg > 1].index
    multi_df = work[work["message_id"].isin(multi_ids)].copy()

    # Sample 100 unique multi-topic message_ids
    sampled_msg_ids = (
        multi_df[["message_id"]]
        .drop_duplicates()
        .sample(n=min(sample_n, multi_df["message_id"].nunique()), random_state=RANDOM_STATE)
        .message_id
        .tolist()
    )
    sampled_multi = multi_df[multi_df["message_id"].isin(sampled_msg_ids)].copy()

    # Build single-topic, high-confidence corpus
    all_counts = work.groupby("message_id")["topic_id"].nunique()
    single_ids = all_counts[all_counts == 1].index
    single_df = work[work["message_id"].isin(single_ids)].copy()
    single_df = single_df[single_df["confidence_score"] >= conf_thresh]
    single_df = single_df.sort_values("confidence_score", ascending=False)

    # Prepare query and corpus tables
    sampled_texts = (
        sampled_multi[["message_id", "community_id", "content", "supplementary_context"]]
        .drop_duplicates("message_id")
        .assign(norm=lambda x: x["content"].map(normalize_text))
        .reset_index(drop=True)
    )

    # For single_df, we need one row per message_id with text/topic/conf/community
    single_core = (
        single_df.sort_values(["message_id", "confidence_score"], ascending=[True, False])
        .drop_duplicates("message_id")
        .loc[:, ["message_id", "community_id", "topic_id", "confidence_score", "content"]]
        .assign(norm=lambda x: x["content"].map(normalize_text))
        .reset_index(drop=True)
    )

    # Device + model
    device = get_device()
    model = SentenceTransformer(model_name, device=str(device))

    # Compute embeddings on GPU
    query_embs = embed_texts(model, sampled_texts["norm"].tolist(), device=device, batch_size=batch_size)
    corpus_embs = embed_texts(model, single_core["norm"].tolist(), device=device, batch_size=batch_size)

    # Optional: same-community mask
    same_comm_mask = None
    if restrict_neighbors_same_community:
        # Build mask (Q, N)
        q_comm = torch.tensor(pd.factorize(sampled_texts["community_id"])[0], device=device)
        c_comm = torch.tensor(pd.factorize(single_core["community_id"])[0], device=device)
        # Align codes by actual values (safer): build dict -> remap
        # Faster approach: map community_id to an index for corpus
        comm2idx: Dict[str, int] = {}
        uniq_c = pd.Index(single_core["community_id"].astype(str).tolist())
        for i, cid in enumerate(uniq_c):
            comm2idx[cid] = i  # not used directly, but keep for clarity

        # Build (Q, N) mask by equality comparison
        # Convert both to strings to ensure exact match
        q_comm_vals = pd.Series(sampled_texts["community_id"].astype(str).tolist())
        c_comm_vals = pd.Series(single_core["community_id"].astype(str).tolist())
        # Create a mapping from community string -> corpus indices where it occurs
        from collections import defaultdict
        comm_to_cidx = defaultdict(list)
        for idx, val in enumerate(c_comm_vals):
            comm_to_cidx[val].append(idx)
        same_comm_mask = torch.zeros((len(q_comm_vals), len(c_comm_vals)), device=device, dtype=torch.bool)
        for qi, qv in enumerate(q_comm_vals):
            idxs = comm_to_cidx.get(qv, [])
            if idxs:
                same_comm_mask[qi, torch.tensor(idxs, device=device)] = True

    # Top-k on GPU
    _, topk_idx = topk_similar(query_embs, corpus_embs, k=neighbor_k, same_comm_mask=same_comm_mask)

    # Build outputs
    results_rows = []
    for qi in range(topk_idx.size(0)):
        q_row = sampled_texts.iloc[qi]
        q_mid = int(q_row["message_id"])
        q_text = str(q_row["content"])
        q_ctx = q_row.get("supplementary_context", "")

        # top-2 original topics for this multi-topic message
        t2 = top_two_topics_for_message(sampled_multi, q_mid)
        t1_id, t1_conf = t2[0]
        t2_id, t2_conf = t2[1]

        # Build example rows and target topic set
        example_rows: List[Tuple[int, str, str, float]] = []
        target_topic_ids = set([t1_id, t2_id])

        neighbor_indices = topk_idx[qi].tolist()
        for ci in neighbor_indices:
            crow = single_core.iloc[ci]
            mid_c = int(crow["message_id"])
            txt_c = str(crow["content"])
            tid_c = str(crow["topic_id"])
            conf_c = float(crow["confidence_score"])
            tname_c = topic_meta.get(tid_c, {}).get("name", tid_c)
            example_rows.append((mid_c, txt_c, tname_c, conf_c))
            target_topic_ids.add(tid_c)

        # Build prompt and call Gemini
        target_topic_ids = [tid for tid in target_topic_ids if tid]
        prompt = build_gemini_prompt(
            message_text=q_text,
            community_context=q_ctx,
            topic_ids_for_prompt=target_topic_ids,
            topic_meta=topic_meta,
            example_rows=example_rows
        )
        gem = call_gemini(prompt)
        pred_name = str(gem.get("predicted_topic_name", "") or "")
        pred_id = str(gem.get("predicted_topic_id", "") or "")
        pred_conf = gem.get("confidence", None)

        if pred_name.strip() == "" and pred_id:
            pred_name = topic_meta.get(pred_id, {}).get("name", pred_id)

        results_rows.append({
            "message_id": q_mid,
            "message content": q_text,
            "topic 1": topic_meta.get(t1_id, {}).get("name", t1_id),
            "confidence 1": t1_conf,
            "topic 2": topic_meta.get(t2_id, {}).get("name", t2_id),
            "confidence 2": t2_conf,
            "predicted topic": pred_name if pred_name else pred_id,
            "predicted confidence": pred_conf
        })

    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(OUT_CSV.as_posix(), index=False)
    return out_df



if __name__ == "__main__":

    messages_df = pd.read_csv("data/messages.csv", usecols=["message_id", "content"])
    df = pd.read_csv("data/message_topic_classifications.csv")
    df2 = pd.read_csv("data/community.csv", usecols=["id", "supplementary_context"])
    df3 = pd.read_csv("data/topic.csv", usecols=["id", "name", "definition"])
    out = run_pipeline_cuda_semantic(df, messages_df, df2, df3)

