"""
Exploratory Data Analysis for Accord Classification Quality

figures to outputs/eda/:
- topic_distribution.png
- top_topic_pairs.png
- ambiguous_topics.png
- topic_usage_statistics.png
- confidence_score_distribution.png
"""
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def _safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if s.dtype.kind in {"i", "u", "f"}:
        return s.astype(bool)
    return s.fillna(False).astype(bool)



def _plot_topic_count_distribution(topic_counts: pd.Series, out_path: Path) -> None:
    if topic_counts.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    vc = topic_counts.value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, alpha=0.8)
    axes[0].set_xlabel("Number of topics per message")
    axes[0].set_ylabel("Messages")
    axes[0].set_title("Distribution of topics per message")
    for x, y in zip(vc.index, vc.values):
        axes[0].text(x, y, f"{int(y):,}", ha="center", va="bottom", fontsize=8)

    cumsum = vc.cumsum()
    pct = cumsum / cumsum.iloc[-1] * 100
    axes[1].plot(pct.index, pct.values, marker="o", linewidth=2)
    axes[1].axhline(50, linestyle="--", linewidth=1)
    axes[1].axhline(90, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of topics per message")
    axes[1].set_ylabel("Cumulative %")
    axes[1].set_title("Cumulative distribution")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_top_pairs(pairs_df: pd.DataFrame, out_path: Path) -> None:
    if pairs_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 10))
    topn = pairs_df.head(20)
    labels = [f"{r['Topic_1']}\n+\n{r['Topic_2']}" for _, r in topn.iterrows()]
    ax.barh(range(len(topn)), topn["Count"].values, alpha=0.85)
    ax.set_yticks(range(len(topn)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Co-occurrences")
    ax.set_title("Top 20 topic pairs")
    ax.invert_yaxis()
    for i, v in enumerate(topn["Count"].values):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ambiguous_topics(counts: pd.Series, out_path: Path) -> None:
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    topn = counts.head(20)
    ax.barh(range(len(topn)), topn.values, alpha=0.85)
    ax.set_yticks(range(len(topn)))
    ax.set_yticklabels(topn.index)
    ax.set_xlabel("Frequency in ambiguous multi-topic messages")
    ax.set_title("Topics most common in ambiguous classifications")
    ax.invert_yaxis()
    for i, v in enumerate(topn.values):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_topic_usage(topic_usage_df: pd.DataFrame, out_path: Path) -> None:
    if topic_usage_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    topn = topic_usage_df.head(20)
    color_flag = _safe_bool_series(topn.get("is_global", pd.Series(False, index=topn.index)))
    colors = ["#3b82f6" if g else "#f97316" for g in color_flag]

    axes[0].barh(range(len(topn)), topn["message_count"].values, color=colors, alpha=0.9)
    axes[0].set_yticks(range(len(topn)))
    axes[0].set_yticklabels(topn["name"].astype(str).tolist(), fontsize=9)
    axes[0].set_xlabel("Messages")
    axes[0].set_title("Top 20 topics (blue=Global, orange=Custom)")
    axes[0].invert_yaxis()
    for i, v in enumerate(topn["message_count"].values):
        axes[0].text(v, i, f" {int(v):,}", va="center", fontsize=8)

    flag = _safe_bool_series(topic_usage_df.get("is_global", pd.Series(False, index=topic_usage_df.index)))
    grouped = topic_usage_df.assign(_flag=flag).groupby("_flag")["message_count"].sum()
    custom = int(grouped.get(False, 0))
    global_ = int(grouped.get(True, 0))
    axes[1].pie([custom, global_], labels=["Custom Topics", "Global Topics"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Global vs Custom usage")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_confidence_score_distribution(message_topics_df: pd.DataFrame, out_path: Path) -> None:
    if "confidence_score" not in message_topics_df.columns:
        return

    confidence_scores = message_topics_df["confidence_score"].dropna()
    if len(confidence_scores) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram
    axes[0, 0].hist(confidence_scores, bins=50, alpha=0.7, color="#3b82f6", edgecolor="black")
    axes[0, 0].set_xlabel("Confidence Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Confidence Scores")
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    axes[0, 1].boxplot(confidence_scores)
    axes[0, 1].set_ylabel("Confidence Score")
    axes[0, 1].set_title("Box Plot of Confidence Scores")
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_scores = np.sort(confidence_scores)
    cumulative_pct = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    axes[1, 0].plot(sorted_scores, cumulative_pct, linewidth=2, color="#f97316")
    axes[1, 0].set_xlabel("Confidence Score")
    axes[1, 0].set_ylabel("Cumulative Percentage")
    axes[1, 0].set_title("Cumulative Distribution of Confidence Scores")
    axes[1, 0].grid(True, alpha=0.3)

    # Summary text (on the plot)
    stats_text = (
        f"Summary Statistics:\n"
        f"Mean: {confidence_scores.mean():.3f}\n"
        f"Median: {confidence_scores.median():.3f}\n"
        f"Std Dev: {confidence_scores.std():.3f}\n"
        f"Min: {confidence_scores.min():.3f}\n"
        f"Max: {confidence_scores.max():.3f}\n"
        f"25th percentile: {confidence_scores.quantile(0.25):.3f}\n"
        f"75th percentile: {confidence_scores.quantile(0.75):.3f}\n"
        f"Total classifications: {len(confidence_scores):,}"
    )
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Summary Statistics")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(data_dir: str = "data", output_dir: str = "outputs/eda") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Required inputs
    messages_df = pd.read_csv(Path(data_dir) / "messages.csv")
    message_topics_df = pd.read_csv(Path(data_dir) / "message_topic_classifications.csv")

    # Optional topics metadata (for nicer labels/usage split)
    topics_path = Path(data_dir) / "topic.csv"
    topics_df = pd.read_csv(topics_path) if topics_path.exists() else pd.DataFrame()

    # 1) Topics-per-message distribution
    topic_counts = message_topics_df.groupby("message_id").size()
    _plot_topic_count_distribution(topic_counts, out / "topic_distribution.png")

    # 2) Top co-occurring topic pairs (requires topic names)
    if not topics_df.empty and {"id", "name"}.issubset(topics_df.columns):
        multi_topic_ids = topic_counts[topic_counts > 1].index
        mt = message_topics_df[message_topics_df["message_id"].isin(multi_topic_ids)].copy()
        name_map = dict(zip(topics_df["id"], topics_df["name"]))
        mt["topic_name"] = mt["topic_id"].map(name_map)

        pairs = []
        for mid, group in mt.groupby("message_id"):
            ts = group["topic_name"].dropna().astype(str)
            ts = [t for t in ts if t.strip()]
            if len(ts) >= 2:
                for i in range(len(ts)):
                    for j in range(i + 1, len(ts)):
                        pairs.append(tuple(sorted([ts[i], ts[j]])))

        pair_counts = Counter(pairs)
        if pair_counts:
            pairs_df = (
                pd.DataFrame(pair_counts.items(), columns=["Topic_Pair", "Count"])
                .assign(Topic_1=lambda d: d["Topic_Pair"].str[0],
                        Topic_2=lambda d: d["Topic_Pair"].str[1])[
                    ["Topic_1", "Topic_2", "Count"]
                ].sort_values("Count", ascending=False)
            )
            _plot_top_pairs(pairs_df, out / "top_topic_pairs.png")

    # 3) Ambiguous keyword messages (plot top topics among those)
    if not messages_df.empty and "content" in messages_df.columns and not topics_df.empty:
        ambiguous_keywords = [
            "crash", "crashed", "crashes", "crashing",
            "lag", "lagging", "lags",
            "drop", "dropped", "drops", "dropping",
            "freeze", "freezing", "freezes", "frozen",
            "stuck", "hang", "hanging", "hangs",
            "bug", "bugs", "buggy",
            "glitch", "glitches", "glitchy",
            "issue", "issues", "problem", "problems",
        ]

        messages_df["content_lower"] = messages_df["content"].fillna("").str.lower()
        mask = pd.Series(False, index=messages_df.index)
        for kw in ambiguous_keywords:
            mask |= messages_df["content_lower"].str.contains(kw, regex=False)

        ambiguous_msg_ids = set(messages_df.loc[mask, "message_id"].tolist())
        multi_topic_ids = set(topic_counts[topic_counts > 1].index.tolist())
        ambiguous_multi_topic = list(ambiguous_msg_ids & multi_topic_ids)

        if ambiguous_multi_topic:
            name_map = dict(zip(topics_df["id"], topics_df["name"]))
            amb = message_topics_df[message_topics_df["message_id"].isin(ambiguous_multi_topic)].copy()
            amb["topic_name"] = amb["topic_id"].map(name_map)
            counts = amb["topic_name"].value_counts()
            _plot_ambiguous_topics(counts, out / "ambiguous_topics.png")

    # 4) Topic usage split and top topics
    if not message_topics_df.empty:
        usage = message_topics_df["topic_id"].value_counts()
        usage_df = pd.DataFrame({"topic_id": usage.index, "message_count": usage.values})

        if not topics_df.empty:
            extra = topics_df[["id", "name"]].copy()
            if "is_global" in topics_df.columns:
                extra["is_global"] = topics_df["is_global"]
            usage_df = usage_df.merge(extra, left_on="topic_id", right_on="id", how="left")
            usage_df = usage_df.sort_values("message_count", ascending=False)

            _plot_topic_usage(usage_df, out / "topic_usage_statistics.png")

        # 5) Confidence score distribution
        _plot_confidence_score_distribution(message_topics_df, out / "confidence_score_distribution.png")


if __name__ == "__main__":
    run()
