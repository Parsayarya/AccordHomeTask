#!/usr/bin/env python3
"""
Main pipeline 

This script:
1. Export data from PostgreSQL (optional)
2. Run exploratory data analysis (EDA)
3. Detect ambiguous messages using PMI-based contextual polysemy detection
4. Rank messages by ambiguity score
5. Run improved classification with semantic retrieval and Gemini API
6. Calculate performance metrics (requires manual labeling)

Usage:
    python main.py --skip-export
    python main.py --dsn "postgresql://..." --export-dir data
    python main.py --skip-export --skip-eda --skip-pmi --skip-ranker
    python main.py --metrics-only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from itertools import compress
from typing import Optional

# Add utils and src to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_export(dsn: str, output_dir: str, chunk_size: int = 100_000) -> None:
    """Export data from PostgreSQL database to CSV files in output_dir."""
    from export_PostGres import build_engine, list_public_relations, export_relation
    import pandas as pd  # noqa: F401

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = build_engine(dsn)
    relations = list_public_relations(engine)

    for _, row in relations.iterrows():
        name = row["name"]
        kind = row["kind"]
        export_relation(engine, name, kind, out_dir, chunk=chunk_size)


def run_eda(data_dir: str = "data", output_dir: str = "outputs/eda") -> None:
    """Run exploratory data analysis and write figures to output_dir."""
    from EDA import run as run_eda_analysis

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_eda_analysis(data_dir=data_dir, output_dir=str(out_dir))


def run_pmi_detection(data_dir: str = "data") -> None:
    """Run PMI-based contextual polysemy detection via helper script."""
    script_path = Path(__file__).parent / "utils" / "PMI-CP-Detector.py"
    subprocess.run([sys.argv[0], str(script_path)], cwd=Path.cwd())


def run_ambiguity_ranking(data_dir: str = "data") -> None:
    """Train ranker and persist ranked ambiguity results."""
    from ranker import train_and_rank_ambiguity
    import pandas as pd

    df = pd.read_csv(Path(data_dir) / "message_topic_classifications.csv")
    messages_df = pd.read_csv(Path(data_dir) / "messages.csv", usecols=["message_id", "content"])

    scored = train_and_rank_ambiguity(
        df,
        messages_df,
        top_quantile=0.9,
        top_n_per_community=200,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "top_ambiguous_ranked.csv").write_text(
        scored.to_csv(index=False)
    )


def run_classification(data_dir: str = "data", sample_n: int = 100) -> None:
    """Run improved classification with semantic retrieval + Gemini and persist results."""
    from Classify import run_pipeline_cuda_semantic
    import pandas as pd

    messages_df = pd.read_csv(Path(data_dir) / "messages.csv", usecols=["message_id", "content"])
    df = pd.read_csv(Path(data_dir) / "message_topic_classifications.csv")
    df2 = pd.read_csv(Path(data_dir) / "community.csv", usecols=["id", "supplementary_context"])
    df3 = pd.read_csv(Path(data_dir) / "topic.csv", usecols=["id", "name", "definition"])

    out = run_pipeline_cuda_semantic(df, messages_df, df2, df3, sample_n=sample_n)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gemini_classification_candidates.csv").write_text(
        out.to_csv(index=False)
    )


def run_metrics(input_file: str = "outputs/gemini_classification_candidates.csv") -> None:
    """Calculate performance metrics on labeled data via helper script."""
    _ = Path(input_file)  # touch for clarity; metrics.py should handle validation
    script_path = Path(__file__).parent / "utils" / "metrics.py"
    subprocess.run([sys.argv[0], str(script_path)], cwd=Path.cwd())


def main():
    parser = argparse.ArgumentParser(
        description="Accord Classification Quality Improvement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --skip-export
  python main.py --dsn "postgresql://user:pass@host/db" --export-dir data
  python main.py --skip-export --skip-eda --skip-pmi --skip-ranker
  python main.py --metrics-only
        """,
    )

    # Data source options
    parser.add_argument("--dsn", type=str, help="PostgreSQL connection string")
    parser.add_argument("--export-dir", type=str, default="data", help="Directory for exported data")
    parser.add_argument("--skip-export", action="store_true", help="Skip PostgreSQL export step")

    # Pipeline control
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-pmi", action="store_true", help="Skip PMI detection step")
    parser.add_argument("--skip-ranker", action="store_true", help="Skip ambiguity ranking step")
    parser.add_argument("--skip-classification", action="store_true", help="Skip Gemini classification step")
    parser.add_argument("--metrics-only", action="store_true", help="Only calculate metrics (requires labeled data)")

    # Configuration
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing input data")
    parser.add_argument("--eda-output", type=str, default="outputs/eda", help="Directory for EDA outputs")
    parser.add_argument("--sample-n", type=int, default=100, help="Number of messages to classify")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for PostgreSQL export")

    args = parser.parse_args()

    dsn = args.dsn or os.getenv("ACCORD_PG_DSN")

    steps = {
        "export":   lambda: run_export(dsn, args.export_dir, args.chunk_size),
        "eda":      lambda: run_eda(args.data_dir, args.eda_output),
        "pmi":      lambda: run_pmi_detection(args.data_dir),
        "ranker":   lambda: run_ambiguity_ranking(args.data_dir),
        "classify": lambda: run_classification(args.data_dir, args.sample_n),
        "metrics":  lambda: run_metrics(),
    }

    order = ["export", "eda", "pmi", "ranker", "classify", "metrics"]
    mask = [
        (not args.skip_export) and (not args.metrics_only),
        (not args.skip_eda) and (not args.metrics_only),
        (not args.skip_pmi) and (not args.metrics_only),
        (not args.skip_ranker) and (not args.metrics_only),
        (not args.skip_classification) and (not args.metrics_only),
        args.metrics_only,
    ]

    for name in compress(order, mask):
        steps[name]()


if __name__ == "__main__":
    main()
