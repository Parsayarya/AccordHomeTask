import argparse
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_CHUNK = 100_000


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_engine(dsn: str) -> Engine:
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
        return create_engine(dsn, pool_pre_ping=True, pool_recycle=1800)


# ---------- Discovery ----------
def list_public_relations(engine: Engine) -> pd.DataFrame:
    q = """
    WITH base AS (
        SELECT table_name AS name, table_type AS kind
        FROM information_schema.tables
        WHERE table_schema = 'public'
    ),
    mat AS (
        SELECT matviewname AS name, 'MATERIALIZED VIEW' AS kind
        FROM pg_matviews
        WHERE schemaname = 'public'
    )
    SELECT * FROM base
    UNION ALL
    SELECT * FROM mat
    ORDER BY name;
    """
    return pd.read_sql(q, engine)


def list_columns(engine: Engine, table: str) -> pd.DataFrame:
    q = """
    SELECT column_name, data_type, is_nullable, ordinal_position
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=:t
    ORDER BY ordinal_position;
    """
    return pd.read_sql(text(q), engine, params={"t": table})


def safe_csv_path(out_dir: Path, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return out_dir / f"{safe}.csv"


def export_relation(engine: Engine, name: str, kind: str, out_dir: Path, chunk: int = DEFAULT_CHUNK) -> Dict[str, Any]:
    csv_path = safe_csv_path(out_dir, name)
    sql = f'SELECT * FROM "public"."{name}"'
    total_rows = 0
    first = True
    try:
        with engine.connect() as conn:
            for df in pd.read_sql(text(sql), conn, chunksize=chunk):
                total_rows += len(df)
                df.to_csv(csv_path, index=False, mode='w' if first else 'a', header=first)
                first = False
    except Exception as e:
        return {
            "name": name,
            "kind": kind,
            "status": "error",
            "error": repr(e),
            "rows": total_rows,
            "csv_path": str(csv_path),
        }

    return {
        "name": name,
        "kind": kind,
        "status": "ok",
        "rows": total_rows,
        "csv_path": str(csv_path),
    }

parser = argparse.ArgumentParser(description="Export Postgres")
parser.add_argument("--dsn", type=str, default=os.getenv("ACCORD_PG_DSN", ""))
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
args = parser.parse_args()
dsn = (args.dsn or "").strip()
out_dir = Path(args.out).expanduser().resolve()
out_dir.mkdir(parents=True, exist_ok=True)
summary_path = out_dir / "summary.txt"
schema_path = out_dir / "schema.json"
engine = build_engine(dsn)
relations = list_public_relations(engine)
schema: Dict[str, Any] = {"generated_at_utc": now_utc_iso(), "relations": []}
for _, row in relations.iterrows():
    name = row["name"]
    kind = row["kind"]
    cols_df = list_columns(engine, name)
    cols = cols_df.to_dict(orient="records")
    schema["relations"].append({
        "name": name,
        "kind": kind,
        "columns": cols,
    })
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
# Export each relation
results = []
with open(summary_path, "w", encoding="utf-8") as logf:
    logf.write(f"Export started: {now_utc_iso()}\n")
    logf.write(f"Output dir: {out_dir}\n")
    logf.write(f"Chunk size: {args.chunk}\n\n")
    for _, row in relations.iterrows():
        name = row["name"]
        kind = row["kind"]
        print(f"[{now_utc_iso()}] Exporting {kind.lower()} '{name}'")
        meta = export_relation(engine, name, kind, out_dir, chunk=args.chunk)
        results.append(meta)
        status_line = f"{name:40s} | {kind:18s} | {meta['status']:6s} | rows={meta.get('rows', 0)} | {meta.get('csv_path','')}\n"
        logf.write(status_line)
        print(status_line, end="")
    logf.write("\nExport finished: " + now_utc_iso() + "\n")
with open(out_dir / "export_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
