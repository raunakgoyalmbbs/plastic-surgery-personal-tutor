#!/usr/bin/env python3
"""
migrate_to_supabase.py — Bulk-load local NanoVectorDB → Supabase pgvector

Reads the local LightRAG rag_storage/ files written by NanoVectorDBStorage and
JsonKVStorage, then bulk-inserts them into the Supabase PostgreSQL tables that
PGVectorStorage / PGKVStorage use.

Strategy (fast):
  1. DROP the HNSW index on each vector table
  2. Bulk INSERT all rows via psycopg2 execute_values (batches of 500)
  3. CREATE INDEX CONCURRENTLY to rebuild HNSW in one pass
     → single-pass build is ~10x faster than incremental upsert per row

Usage:
  pip install psycopg2-binary python-dotenv numpy
  python scripts/migrate_to_supabase.py

  # Dry run (count rows, don't write):
  python scripts/migrate_to_supabase.py --dry-run

  # Re-run safely (ON CONFLICT DO UPDATE overwrites existing rows):
  python scripts/migrate_to_supabase.py
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import dotenv_values

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
ENV_FILE = REPO_ROOT / "LightRAG" / ".env"
RAG_STORAGE = REPO_ROOT / "LightRAG" / "rag_storage"

BATCH_SIZE = 500          # rows per INSERT batch
EMBEDDING_DIM = 1536      # must match EMBEDDING_DIM in .env

# NanoVectorDB namespace → Supabase table name
VECTOR_TABLES = {
    "entities":      "LIGHTRAG_VDB_ENTITY",
    "relations":     "LIGHTRAG_VDB_RELATION",
    "chunks":        "LIGHTRAG_VDB_CHUNKS",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_env() -> dict:
    cfg = dotenv_values(ENV_FILE)
    required = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER",
                "POSTGRES_PASSWORD", "POSTGRES_DATABASE"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        sys.exit(f"ERROR: Missing in {ENV_FILE}: {', '.join(missing)}")
    return cfg


def pg_connect(cfg: dict):
    ssl_root = cfg.get("POSTGRES_SSL_ROOT_CERT", "")
    sslrootcert = ssl_root if ssl_root and Path(ssl_root).exists() else None
    conn = psycopg2.connect(
        host=cfg["POSTGRES_HOST"],
        port=int(cfg.get("POSTGRES_PORT", 5432)),
        user=cfg["POSTGRES_USER"],
        password=cfg["POSTGRES_PASSWORD"].strip("'\""),
        dbname=cfg["POSTGRES_DATABASE"],
        sslmode=cfg.get("POSTGRES_SSL_MODE", "require"),
        sslrootcert=sslrootcert,
        options="-c statement_timeout=0",   # no timeout for index builds
        connect_timeout=30,
    )
    conn.autocommit = False
    return conn


def decode_matrix(vdb_json: dict) -> np.ndarray:
    """Decode base64 float32 matrix from NanoVectorDB JSON."""
    raw = base64.b64decode(vdb_json["matrix"])
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
    return arr


def load_vdb(namespace: str) -> tuple[list[dict], np.ndarray] | None:
    """Load NanoVectorDB JSON for a namespace. Returns (data_list, matrix) or None."""
    path = RAG_STORAGE / f"vdb_{namespace}.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    with open(path, encoding="utf-8") as f:
        vdb = json.load(f)
    matrix = decode_matrix(vdb)
    data = vdb["data"]
    assert len(data) == len(matrix), \
        f"Row count mismatch in {path.name}: data={len(data)}, matrix={len(matrix)}"
    return data, matrix


def table_exists(cur, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
        (table.lower(),)
    )
    return cur.fetchone() is not None


def drop_hnsw_index(cur, table: str):
    """Drop the HNSW index on content_vector to speed up bulk INSERT."""
    cur.execute(
        """
        SELECT indexname FROM pg_indexes
        WHERE tablename = %s AND indexdef ILIKE '%hnsw%'
        """,
        (table.lower(),)
    )
    rows = cur.fetchall()
    for (idx_name,) in rows:
        print(f"    Dropping HNSW index: {idx_name}")
        cur.execute(f'DROP INDEX IF EXISTS "{idx_name}"')


def rebuild_hnsw_index(conn, table: str, hnsw_m: int = 16, ef: int = 200):
    """Rebuild HNSW index on content_vector after bulk INSERT."""
    idx_name = f"idx_{table.lower()}_hnsw"
    # Must run outside a transaction (CREATE INDEX CONCURRENTLY)
    old_autocommit = conn.autocommit
    conn.autocommit = True
    cur = conn.cursor()
    print(f"    Building HNSW index {idx_name} (this takes a few minutes)...")
    t0 = time.time()
    cur.execute(
        f"""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS "{idx_name}"
        ON "{table}" USING hnsw (content_vector vector_cosine_ops)
        WITH (m = {hnsw_m}, ef_construction = {ef})
        """
    )
    cur.close()
    conn.autocommit = old_autocommit
    print(f"    Done in {time.time() - t0:.0f}s")


# ── Per-table migration ───────────────────────────────────────────────────────

def _arr_or_none(val):
    """Convert list → postgres array literal, or None."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    return None


def migrate_entities(cur, data: list[dict], matrix: np.ndarray, dry_run: bool):
    table = "LIGHTRAG_VDB_ENTITY"
    print(f"  {table}: {len(data)} rows")
    if dry_run:
        return

    drop_hnsw_index(cur, table)

    rows = []
    for item, vec in zip(data, matrix):
        rows.append((
            item["__id__"],
            item.get("workspace", ""),
            item.get("entity_name") or item.get("__id__"),
            item.get("content", ""),
            vec.tolist(),
            _arr_or_none(item.get("chunk_ids")),
            item.get("file_path"),
        ))

    sql = f"""
        INSERT INTO {table}
          (id, workspace, entity_name, content, content_vector, chunk_ids, file_path)
        VALUES %s
        ON CONFLICT (workspace, id) DO UPDATE SET
          entity_name   = EXCLUDED.entity_name,
          content       = EXCLUDED.content,
          content_vector= EXCLUDED.content_vector,
          chunk_ids     = EXCLUDED.chunk_ids,
          file_path     = EXCLUDED.file_path,
          update_time   = CURRENT_TIMESTAMP
    """
    _batch_insert(cur, sql, rows)


def migrate_relations(cur, data: list[dict], matrix: np.ndarray, dry_run: bool):
    table = "LIGHTRAG_VDB_RELATION"
    print(f"  {table}: {len(data)} rows")
    if dry_run:
        return

    drop_hnsw_index(cur, table)

    rows = []
    for item, vec in zip(data, matrix):
        rows.append((
            item["__id__"],
            item.get("workspace", ""),
            item.get("source_id", ""),
            item.get("target_id", ""),
            item.get("content", ""),
            vec.tolist(),
            _arr_or_none(item.get("chunk_ids")),
            item.get("file_path"),
        ))

    sql = f"""
        INSERT INTO {table}
          (id, workspace, source_id, target_id, content, content_vector, chunk_ids, file_path)
        VALUES %s
        ON CONFLICT (workspace, id) DO UPDATE SET
          source_id     = EXCLUDED.source_id,
          target_id     = EXCLUDED.target_id,
          content       = EXCLUDED.content,
          content_vector= EXCLUDED.content_vector,
          chunk_ids     = EXCLUDED.chunk_ids,
          file_path     = EXCLUDED.file_path,
          update_time   = CURRENT_TIMESTAMP
    """
    _batch_insert(cur, sql, rows)


def migrate_chunks(cur, data: list[dict], matrix: np.ndarray, dry_run: bool):
    table = "LIGHTRAG_VDB_CHUNKS"
    print(f"  {table}: {len(data)} rows")
    if dry_run:
        return

    drop_hnsw_index(cur, table)

    rows = []
    for item, vec in zip(data, matrix):
        rows.append((
            item["__id__"],
            item.get("workspace", ""),
            item.get("full_doc_id", ""),
            item.get("chunk_order_index") or 0,
            item.get("tokens") or 0,
            item.get("content", ""),
            vec.tolist(),
            item.get("file_path"),
        ))

    sql = f"""
        INSERT INTO {table}
          (id, workspace, full_doc_id, chunk_order_index, tokens,
           content, content_vector, file_path)
        VALUES %s
        ON CONFLICT (workspace, id) DO UPDATE SET
          full_doc_id        = EXCLUDED.full_doc_id,
          chunk_order_index  = EXCLUDED.chunk_order_index,
          tokens             = EXCLUDED.tokens,
          content            = EXCLUDED.content,
          content_vector     = EXCLUDED.content_vector,
          file_path          = EXCLUDED.file_path,
          update_time        = CURRENT_TIMESTAMP
    """
    _batch_insert(cur, sql, rows)


def _batch_insert(cur, sql: str, rows: list):
    total = len(rows)
    inserted = 0
    t0 = time.time()
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH_SIZE)
        inserted += len(batch)
        pct = inserted / total * 100
        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"    {inserted}/{total} ({pct:.0f}%) — {rate:.0f} rows/s", end="\r")
    print(f"    {total}/{total} (100%) — done in {time.time() - t0:.1f}s        ")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migrate NanoVectorDB → Supabase pgvector")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count rows and verify files exist, don't write anything")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no writes ===")

    cfg = load_env()
    print(f"Config loaded from {ENV_FILE}")
    print(f"RAG storage: {RAG_STORAGE}")
    print()

    # Load all three VDB files
    loaded = {}
    for namespace in ("entities", "relations", "chunks"):
        result = load_vdb(namespace)
        if result:
            data, matrix = result
            loaded[namespace] = (data, matrix)
            print(f"  Loaded vdb_{namespace}.json: {len(data)} rows, matrix {matrix.shape}")

    if not loaded:
        sys.exit("No VDB files found in rag_storage/. Run indexing first.")

    if args.dry_run:
        print("\nDry run complete — run without --dry-run to migrate.")
        return

    print(f"\nConnecting to Supabase ({cfg['POSTGRES_HOST']})...")
    conn = pg_connect(cfg)
    cur = conn.cursor()
    print("Connected.\n")

    hnsw_m = int(cfg.get("POSTGRES_HNSW_M", 16))
    hnsw_ef = int(cfg.get("POSTGRES_HNSW_EF", 200))

    try:
        if "entities" in loaded:
            print("── Entities ──")
            migrate_entities(cur, *loaded["entities"], dry_run=False)
            conn.commit()

        if "relations" in loaded:
            print("── Relations ──")
            migrate_relations(cur, *loaded["relations"], dry_run=False)
            conn.commit()

        if "chunks" in loaded:
            print("── Chunks ──")
            migrate_chunks(cur, *loaded["chunks"], dry_run=False)
            conn.commit()

    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        sys.exit(f"ERROR during migration: {e}")

    cur.close()
    print()

    # Rebuild HNSW indexes (outside transaction — CREATE INDEX CONCURRENTLY)
    for namespace, table in [
        ("entities",  "LIGHTRAG_VDB_ENTITY"),
        ("relations", "LIGHTRAG_VDB_RELATION"),
        ("chunks",    "LIGHTRAG_VDB_CHUNKS"),
    ]:
        if namespace in loaded:
            print(f"── Rebuilding HNSW: {table} ──")
            rebuild_hnsw_index(conn, table, hnsw_m, hnsw_ef)

    conn.close()

    total_rows = sum(len(d) for d, _ in loaded.values())
    print(f"\n✓ Migration complete — {total_rows:,} total rows migrated to Supabase.")
    print("  Switch LightRAG/.env back to PGKVStorage / PGVectorStorage when ready.")


if __name__ == "__main__":
    main()
