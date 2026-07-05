#!/usr/bin/env python3
"""
Re-embed ~77K missing relationships from the LLM extraction cache.

The LLM cache (kv_store_llm_response_cache.json) contains the original
extraction output with relationship descriptions in this format:
  relation<|#|>SOURCE<|#|>TARGET<|#|>KEYWORDS<|#|>DESCRIPTION

This script:
  1. Parses all 10K+ cache entries to extract every relationship
  2. Cross-references against existing vdb_relationships.json
  3. Embeds missing ones with OpenAI text-embedding-3-small
  4. Appends to vdb_relationships.json
  5. Adds new edges to the GraphML graph
  6. Re-migrates relationships to Supabase

Run AFTER overnight_rebuild_migrate.py has finished.

Usage:
  cd <repo-root>
  LightRAG/.venv/bin/python3 scripts/reembed_missing_relations.py 2>&1 | tee /tmp/reembed_relations.log
"""

import json, os, sys, time, base64, zlib, ssl, logging, hashlib, subprocess
from pathlib import Path
from datetime import datetime

import asyncio
import asyncpg
import numpy as np
import networkx as nx
import requests
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAG_STORAGE = ROOT / "LightRAG" / "rag_storage"

# Load secrets from LightRAG/.env (gitignored)
load_dotenv(ROOT / "LightRAG" / ".env")

PG_HOST = os.environ.get("POSTGRES_HOST", "aws-1-us-east-1.pooler.supabase.com")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "6543"))
PG_USER = os.environ.get("POSTGRES_USER", "postgres.your-project-ref")
PG_PASS = os.environ["POSTGRES_PASSWORD"]
PG_DB = os.environ.get("POSTGRES_DATABASE", "postgres")
PG_SSL_CERT = ROOT / "LightRAG" / "supabase-ca.crt"

RELATION_TABLE = "lightrag_vdb_relation_text_embedding_3_small_1536d"

OPENAI_API_KEY = os.environ["OPENAI_KEY_1"]
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

EMBED_BATCH_SIZE = 100
PG_BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reembed-rel")


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_vdb(filepath):
    """Load a NanoVectorDB JSON file, stripping the matrix."""
    with open(filepath, "r") as f:
        content = f.read()
    data_end = content.rfind('], "matrix"')
    if data_end != -1:
        content = content[: data_end + 1] + "}"
    else:
        last = content.rfind('}, {"__id__"')
        if last != -1:
            content = content[: last + 1] + "]}"
    return json.loads(content)


def decode_vector(b64_str):
    raw = zlib.decompress(base64.b64decode(b64_str))
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def encode_vector(vec_f32):
    raw = vec_f32.astype(np.float16).tobytes()
    return base64.b64encode(zlib.compress(raw)).decode("ascii")


def vec_to_pg(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"


def split_source_ids(source_id_str):
    if not source_id_str:
        return []
    return [p[:255] for p in source_id_str.split("<SEP>") if p]


def openai_embed_batch(texts, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                    "dimensions": EMBEDDING_DIM,
                },
                timeout=120,
            )
            if r.status_code == 429:
                wait = min(2 ** (attempt + 1) * 5, 60)
                log.warning(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()["data"]
            return [np.array(d["embedding"], dtype=np.float32) for d in data]
        except Exception as ex:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"  Embed error (attempt {attempt+1}): {ex}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return []


# ── Step 1: Parse LLM cache for all relationships ──────────────────────────
def parse_relations_from_cache():
    log.info("Step 1: Parsing LLM cache for relationship descriptions...")

    cache_path = RAG_STORAGE / "kv_store_llm_response_cache.json"
    with open(cache_path, "r") as f:
        cache = json.load(f)
    log.info(f"  Cache entries: {len(cache)}")

    # Parse relation lines: relation<|#|>SRC<|#|>TGT<|#|>KEYWORDS<|#|>DESCRIPTION
    all_relations = {}  # (src, tgt) → {"keywords": ..., "description": ..., "chunk_id": ...}
    parse_errors = 0

    for cache_key, cache_val in cache.items():
        text = cache_val.get("return", "")
        chunk_id = cache_val.get("chunk_id", "")

        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("relation"):
                continue

            parts = line.split("<|#|>")
            if len(parts) < 5:
                parse_errors += 1
                continue

            # parts[0] = "relation", parts[1] = src, parts[2] = tgt,
            # parts[3] = keywords, parts[4] = description
            src = parts[1].strip()
            tgt = parts[2].strip()
            keywords = parts[3].strip()
            description = parts[4].strip()

            if not src or not tgt:
                parse_errors += 1
                continue

            key = (src, tgt)
            # Keep the longest description if duplicates
            existing = all_relations.get(key)
            if not existing or len(description) > len(existing["description"]):
                all_relations[key] = {
                    "keywords": keywords,
                    "description": description,
                    "chunk_id": chunk_id,
                }

    log.info(f"  Parsed {len(all_relations)} unique relationships, {parse_errors} parse errors")
    return all_relations


# ── Step 2: Find missing relationships ──────────────────────────────────────
def find_missing(all_relations):
    log.info("Step 2: Finding missing relationships...")

    vdb = load_vdb(RAG_STORAGE / "vdb_relationships.json")
    existing_entries = vdb["data"]
    existing_pairs = {(e.get("src_id", ""), e.get("tgt_id", "")) for e in existing_entries}
    log.info(f"  Existing VDB relationships: {len(existing_entries)}")

    missing = {}
    for pair, info in all_relations.items():
        if pair not in existing_pairs:
            missing[pair] = info

    log.info(f"  Missing relationships to re-embed: {len(missing)}")
    return existing_entries, missing


# ── Step 3: Embed missing relationships ─────────────────────────────────────
def embed_missing(missing):
    log.info(f"Step 3: Embedding {len(missing)} relationships...")

    items = list(missing.items())
    new_entries = []
    epoch_now = str(int(time.time()))

    for i in range(0, len(items), EMBED_BATCH_SIZE):
        batch = items[i : i + EMBED_BATCH_SIZE]

        # Build text to embed: "keywords: description" (matches LightRAG's format)
        texts = []
        for (src, tgt), info in batch:
            content = f"{info['keywords']}: {info['description']}" if info["keywords"] else info["description"]
            texts.append(content)

        try:
            vectors = openai_embed_batch(texts)
        except Exception as ex:
            log.error(f"  Embed batch {i}-{i+len(batch)} failed: {ex}")
            continue

        for ((src, tgt), info), vec in zip(batch, vectors):
            content = f"{info['keywords']}: {info['description']}" if info["keywords"] else info["description"]
            entry_id = hashlib.md5(f"{src}<SEP>{tgt}".encode()).hexdigest()
            new_entries.append({
                "__id__": entry_id,
                "__created_at__": epoch_now,
                "__updated_at__": epoch_now,
                "src_id": src,
                "tgt_id": tgt,
                "content": content,
                "source_id": info["chunk_id"],
                "file_path": "",
                "vector": encode_vector(vec),
            })

        done = min(i + EMBED_BATCH_SIZE, len(items))
        if done % 1000 == 0 or done >= len(items):
            log.info(f"  Embedded: {done}/{len(items)}")

        time.sleep(0.5)

    log.info(f"  Generated {len(new_entries)} new entries")
    return new_entries


# ── Step 4: Save to vdb_relationships.json ──────────────────────────────────
def save_to_vdb(existing_entries, new_entries):
    log.info("Step 4: Saving to vdb_relationships.json...")

    all_entries = existing_entries + new_entries
    log.info(f"  Total entries: {len(all_entries)}")

    # Build matrix
    log.info("  Building matrix...")
    matrix = np.zeros((len(all_entries), EMBEDDING_DIM), dtype=np.float32)
    for idx, e in enumerate(all_entries):
        matrix[idx] = decode_vector(e["vector"])

    matrix_b64 = base64.b64encode(matrix.tobytes()).decode("ascii")
    vdb_out = {
        "embedding_dim": EMBEDDING_DIM,
        "data": all_entries,
        "matrix": matrix_b64,
    }

    out_path = RAG_STORAGE / "vdb_relationships.json"
    log.info(f"  Writing {out_path}...")
    with open(out_path, "w") as f:
        json.dump(vdb_out, f)

    size_mb = os.path.getsize(out_path) / 1e6
    log.info(f"  Saved ({size_mb:.1f} MB)")


# ── Step 5: Add edges to graph ──────────────────────────────────────────────
def add_edges_to_graph(new_entries):
    log.info("Step 5: Adding edges to graph...")

    graphml_path = RAG_STORAGE / "graph_chunk_entity_relation.graphml"
    G = nx.read_graphml(str(graphml_path))
    edges_before = G.number_of_edges()

    existing_nodes = set(G.nodes())
    for e in new_entries:
        src = e["src_id"]
        tgt = e["tgt_id"]
        if src not in existing_nodes:
            G.add_node(src)
            existing_nodes.add(src)
        if tgt not in existing_nodes:
            G.add_node(tgt)
            existing_nodes.add(tgt)
        G.add_edge(
            src, tgt,
            relationship_id=e["__id__"],
            description=e["content"],
            source_id=e.get("source_id", ""),
            created_at=e["__created_at__"],
        )

    nx.write_graphml(G, str(graphml_path))
    log.info(f"  Edges: {edges_before} → {G.number_of_edges()}")
    log.info(f"  Nodes: {G.number_of_nodes()}")


# ── Step 6: Re-migrate relationships to Supabase ───────────────────────────
async def migrate_relations_to_supabase():
    log.info("Step 6: Re-migrating relationships to Supabase...")

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    conn = await asyncpg.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS,
        database=PG_DB, ssl=ssl_ctx, statement_cache_size=0,
    )
    await conn.execute("SET statement_timeout = 0")

    # Reload the full vdb (with new entries)
    vdb = load_vdb(RAG_STORAGE / "vdb_relationships.json")
    entries = vdb["data"]
    log.info(f"  Total entries to migrate: {len(entries)}")

    inserted = 0
    errors = 0
    for i in range(0, len(entries), PG_BATCH_SIZE):
        batch = entries[i : i + PG_BATCH_SIZE]
        rows = []
        for e in batch:
            try:
                vec = vec_to_pg(decode_vector(e["vector"]))
                ts = datetime.utcnow()
                try:
                    ts = datetime.utcfromtimestamp(int(e.get("__created_at__", "")))
                except (ValueError, TypeError, OSError):
                    pass
                chunk_ids = split_source_ids(e.get("source_id", ""))
                rows.append((
                    e.get("__id__", ""),
                    "",
                    e.get("src_id", ""),
                    e.get("tgt_id", ""),
                    e.get("content", ""),
                    vec,
                    ts, ts,
                    chunk_ids,
                    e.get("file_path", ""),
                ))
            except Exception as ex:
                errors += 1
                if errors <= 3:
                    log.warning(f"  Decode error: {ex}")

        if rows:
            try:
                await conn.executemany(
                    f"""INSERT INTO "{RELATION_TABLE}"
                    (id, workspace, source_id, target_id, content, content_vector,
                     create_time, update_time, chunk_ids, file_path)
                    VALUES ($1,$2,$3,$4,$5,$6::vector,$7,$8,$9,$10)
                    ON CONFLICT (id, workspace) DO NOTHING""",
                    rows,
                )
                inserted += len(rows)
            except Exception as ex:
                log.error(f"  Batch error: {ex}")
                errors += len(rows)

        if (i + PG_BATCH_SIZE) % 5000 == 0 or i + PG_BATCH_SIZE >= len(entries):
            log.info(f"  Relations: {min(i + PG_BATCH_SIZE, len(entries))}/{len(entries)}")

    final = await conn.fetchval(f'SELECT COUNT(*) FROM "{RELATION_TABLE}"')
    log.info(f"  Done: {inserted} inserted, {errors} errors, {final} in table")
    await conn.close()


# ── Step 7: Push migrations and deploy Edge Function ───────────────────────
def deploy_to_supabase():
    log.info("Step 7: Pushing migrations and deploying Edge Function...")

    # Push SQL migrations (creates match_entities + match_relations RPCs)
    log.info("  Pushing DB migrations...")
    result = subprocess.run(
        ["npx", "supabase", "db", "push"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        log.info("  DB migrations pushed successfully")
    else:
        log.error(f"  DB push failed (exit {result.returncode}): {result.stderr[:500]}")
        # Try to continue anyway — migrations may already be applied

    # Deploy updated rag-query Edge Function
    log.info("  Deploying rag-query Edge Function...")
    result = subprocess.run(
        ["npx", "supabase", "functions", "deploy", "rag-query"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        log.info("  rag-query Edge Function deployed successfully")
    else:
        log.error(f"  Deploy failed (exit {result.returncode}): {result.stderr[:500]}")

    log.info("Step 7 complete.")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Re-embed missing relationships from LLM cache")
    log.info(f"Time: {datetime.now()}")
    log.info("=" * 60)

    # Step 1: Parse cache
    all_relations = parse_relations_from_cache()

    # Step 2: Find missing
    existing_entries, missing = find_missing(all_relations)

    if not missing:
        log.info("No missing relationships — skipping to deployment.")
    else:
        # Step 3: Embed
        new_entries = embed_missing(missing)

        if new_entries:
            # Step 4: Save to VDB
            save_to_vdb(existing_entries, new_entries)

            # Step 5: Add to graph
            add_edges_to_graph(new_entries)

            # Step 6: Migrate to Supabase
            asyncio.run(migrate_relations_to_supabase())
        else:
            log.error("No entries generated — check errors above")

    # Step 7: Push migrations + deploy Edge Function (always run)
    deploy_to_supabase()

    log.info("=" * 60)
    log.info("ALL DONE!")
    log.info(f"Time: {datetime.now()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
