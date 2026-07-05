#!/usr/bin/env python3
"""
Overnight watchdog: rebuild local graph edges, re-embed missing data, migrate to Supabase.

Steps:
  1. Wait for any in-progress indexing to finish
  2. Rebuild graph edges from vdb_relationships.json
  3. Restart LightRAG server to validate repaired graph
  4. Re-embed ~21K missing entities (lost vectors in disk-full crash)
  5. Re-embed ~77K missing relationships (lost vectors in disk-full crash)
  6. Migrate local NanoVectorDB data → Supabase Postgres
  7. Update match_chunks() to point to the correct table + verify

Usage:
  cd <repo-root>
  LightRAG/.venv/bin/python3 scripts/overnight_rebuild_migrate.py 2>&1 | tee /tmp/overnight_rebuild.log
"""

import json, os, sys, time, subprocess, base64, zlib, ssl, logging, hashlib
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
LIGHTRAG_URL = "http://localhost:9621"

# Load secrets from LightRAG/.env (gitignored)
load_dotenv(ROOT / "LightRAG" / ".env")

PG_HOST = os.environ.get("POSTGRES_HOST", "aws-1-us-east-1.pooler.supabase.com")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "6543"))
PG_USER = os.environ.get("POSTGRES_USER", "postgres.your-project-ref")
PG_PASS = os.environ["POSTGRES_PASSWORD"]
PG_DB = os.environ.get("POSTGRES_DATABASE", "postgres")
PG_SSL_CERT = ROOT / "LightRAG" / "supabase-ca.crt"

# Target tables — must match the embedding model used during indexing
CHUNK_TABLE = "lightrag_vdb_chunks_text_embedding_3_small_1536d"
ENTITY_TABLE = "lightrag_vdb_entity_text_embedding_3_small_1536d"
RELATION_TABLE = "lightrag_vdb_relation_text_embedding_3_small_1536d"

# OpenAI embedding config (must match what LightRAG used during indexing)
OPENAI_API_KEY = os.environ["OPENAI_KEY_1"]
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

BATCH_SIZE = 50  # Small batches to avoid pooler timeouts
EMBED_BATCH_SIZE = 100  # OpenAI allows up to 2048, but 100 is safe for rate limits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("overnight")


# ── Helpers ─────────────────────────────────────────────────────────────────
def lightrag_status():
    try:
        r = requests.get(f"{LIGHTRAG_URL}/documents/pipeline_status", timeout=10)
        return r.json()
    except Exception:
        return None


def lightrag_docs():
    try:
        r = requests.get(f"{LIGHTRAG_URL}/documents", timeout=10)
        return r.json()
    except Exception:
        return None


def load_vdb(filepath):
    """Load a NanoVectorDB JSON file, handling truncated matrix."""
    with open(filepath, "r") as f:
        content = f.read()

    # Strip matrix (large base64 blob) — we only need the data array
    data_end = content.rfind('], "matrix"')
    if data_end != -1:
        content = content[: data_end + 1] + "}"
    else:
        # File may be truncated mid-entry — find last complete entry
        last = content.rfind('}, {"__id__"')
        if last != -1:
            content = content[: last + 1] + "]}"

    return json.loads(content)


def decode_vector(b64_str):
    """Decode a zlib-compressed float16 vector to float32 numpy array."""
    raw = zlib.decompress(base64.b64decode(b64_str))
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def encode_vector(vec_f32):
    """Encode a float32 numpy array to zlib-compressed float16 base64 string."""
    raw = vec_f32.astype(np.float16).tobytes()
    return base64.b64encode(zlib.compress(raw)).decode("ascii")


def vec_to_pg(vec):
    """Convert numpy float32 array to Postgres vector literal."""
    return "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"


def ts_from_epoch(epoch_str):
    """Convert epoch seconds string to datetime for Postgres."""
    try:
        return datetime.utcfromtimestamp(int(epoch_str))
    except (ValueError, TypeError, OSError):
        return datetime.utcnow()


def split_source_ids(source_id_str):
    """Split a <SEP>-joined source_id string into individual chunk IDs.
    Each element must fit in varchar(255)."""
    if not source_id_str:
        return []
    parts = source_id_str.split("<SEP>")
    # Truncate any individual ID that still exceeds 255 (shouldn't happen)
    return [p[:255] for p in parts if p]


def openai_embed_batch(texts, retries=3):
    """Call OpenAI text-embedding-3-small for a batch of texts. Returns list of 1536-dim float32 arrays."""
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
                log.warning(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()["data"]
            return [np.array(d["embedding"], dtype=np.float32) for d in data]
        except Exception as ex:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"    Embed error (attempt {attempt+1}): {ex}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return []


# ── Step 1: Wait for indexing ───────────────────────────────────────────────
def wait_for_indexing():
    log.info("Step 1: Waiting for indexing to complete...")
    while True:
        status = lightrag_status()
        if status is None:
            log.warning("  LightRAG not reachable, waiting 30s...")
            time.sleep(30)
            continue
        if not status.get("busy", True):
            log.info(f"  Pipeline idle: {status.get('latest_message', '')}")
            break
        log.info(f"  Still busy: {status.get('latest_message', '')[:80]}")
        time.sleep(30)

    docs = lightrag_docs()
    if docs:
        statuses = docs.get("statuses", {})
        for s, d in statuses.items():
            log.info(f"  {s}: {len(d)} docs")
    log.info("Step 1 complete: indexing done.")


# ── Step 2: Rebuild graph edges ─────────────────────────────────────────────
def rebuild_graph_edges():
    log.info("Step 2: Rebuilding graph edges from vdb_relationships...")

    graphml_path = RAG_STORAGE / "graph_chunk_entity_relation.graphml"

    # Load current graph
    G = nx.read_graphml(str(graphml_path))
    log.info(f"  Current graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load VDB relationships (source of truth for edges)
    vdb_rel = load_vdb(RAG_STORAGE / "vdb_relationships.json")
    rel_entries = vdb_rel["data"]
    log.info(f"  VDB relationship entries: {len(rel_entries)}")

    # Load VDB entities to ensure all nodes exist
    vdb_ent = load_vdb(RAG_STORAGE / "vdb_entities.json")
    ent_entries = vdb_ent["data"]
    log.info(f"  VDB entity entries: {len(ent_entries)}")

    # Add missing entity nodes
    existing_nodes = set(G.nodes())
    nodes_added = 0
    for e in ent_entries:
        name = e.get("entity_name", "")
        if name and name not in existing_nodes:
            G.add_node(
                name,
                entity_id=e.get("__id__", ""),
                entity_type=e.get("entity_type", ""),
                description=e.get("content", ""),
                source_id=e.get("source_id", ""),
                file_path=e.get("file_path", ""),
                created_at=str(e.get("__created_at__", "")),
            )
            existing_nodes.add(name)
            nodes_added += 1

    log.info(f"  Added {nodes_added} missing nodes")

    # Add edges from VDB relationships
    edges_added = 0
    edges_skipped = 0
    for r in rel_entries:
        src = r.get("src_id", "")
        tgt = r.get("tgt_id", "")
        if not src or not tgt:
            edges_skipped += 1
            continue

        # Add source/target nodes if missing
        if src not in existing_nodes:
            G.add_node(src)
            existing_nodes.add(src)
        if tgt not in existing_nodes:
            G.add_node(tgt)
            existing_nodes.add(tgt)

        # Add edge (NetworkX overwrites if edge already exists for same src→tgt)
        G.add_edge(
            src,
            tgt,
            relationship_id=r.get("__id__", ""),
            description=r.get("content", ""),
            source_id=r.get("source_id", ""),
            file_path=r.get("file_path", ""),
            created_at=str(r.get("__created_at__", "")),
        )
        edges_added += 1

    log.info(f"  Added {edges_added} edges, skipped {edges_skipped}")
    log.info(f"  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save
    nx.write_graphml(G, str(graphml_path))
    size_mb = os.path.getsize(graphml_path) / 1e6
    log.info(f"  Saved to {graphml_path} ({size_mb:.1f} MB)")
    log.info("Step 2 complete: graph edges rebuilt.")


# ── Step 3: Restart LightRAG to validate ────────────────────────────────────
def restart_lightrag():
    log.info("Step 3: Restarting LightRAG server...")

    # Kill existing
    subprocess.run(["pkill", "-f", "lightrag-server"], capture_output=True)
    time.sleep(3)

    # Start
    venv_bin = ROOT / "LightRAG" / ".venv" / "bin" / "lightrag-server"
    with open("/tmp/lightrag.log", "a") as logf:
        subprocess.Popen(
            [str(venv_bin)],
            cwd=str(ROOT / "LightRAG"),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    # Wait for it to come up (loading 90K+ entity vectors takes time)
    for i in range(90):  # 7.5 minutes max
        time.sleep(5)
        status = lightrag_status()
        if status is not None:
            log.info("  Server is up!")
            return True
        if i % 6 == 5:
            log.info(f"  Still loading... ({(i+1)*5}s)")

    log.error("  Server failed to start in 7.5 minutes!")
    return False


# ── Step 4: Re-embed missing entities ───────────────────────────────────────
def reembed_missing_entities():
    log.info("Step 4: Re-embedding missing entities...")

    # Load complete entity set from KV store
    kv_path = RAG_STORAGE / "kv_store_entity_chunks.json"
    with open(kv_path, "r") as f:
        kv_entities = json.load(f)
    all_entity_names = set(kv_entities.keys())
    log.info(f"  Total entities in KV store: {len(all_entity_names)}")

    # Load existing VDB entities (those that have vectors)
    vdb = load_vdb(RAG_STORAGE / "vdb_entities.json")
    existing_entries = vdb["data"]
    existing_names = {e.get("entity_name", "") for e in existing_entries}
    log.info(f"  Existing VDB entities: {len(existing_entries)}")

    # Find missing entities
    missing_names = all_entity_names - existing_names
    log.info(f"  Missing entities to re-embed: {len(missing_names)}")

    if not missing_names:
        log.info("Step 4 complete: no missing entities.")
        return

    # Build content for each missing entity from kv_store_entity_chunks
    # Each entry in kv_store_entity_chunks maps entity_name → list of chunk data
    # The "content" for the entity vector is typically: "entity_name: description"
    # We reconstruct it the same way LightRAG does
    missing_items = []
    for name in missing_names:
        chunks = kv_entities.get(name, [])
        # LightRAG stores entity content as the entity description
        # The KV entry has chunk_order_index, full_doc_id, tokens, content, etc.
        # For the VDB, content is typically just the entity name + description
        # We'll use entity name as content (same as what LightRAG embeds)
        content = name
        source_ids = []
        file_paths = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                if chunk.get("source_id"):
                    source_ids.append(chunk["source_id"])
                if chunk.get("file_path"):
                    file_paths.append(chunk["file_path"])

        missing_items.append({
            "entity_name": name,
            "content": content,
            "source_ids": source_ids,
            "file_path": file_paths[0] if file_paths else "",
        })

    # Also try to get richer descriptions from kv_store_full_entities
    full_ent_path = RAG_STORAGE / "kv_store_full_entities.json"
    if full_ent_path.exists():
        with open(full_ent_path, "r") as f:
            full_entities = json.load(f)
        # kv_store_full_entities maps chunk_key → {entity_name: description, ...}
        # Flatten to entity_name → description
        entity_descriptions = {}
        for chunk_key, entities_dict in full_entities.items():
            if isinstance(entities_dict, dict):
                for ename, desc in entities_dict.items():
                    if ename in missing_names and isinstance(desc, str) and len(desc) > len(entity_descriptions.get(ename, "")):
                        entity_descriptions[ename] = desc

        enriched = 0
        for item in missing_items:
            desc = entity_descriptions.get(item["entity_name"])
            if desc:
                item["content"] = desc
                enriched += 1
        log.info(f"  Enriched {enriched}/{len(missing_items)} entities with full descriptions")

    # Embed in batches
    log.info(f"  Embedding {len(missing_items)} entities in batches of {EMBED_BATCH_SIZE}...")
    new_entries = []
    epoch_now = str(int(time.time()))

    for i in range(0, len(missing_items), EMBED_BATCH_SIZE):
        batch = missing_items[i : i + EMBED_BATCH_SIZE]
        texts = [item["content"] for item in batch]

        try:
            vectors = openai_embed_batch(texts)
        except Exception as ex:
            log.error(f"    Embed batch {i}-{i+len(batch)} failed: {ex}")
            continue

        for item, vec in zip(batch, vectors):
            entry_id = hashlib.md5(item["entity_name"].encode()).hexdigest()
            source_id = "<SEP>".join(item["source_ids"]) if item["source_ids"] else ""
            new_entries.append({
                "__id__": entry_id,
                "__created_at__": epoch_now,
                "__updated_at__": epoch_now,
                "entity_name": item["entity_name"],
                "content": item["content"],
                "source_id": source_id,
                "file_path": item["file_path"],
                "vector": encode_vector(vec),
            })

        if (i + EMBED_BATCH_SIZE) % 1000 == 0 or i + EMBED_BATCH_SIZE >= len(missing_items):
            log.info(f"    Embedded: {min(i + EMBED_BATCH_SIZE, len(missing_items))}/{len(missing_items)}")

        # Small delay to avoid rate limits
        time.sleep(0.5)

    log.info(f"  Generated {len(new_entries)} new entity entries")

    # Append to vdb_entities.json
    if new_entries:
        log.info("  Appending to vdb_entities.json...")
        # Re-read the full file to append properly
        all_entries = existing_entries + new_entries
        log.info(f"  Total entries after merge: {len(all_entries)}")

        # Build matrix from all vectors
        log.info("  Building matrix...")
        matrix = np.zeros((len(all_entries), EMBEDDING_DIM), dtype=np.float32)
        for idx, e in enumerate(all_entries):
            matrix[idx] = decode_vector(e["vector"])

        # Save in NanoVectorDB format
        matrix_b64 = base64.b64encode(matrix.tobytes()).decode("ascii")
        vdb_out = {
            "embedding_dim": EMBEDDING_DIM,
            "data": all_entries,
            "matrix": matrix_b64,
        }

        out_path = RAG_STORAGE / "vdb_entities.json"
        log.info(f"  Writing {out_path}...")
        with open(out_path, "w") as f:
            json.dump(vdb_out, f)

        size_mb = os.path.getsize(out_path) / 1e6
        log.info(f"  Saved ({size_mb:.1f} MB, {len(all_entries)} entries)")

    log.info(f"Step 4 complete: {len(new_entries)} entities re-embedded.")


# ── Step 5: Re-embed missing relationships ──────────────────────────────────
def reembed_missing_relationships():
    log.info("Step 5: Re-embedding missing relationships...")

    # Load complete relation set from KV store
    kv_rel_path = RAG_STORAGE / "kv_store_full_relations.json"
    if not kv_rel_path.exists():
        log.warning("  kv_store_full_relations.json not found, skipping.")
        return

    with open(kv_rel_path, "r") as f:
        kv_relations = json.load(f)

    # kv_store_full_relations maps chunk_key → { "src<SEP>tgt": description, ... }
    # Flatten to set of (src, tgt) pairs with descriptions
    all_relations = {}  # (src, tgt) → {"content": desc, "chunk_keys": [...]}
    for chunk_key, rels_dict in kv_relations.items():
        if not isinstance(rels_dict, dict):
            continue
        for pair_key, desc in rels_dict.items():
            parts = pair_key.split("<SEP>")
            if len(parts) != 2:
                continue
            src, tgt = parts
            key = (src, tgt)
            if key not in all_relations:
                all_relations[key] = {"content": "", "chunk_keys": []}
            if isinstance(desc, str) and len(desc) > len(all_relations[key]["content"]):
                all_relations[key]["content"] = desc
            all_relations[key]["chunk_keys"].append(chunk_key)

    log.info(f"  Total relations in KV store: {len(all_relations)}")

    # Load existing VDB relationships
    vdb = load_vdb(RAG_STORAGE / "vdb_relationships.json")
    existing_entries = vdb["data"]
    existing_pairs = {(e.get("src_id", ""), e.get("tgt_id", "")) for e in existing_entries}
    log.info(f"  Existing VDB relationships: {len(existing_entries)}")

    # Find missing
    missing_pairs = set(all_relations.keys()) - existing_pairs
    log.info(f"  Missing relationships to re-embed: {len(missing_pairs)}")

    if not missing_pairs:
        log.info("Step 5 complete: no missing relationships.")
        return

    # Build items to embed
    missing_items = []
    for src, tgt in missing_pairs:
        info = all_relations[(src, tgt)]
        # LightRAG typically embeds the relationship description
        content = info["content"] if info["content"] else f"{src} -> {tgt}"
        missing_items.append({
            "src_id": src,
            "tgt_id": tgt,
            "content": content,
            "source_ids": info["chunk_keys"],
        })

    # Embed in batches
    log.info(f"  Embedding {len(missing_items)} relationships in batches of {EMBED_BATCH_SIZE}...")
    new_entries = []
    epoch_now = str(int(time.time()))

    for i in range(0, len(missing_items), EMBED_BATCH_SIZE):
        batch = missing_items[i : i + EMBED_BATCH_SIZE]
        texts = [item["content"] for item in batch]

        try:
            vectors = openai_embed_batch(texts)
        except Exception as ex:
            log.error(f"    Embed batch {i}-{i+len(batch)} failed: {ex}")
            continue

        for item, vec in zip(batch, vectors):
            entry_id = hashlib.md5(f"{item['src_id']}<SEP>{item['tgt_id']}".encode()).hexdigest()
            source_id = "<SEP>".join(item["source_ids"]) if item["source_ids"] else ""
            new_entries.append({
                "__id__": entry_id,
                "__created_at__": epoch_now,
                "__updated_at__": epoch_now,
                "src_id": item["src_id"],
                "tgt_id": item["tgt_id"],
                "content": item["content"],
                "source_id": source_id,
                "file_path": "",
                "vector": encode_vector(vec),
            })

        if (i + EMBED_BATCH_SIZE) % 1000 == 0 or i + EMBED_BATCH_SIZE >= len(missing_items):
            log.info(f"    Embedded: {min(i + EMBED_BATCH_SIZE, len(missing_items))}/{len(missing_items)}")

        # Small delay to avoid rate limits
        time.sleep(0.5)

    log.info(f"  Generated {len(new_entries)} new relationship entries")

    # Append to vdb_relationships.json
    if new_entries:
        log.info("  Appending to vdb_relationships.json...")
        all_entries = existing_entries + new_entries
        log.info(f"  Total entries after merge: {len(all_entries)}")

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
        log.info(f"  Saved ({size_mb:.1f} MB, {len(all_entries)} entries)")

    # Also add new edges to the graph
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
    log.info(f"  Graph edges: {edges_before} → {G.number_of_edges()}")
    log.info(f"  Graph nodes: {G.number_of_nodes()}")

    log.info(f"Step 5 complete: {len(new_entries)} relationships re-embedded.")


# ── Supabase connection helper ──────────────────────────────────────────────
async def pg_connect(retries=5):
    """Connect to Supabase Postgres via pooler with no statement timeout."""
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    for attempt in range(retries):
        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS,
                    database=PG_DB, ssl=ssl_ctx, statement_cache_size=0,
                ),
                timeout=30,
            )
            await conn.execute("SET statement_timeout = 0")
            return conn
        except Exception as ex:
            if attempt < retries - 1:
                wait = 3 * (attempt + 1)
                log.warning(f"    pg_connect attempt {attempt+1} failed: {ex}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


# ── Step 6: Migrate to Supabase ─────────────────────────────────────────────
async def migrate_to_supabase():
    log.info("Step 6: Migrating local VDB data to Supabase Postgres...")
    conn = await pg_connect()

    # ── Chunks ──────────────────────────────────────────────────────────────
    log.info(f"  --- Migrating chunks → {CHUNK_TABLE} ---")
    vdb = load_vdb(RAG_STORAGE / "vdb_chunks.json")
    entries = vdb["data"]
    log.info(f"    Local entries: {len(entries)}")

    inserted = 0
    errors = 0
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        rows = []
        for e in batch:
            try:
                vec = vec_to_pg(decode_vector(e["vector"]))
                ts = ts_from_epoch(e.get("__created_at__", ""))
                rows.append((
                    e.get("__id__", ""),       # id
                    "",                         # workspace
                    e.get("full_doc_id", ""),   # full_doc_id
                    0,                          # chunk_order_index
                    0,                          # tokens
                    e.get("content", ""),       # content
                    vec,                        # content_vector
                    e.get("file_path", ""),     # file_path
                    ts,                         # create_time
                    ts,                         # update_time
                ))
            except Exception as ex:
                errors += 1
                if errors <= 3:
                    log.warning(f"    Decode error: {ex}")

        if rows:
            try:
                await conn.executemany(
                    f"""INSERT INTO "{CHUNK_TABLE}"
                    (id, workspace, full_doc_id, chunk_order_index, tokens,
                     content, content_vector, file_path, create_time, update_time)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::vector,$8,$9,$10)
                    ON CONFLICT (id, workspace) DO NOTHING""",
                    rows,
                )
                inserted += len(rows)
            except Exception as ex:
                log.error(f"    Batch error: {ex}")
                errors += len(rows)

        if (i + BATCH_SIZE) % 1000 == 0:
            log.info(f"    Chunks: {i + BATCH_SIZE}/{len(entries)}")

    log.info(f"    Chunks done: {inserted} attempted, {errors} errors")

    # ── Entities ────────────────────────────────────────────────────────────
    log.info(f"  --- Migrating entities → {ENTITY_TABLE} ---")
    vdb = load_vdb(RAG_STORAGE / "vdb_entities.json")
    entries = vdb["data"]
    log.info(f"    Local entries: {len(entries)}")

    inserted = 0
    errors = 0
    batch_num = 0
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        rows = []
        for e in batch:
            try:
                vec = vec_to_pg(decode_vector(e["vector"]))
                ts = ts_from_epoch(e.get("__created_at__", ""))
                chunk_ids = split_source_ids(e.get("source_id", ""))
                rows.append((
                    e.get("__id__", ""),        # id
                    "",                          # workspace
                    e.get("entity_name", ""),    # entity_name
                    e.get("content", ""),        # content
                    vec,                         # content_vector
                    ts,                          # create_time
                    ts,                          # update_time
                    chunk_ids,                   # chunk_ids (varchar(255)[])
                    e.get("file_path", ""),      # file_path
                ))
            except Exception as ex:
                errors += 1
                if errors <= 3:
                    log.warning(f"    Decode error: {ex}")

        if rows:
            # Reconnect every 10 batches to avoid pooler dropping idle connections
            if batch_num % 10 == 0:
                try:
                    await conn.close()
                except Exception:
                    pass
                conn = await pg_connect()
            try:
                await asyncio.wait_for(
                    conn.executemany(
                        f"""INSERT INTO "{ENTITY_TABLE}"
                        (id, workspace, entity_name, content, content_vector,
                         create_time, update_time, chunk_ids, file_path)
                        VALUES ($1,$2,$3,$4,$5::vector,$6,$7,$8,$9)
                        ON CONFLICT (id, workspace) DO NOTHING""",
                        rows,
                    ),
                    timeout=120,
                )
                inserted += len(rows)
            except (asyncio.TimeoutError, Exception) as ex:
                log.warning(f"    Batch failed at {i + BATCH_SIZE}/{len(entries)}: {type(ex).__name__}, falling back to row-by-row...")
                try:
                    await conn.close()
                except Exception:
                    pass
                conn = await pg_connect()
                # Row-by-row fallback
                sql = f"""INSERT INTO "{ENTITY_TABLE}"
                    (id, workspace, entity_name, content, content_vector,
                     create_time, update_time, chunk_ids, file_path)
                    VALUES ($1,$2,$3,$4,$5::vector,$6,$7,$8,$9)
                    ON CONFLICT (id, workspace) DO NOTHING"""
                for row in rows:
                    try:
                        await asyncio.wait_for(conn.execute(sql, *row), timeout=30)
                        inserted += 1
                    except Exception as row_ex:
                        errors += 1
                        if errors <= 10:
                            log.warning(f"    Row error [{type(row_ex).__name__}]: {repr(row_ex)}")
                        try:
                            await conn.close()
                        except Exception:
                            pass
                        conn = await pg_connect()
            batch_num += 1

        if (i + BATCH_SIZE) % 1000 == 0:
            log.info(f"    Entities: {i + BATCH_SIZE}/{len(entries)}")

    log.info(f"    Entities done: {inserted} attempted, {errors} errors")

    # ── Relationships ───────────────────────────────────────────────────────
    log.info(f"  --- Migrating relationships → {RELATION_TABLE} ---")
    vdb = load_vdb(RAG_STORAGE / "vdb_relationships.json")
    entries = vdb["data"]
    log.info(f"    Local entries: {len(entries)}")

    inserted = 0
    errors = 0
    batch_num = 0
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        rows = []
        for e in batch:
            try:
                vec = vec_to_pg(decode_vector(e["vector"]))
                ts = ts_from_epoch(e.get("__created_at__", ""))
                chunk_ids = split_source_ids(e.get("source_id", ""))
                rows.append((
                    e.get("__id__", ""),        # id
                    "",                          # workspace
                    e.get("src_id", ""),         # source_id
                    e.get("tgt_id", ""),         # target_id
                    e.get("content", ""),        # content
                    vec,                         # content_vector
                    ts,                          # create_time
                    ts,                          # update_time
                    chunk_ids,                   # chunk_ids (varchar(255)[])
                    e.get("file_path", ""),      # file_path
                ))
            except Exception as ex:
                errors += 1
                if errors <= 3:
                    log.warning(f"    Decode error: {ex}")

        if rows:
            if batch_num % 10 == 0:
                try:
                    await conn.close()
                except Exception:
                    pass
                conn = await pg_connect()
            try:
                await asyncio.wait_for(
                    conn.executemany(
                        f"""INSERT INTO "{RELATION_TABLE}"
                        (id, workspace, source_id, target_id, content, content_vector,
                         create_time, update_time, chunk_ids, file_path)
                        VALUES ($1,$2,$3,$4,$5,$6::vector,$7,$8,$9,$10)
                        ON CONFLICT (id, workspace) DO NOTHING""",
                        rows,
                    ),
                    timeout=120,
                )
                inserted += len(rows)
            except (asyncio.TimeoutError, Exception) as ex:
                log.warning(f"    Batch failed at {i + BATCH_SIZE}/{len(entries)}: {type(ex).__name__}, falling back to row-by-row...")
                try:
                    await conn.close()
                except Exception:
                    pass
                conn = await pg_connect()
                sql = f"""INSERT INTO "{RELATION_TABLE}"
                    (id, workspace, source_id, target_id, content, content_vector,
                     create_time, update_time, chunk_ids, file_path)
                    VALUES ($1,$2,$3,$4,$5,$6::vector,$7,$8,$9,$10)
                    ON CONFLICT (id, workspace) DO NOTHING"""
                for row in rows:
                    try:
                        await asyncio.wait_for(conn.execute(sql, *row), timeout=30)
                        inserted += 1
                    except Exception as row_ex:
                        errors += 1
                        if errors <= 10:
                            log.warning(f"    Row error: {row_ex}")
                        try:
                            await conn.close()
                        except Exception:
                            pass
                        conn = await pg_connect()
                conn = await pg_connect()
            batch_num += 1

        if (i + BATCH_SIZE) % 1000 == 0:
            log.info(f"    Relations: {i + BATCH_SIZE}/{len(entries)}")

    log.info(f"    Relations done: {inserted} attempted, {errors} errors")

    await conn.close()
    log.info("Step 6 complete: migration done.")
    return True


# ── Step 7: Update match_chunks() and verify ────────────────────────────────
async def update_and_verify():
    log.info("Step 7: Updating match_chunks() and verifying...")
    conn = await pg_connect()

    # Update match_chunks to point to the correct table
    log.info(f"  Updating match_chunks() to use {CHUNK_TABLE}...")
    await conn.execute(f"""
        CREATE OR REPLACE FUNCTION match_chunks(
          query_embedding vector(1536),
          match_count     int DEFAULT 80
        )
        RETURNS TABLE(content text, similarity float8)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
        AS $$
        BEGIN
          RETURN QUERY EXECUTE format(
            'SELECT content, 1 - (content_vector <=> $1) AS similarity
             FROM %I
             WHERE workspace = %L
             ORDER BY content_vector <=> $1
             LIMIT %s',
            '{CHUNK_TABLE}',
            '',
            match_count
          ) USING query_embedding;
        END;
        $$;
    """)
    log.info("  match_chunks() updated!")

    # Ensure permissions
    await conn.execute("""
        REVOKE ALL ON FUNCTION match_chunks(vector, int) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION match_chunks(vector, int) TO authenticated, service_role;
    """)

    # Test match_chunks with a real query
    try:
        sample = await conn.fetchrow(
            f'SELECT content_vector FROM "{CHUNK_TABLE}" LIMIT 1'
        )
        if sample:
            result = await conn.fetch(
                "SELECT content, similarity FROM match_chunks($1, 3)",
                sample["content_vector"],
            )
            log.info(f"  match_chunks test: returned {len(result)} rows")
            for r in result:
                log.info(f"    sim={r['similarity']:.4f} | {r['content'][:80]}...")
        else:
            log.warning("  No chunks in table — match_chunks test skipped")
    except Exception as ex:
        log.error(f"  match_chunks test failed: {ex}")

    # Rebuild HNSW indexes (dropped before bulk insert)
    log.info("  Rebuilding HNSW indexes (this may take a while)...")
    await conn.execute("SET statement_timeout = 0")
    for table, idx_name in [
        (CHUNK_TABLE, "idx_chunks_hnsw_cosine"),
        (ENTITY_TABLE, "idx_entity_hnsw_cosine"),
        (RELATION_TABLE, "idx_relation_hnsw_cosine"),
    ]:
        try:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx_name}
                ON "{table}" USING hnsw (content_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
            log.info(f"    Created {idx_name}")
        except Exception as ex:
            log.error(f"    Failed to create {idx_name}: {ex}")

    await conn.close()
    log.info("Step 7 complete: verification done.")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Overnight rebuild + migrate watchdog started")
    log.info(f"Time: {datetime.now()}")
    log.info("=" * 60)

    skip_to_step6 = "--step6" in sys.argv

    if not skip_to_step6:
        # Step 1: Wait for indexing
        wait_for_indexing()

        # Step 2: Rebuild graph edges (from existing VDB relationships only)
        rebuild_graph_edges()

        # Step 3: Restart LightRAG to validate
        if not restart_lightrag():
            log.error("Server restart failed — check /tmp/lightrag.log")
            log.error("Continuing anyway...")

        # Step 4: Re-embed ~21K missing entities
        reembed_missing_entities()

        # Step 5: Re-embed ~77K missing relationships + add edges to graph
        reembed_missing_relationships()

    # Stop LightRAG server to free pooler connections for migration
    log.info("Stopping LightRAG server to free pooler connections...")
    subprocess.run(["pkill", "-f", "lightrag-server"], capture_output=True)
    time.sleep(3)

    # Step 6: Migrate all VDB data (chunks + entities + relationships) to Supabase
    asyncio.run(migrate_to_supabase())

    # Step 7: Update match_chunks + verify
    asyncio.run(update_and_verify())

    log.info("=" * 60)
    log.info("ALL DONE!")
    log.info(f"Time: {datetime.now()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
