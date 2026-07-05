# Architecture

This document is the technical reference for Plastic Surgery Personal Tutor. It describes the repository layout, the backend Edge Functions, the retrieval pipeline, the database schema for vector search, the audio tutoring mode, and the complete map of LLM calls. For a step-by-step reproduction guide, see the [README](../README.md).

## 1. Design overview

The system separates **indexing** (offline, expensive, laptop-only) from **querying** (online, cheap, serverless):

```
                 OFFLINE (corpus build, laptop)
  PDFs ──▶ LightRAG server ──▶ chunks + knowledge graph + embeddings
                │   (OpenAI gpt-4o-mini extraction,
                │    OpenAI text-embedding-3-small)
                ▼
  scripts/migrate_to_supabase.py ──▶ Supabase Postgres (pgvector)

                 ONLINE (every user query, serverless)
  React SPA ──▶ rag-query Edge Function ──▶ pgvector search + graph traversal
      │                                          │
      │◀───────────── combined context ──────────┘
      ▼
  chat Edge Function ──▶ Google Gemini ──▶ grounded answer
```

Three properties follow from this split:

1. End users never depend on the indexing machine; all query traffic terminates at Supabase and the AI providers.
2. No LLM is called inside the retrieval layer at query time — LightRAG's synthesis step is bypassed (`only_need_context: true` semantics), and answer generation is performed exclusively by Gemini in the `chat` function.
3. All provider API keys live in Supabase Edge Function secrets; the frontend bundle contains none.

## 2. Repository layout

```
index.tsx                          # Entire frontend — a single React component plus audio helpers
index.html                         # HTML entry point (all dependencies bundled by Vite; no CDN scripts)
index.css                          # Tailwind import + custom styles
vite.config.ts                     # Vite config: Tailwind plugin; dev proxy /lightrag → localhost:9621
package.json                       # Scripts: dev, build, preview
.env.local.example                 # Template for .env.local (VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY)
supabase/
  config.toml                      # Supabase project configuration
  migrations/                      # Ordered SQL migrations (tables, RLS, pgvector, RPCs)
  functions/
    chat/index.ts                  # Gemini text chat (system instruction lives here, server-side)
    rag-query/index.ts             # Query embedding → pgvector search → graph traversal → context
    gemini-token/index.ts          # Auth-gated delivery of the Gemini key for audio sessions
scripts/
  migrate_to_supabase.py           # Bulk-load local LightRAG vectors → Supabase pgvector
  openai_proxy.py                  # Optional round-robin proxy across multiple OpenAI keys (indexing)
  reembed_missing_relations.py     # Repair utility: re-embed relationships from the LLM cache
  watchdog.sh                      # LightRAG server health watchdog for long indexing runs
LightRAG/                          # Vendored LightRAG (HKUDS, MIT) — offline indexing only
  .env.example                     # Template for LightRAG/.env (models, keys, Postgres connection)
  lightrag/                        # LightRAG library + API server source
  lightrag_webui/                  # Administration web UI (Documents tab used for PDF upload)
  rag_storage/                     # Runtime: local vector/graph storage during indexing (gitignored)
  inputs/                          # Runtime: uploaded PDF staging area (gitignored)
docs/
  ARCHITECTURE.md                  # This document
```

The frontend is intentionally a single file (`index.tsx`) with plain React state (`useState`/`useRef`) — no router and no state-management library. All UI logic, the audio-session machinery, quiz generation, and the Supabase client live there.

## 3. Edge Functions

All three functions run on Supabase's Deno runtime and read their secrets from the Edge Function secret store (`GEMINI_API_KEY`, `OPENAI_API_KEY`; `SUPABASE_SERVICE_ROLE_KEY` is injected automatically by the platform).

### 3.1 `rag-query`

Input: `{ query, top_k, entity_k, relation_k, graph_k }`.

1. Embeds the query with OpenAI `text-embedding-3-small` (1536 dimensions — this must match the model used at indexing time).
2. **Phase 1 — vector search.** Executes three pgvector similarity searches in parallel via `SECURITY DEFINER` RPCs: `match_chunks`, `match_entities`, `match_relations`.
3. **Phase 2 — graph traversal.** Takes the top entities from Phase 1 and calls `get_entity_edges` to retrieve their one-hop neighborhood in the knowledge graph (edges plus neighbor entity descriptions).
4. Merges and deduplicates the entities and relationships obtained from vector search and graph traversal.
5. Returns a combined plain-text context with three structured sections:

```
-----Chunks-----
-----Entities-----
-----Relationships-----
```

### 3.2 `chat`

Input: the conversation history plus the retrieved context. The function constructs a Gemini chat session (`gemini-2.5-flash`) with a server-side system instruction (the tutoring persona and grounding rules live in `supabase/functions/chat/index.ts`, never in the frontend bundle) and returns the generated answer. The function is stateless: the full history is sent on each request.

The frontend calls this function with a direct `fetch` using the stable anonymous key in the `Authorization` header, rather than a per-user JWT. User JWTs expire after roughly one hour and would cause intermittent 401 responses on a stateless, non-user-scoped function.

### 3.3 `gemini-token`

Verifies that the request carries a bearer token (authentication gate) and returns the Gemini API key for use by the frontend's real-time audio session. The key is served only for the duration of establishing a Live API connection and only to authenticated clients.

Design note: Gemini's ephemeral-token mechanism (API version `v1alpha`) was evaluated and rejected because, at the time of development, it ignored the `systemInstruction` field (a provider-acknowledged defect) and timed out when large system instructions were supplied at token minting. The auth-gated real-key approach on `v1beta` restores full system-instruction support.

## 4. Retrieval layer: tables, indexes, and RPCs

### 4.1 pgvector tables

LightRAG's vector namespaces map to three Postgres tables (row counts from the example deployment, for scale):

| Table | Contents | Rows (example) | HNSW index |
|---|---|---|---|
| `lightrag_vdb_chunks_…_1536d` | Document text chunks + embeddings | ~5,100 | `idx_chunks_hnsw_cosine` |
| `lightrag_vdb_entity_…_1536d` | Extracted entities (name, description) + embeddings | ~112,500 | `idx_entity_hnsw_cosine` |
| `lightrag_vdb_relation_…_1536d` | Extracted relationships (source, target, description) + embeddings | ~62,500 | `idx_relation_hnsw_cosine` |

All HNSW indexes use `m = 16`, `ef_construction = 200`, and `vector_cosine_ops`. Three btree indexes support graph traversal: `idx_relation_source_id` and `idx_relation_target_id` on the relation table, and `idx_entity_name` on the entity table.

### 4.2 RPCs (`SECURITY DEFINER`)

Direct client access to the `LIGHTRAG_*` tables is blocked (Section 6); retrieval goes exclusively through four SQL functions defined in the migrations:

| RPC | Signature (abridged) | Purpose |
|---|---|---|
| `match_chunks` | `(query_embedding vector(1536), match_count int DEFAULT 80)` | Chunk similarity search |
| `match_entities` | `(query_embedding vector(1536), match_count int DEFAULT 20)` | Entity similarity search |
| `match_relations` | `(query_embedding vector(1536), match_count int DEFAULT 20)` | Relationship similarity search |
| `get_entity_edges` | `(entity_names text[], max_edges int DEFAULT 40)` | One-hop graph traversal from named entities |

### 4.3 Query flow (text chat)

```
user message
  → frontend queryLightRAG()
  → rag-query { query, top_k: 80, entity_k: 20, relation_k: 20, graph_k: 40 }
      → embed query (OpenAI text-embedding-3-small)
      → Phase 1: match_chunks ∥ match_entities ∥ match_relations
      → Phase 2: get_entity_edges(top entities)
      → merge + deduplicate
  → combined context + message → chat Edge Function
      → Gemini 2.5 Flash synthesis
  → answer rendered in the UI
```

Error handling: `queryLightRAG()` retries once with a 2-second delay on HTTP 5xx. If the retrieval layer is unreachable, the UI states that the knowledge base is unavailable; if retrieval returns no relevant context, the answer explicitly states that the topic was not found in the knowledge base rather than falling back silently to the model's general knowledge.

### 4.4 Quiz generation

Quiz mode uses a reduced retrieval call (`top_k: 40`, 8-second timeout) followed by a direct Gemini call from the frontend with a quiz-specific system instruction. Recall-style questions run with a zero thinking budget for latency; board-style questions allow an extended thinking budget for clinical-vignette construction.

## 5. Audio tutoring mode

Audio mode connects the browser to the Gemini Live API (`gemini-2.5-flash-native-audio-preview-12-2025`, API version `v1beta`) for real-time spoken tutoring. It grounds the conversation in the knowledge base through two complementary mechanisms.

### 5.1 Pre-session context loading

Each audio topic is defined with four targeted sub-queries. At session start, all four run in parallel against `rag-query`; the concatenated results are capped at 120,000 characters and embedded directly into the Live API system instruction, together with level-specific teaching behavior (medical student / junior resident / senior resident) and quiz rules.

### 5.2 Real-time tool calling

The session registers a `search_knowledge_base` function tool (non-blocking behavior). When the model invokes it mid-conversation, the frontend intercepts the tool call, runs a fast retrieval variant — `top_k: 30`, results capped at 20,000 characters, a hard 8-second `AbortController` timeout, raw `fetch` rather than the Supabase client — and returns the chunks via `sendToolResponse()`. The UI shows a "Searching knowledge base…" indicator while the call is in flight.

### 5.3 Session establishment and audio

1. The frontend requests the Gemini key from `gemini-token` (authenticated request).
2. It constructs a Live API client with `apiVersion: "v1beta"` and opens the session with the full system instruction (knowledge-base context + behavioral rules).
3. Microphone audio is captured with echo cancellation, noise suppression, and automatic gain control; downsampled to 16 kHz PCM; and streamed to the model. The 24 kHz PCM response is played back through a separate output audio context.

Implementation notes: audio downsampling currently uses `ScriptProcessorNode` (deprecated but broadly supported; a migration to `AudioWorklet` is anticipated), and a audio session must be stopped before a new one is started to avoid multiple concurrent audio contexts.

## 6. Security architecture

### 6.1 Secret placement

| Secret | Location | Consumer |
|---|---|---|
| `GEMINI_API_KEY` | Edge Function secrets | `chat`, `gemini-token` |
| `OPENAI_API_KEY` | Edge Function secrets | `rag-query` (query embedding) |
| `SUPABASE_SERVICE_ROLE_KEY` | Injected by Supabase | `rag-query` (RPC invocation) |
| `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` | `.env.local` / frontend bundle | Public by design |
| OpenAI indexing key(s), Postgres password | `LightRAG/.env` (gitignored, laptop only) | Indexing pipeline |

### 6.2 Row-Level Security

| Table | SELECT | INSERT | UPDATE | DELETE |
|---|---|---|---|---|
| `LIGHTRAG_*` (knowledge base) | service role only | service role only | service role only | service role only |
| `user_progress` | authenticated users (leaderboard) | own row | own row | nobody |
| `chat_history` | own rows | own rows | own rows | own rows (data erasure) |
| `account_requests` | nobody (service role only) | anonymous (request form) | nobody | nobody |

The knowledge-base tables have RLS enabled with **no policies**, which denies every direct client query; the service role used by Edge Functions bypasses RLS, and the `SECURITY DEFINER` RPCs provide the only read path.

### 6.3 Access control

Public self-registration is disabled. The access-request form writes name and email (no password) to `account_requests`; the administrator — the owner of the Supabase project — reviews requests in the dashboard and issues invitations via **Authentication → Users → Invite user**, whereupon Supabase emails the invitee a link to set their own password.

## 7. LLM call map

| Step | Caller | Model / service | When |
|---|---|---|---|
| Entity/relationship extraction | LightRAG server (laptop) | OpenAI `gpt-4o-mini` (optionally via local key-rotation proxy) | Once per PDF, at indexing time |
| Chunk/entity/relation embedding | LightRAG server (laptop) | OpenAI `text-embedding-3-small` | Once per PDF, at indexing time |
| Query embedding | `rag-query` Edge Function | OpenAI `text-embedding-3-small` | Every retrieval |
| Retrieval | `rag-query` Edge Function | none (pgvector + SQL only) | Every retrieval |
| Chat answer synthesis | `chat` Edge Function | Gemini `gemini-2.5-flash` | Every user message |
| Quiz generation | Frontend (direct) | Gemini `gemini-2.5-flash` | On quiz request |
| Audio tool-call retrieval | Frontend → `rag-query` | none (context only) | During audio session |
| Audio audio/speech | Gemini Live API (auth-gated key, `v1beta`) | `gemini-2.5-flash-native-audio-preview-12-2025` | During audio session |

## 8. Indexing pipeline details

- **Chunking and extraction.** LightRAG splits each PDF into chunks and prompts `gpt-4o-mini` to extract entities and typed relationships, producing a document-level knowledge graph that is merged into the global graph.
- **Local-first storage.** During indexing, vectors are held in NanoVectorDB and the graph in NetworkX/JSON storage under `LightRAG/rag_storage/`. `scripts/migrate_to_supabase.py` then bulk-loads the vectors into Postgres: it drops each HNSW index, inserts rows in batches of 500 (`ON CONFLICT DO UPDATE`, so re-runs are safe), and rebuilds each index concurrently in a single pass.
- **Rate limiting.** The optional `scripts/openai_proxy.py` distributes extraction requests across up to three OpenAI keys; concurrency parameters in `LightRAG/.env` (`MAX_ASYNC`, `MAX_PARALLEL_INSERT`, `EMBEDDING_FUNC_MAX_ASYNC`, `EMBEDDING_BATCH_NUM`) should be reduced if the OpenAI API returns 429 responses.
- **Extraction caching.** LightRAG caches extraction LLM responses; re-processing an already-indexed document after a transient failure is served from cache at no additional API cost.
- **Invariant.** The embedding model and dimensionality (`text-embedding-3-small`, 1536) must remain constant across indexing and querying. Changing the embedding model requires deleting all vector tables and local storage and re-indexing from scratch.
- **Known failure modes.** Scanned PDFs without an OCR text layer fail with a "content contains only whitespace" error and must be OCR-processed externally; duplicate uploads fail with "content already exists" and are safe to ignore.

## 9. Frontend notes

- Tailwind CSS v4 is integrated through the Vite plugin; there is no `tailwind.config.js`. All dependencies (React, Tailwind, icons, the Gemini SDK) are bundled from `node_modules` — the page loads no CDN scripts.
- The Vite dev server proxies `/lightrag` to `localhost:9621` for local development against a running LightRAG server; the deployed app does not use this path.
- The header is responsive (two-row layout below the `md` breakpoint), and all dropdown menus are click-driven rather than hover-driven for touch-device compatibility.
