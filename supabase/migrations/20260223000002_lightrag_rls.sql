-- Enable Row Level Security on all LightRAG knowledge base tables.
--
-- Design: RLS is enabled with NO policies defined for anon/authenticated roles.
-- This means:
--   - Anon users:          blocked from all operations
--   - Authenticated users: blocked from all operations (access is via Edge Functions only)
--   - Service role key:    bypasses RLS automatically (used by rag-query Edge Function + LightRAG indexing)
--
-- NOTE: These tables are created by LightRAG on first server boot.
-- Run this migration AFTER starting the LightRAG server at least once so the tables exist.
-- LightRAG v1.4.10+ creates tables with lowercase names in Postgres.

-- KV / document tables
ALTER TABLE IF EXISTS public.lightrag_doc_full        ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_doc_chunks      ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_doc_status      ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_llm_cache       ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_full_entities   ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_full_relations  ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_entity_chunks   ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_relation_chunks ENABLE ROW LEVEL SECURITY;

-- Vector tables (gemini-embedding-001, 1536 dims)
ALTER TABLE IF EXISTS public.lightrag_vdb_chunks_gemini_embedding_001_1536d   ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_vdb_entity_gemini_embedding_001_1536d   ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lightrag_vdb_relation_gemini_embedding_001_1536d ENABLE ROW LEVEL SECURITY;
