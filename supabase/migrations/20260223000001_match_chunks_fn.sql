-- match_chunks: similarity search over LightRAG chunk vectors
-- Uses dynamic SQL so this function can be created before LightRAG has run and created its tables.
-- Table name is fixed to the model + dimension used at indexing time:
--   gemini-embedding-001, 1536 dims → lightrag_vdb_chunks_gemini_embedding_001_1536d
--   NOTE: LightRAG v1.4.10+ creates tables with lowercase names in Postgres.
-- SECURITY DEFINER: runs with owner privileges so it can read the RLS-locked LightRAG table.
-- Only authenticated users and service role are allowed to call this function.

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
    'lightrag_vdb_chunks_gemini_embedding_001_1536d',
    '',
    match_count
  ) USING query_embedding;
END;
$$;

-- Remove default public execute privilege; grant only to authenticated users and service role
REVOKE ALL ON FUNCTION match_chunks(vector, int) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION match_chunks(vector, int) TO authenticated, service_role;
