-- match_entities: similarity search over LightRAG entity vectors
-- Same pattern as match_chunks — SECURITY DEFINER to bypass RLS.
-- Table: lightrag_vdb_entity_text_embedding_3_small_1536d

CREATE OR REPLACE FUNCTION match_entities(
  query_embedding vector(1536),
  match_count     int DEFAULT 20
)
RETURNS TABLE(entity_name text, content text, similarity float8)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY EXECUTE format(
    'SELECT entity_name::text, content, 1 - (content_vector <=> $1) AS similarity
     FROM %I
     WHERE workspace = %L
     ORDER BY content_vector <=> $1
     LIMIT %s',
    'lightrag_vdb_entity_text_embedding_3_small_1536d',
    '',
    match_count
  ) USING query_embedding;
END;
$$;

REVOKE ALL ON FUNCTION match_entities(vector, int) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION match_entities(vector, int) TO authenticated, service_role;
