-- Graph traversal support: btree indexes for 1-hop entity neighbor lookup
-- + get_entity_edges() RPC for the rag-query Edge Function

-- Btree indexes for fast 1-hop graph traversal on the relation table
CREATE INDEX IF NOT EXISTS idx_relation_source_id
ON lightrag_vdb_relation_text_embedding_3_small_1536d (source_id);

CREATE INDEX IF NOT EXISTS idx_relation_target_id
ON lightrag_vdb_relation_text_embedding_3_small_1536d (target_id);

-- Btree index on entity_name for fast neighbor description lookup
CREATE INDEX IF NOT EXISTS idx_entity_name
ON lightrag_vdb_entity_text_embedding_3_small_1536d (entity_name);

-- RPC: given entity names, return all 1-hop edges + neighbor entity descriptions
-- Used by rag-query Edge Function for graph traversal (LightRAG mix mode parity)
CREATE OR REPLACE FUNCTION get_entity_edges(
  entity_names text[],
  max_edges int DEFAULT 40
)
RETURNS TABLE(
  source_entity text,
  target_entity text,
  edge_description text,
  neighbor_name text,
  neighbor_description text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  WITH edges AS (
    SELECT DISTINCT ON (r.source_id, r.target_id)
      r.source_id::text,
      r.target_id::text,
      r.content AS edge_desc
    FROM lightrag_vdb_relation_text_embedding_3_small_1536d r
    WHERE r.workspace = ''
      AND (r.source_id::text = ANY(entity_names) OR r.target_id::text = ANY(entity_names))
    ORDER BY r.source_id, r.target_id
    LIMIT max_edges
  )
  SELECT
    e.source_id::text,
    e.target_id::text,
    e.edge_desc,
    n.entity_name::text,
    n.content
  FROM edges e
  LEFT JOIN lightrag_vdb_entity_text_embedding_3_small_1536d n
    ON n.entity_name = (
      CASE
        WHEN e.source_id::text = ANY(entity_names) THEN e.target_id
        ELSE e.source_id
      END
    )
    AND n.workspace = ''
  ;
END;
$$;

REVOKE ALL ON FUNCTION get_entity_edges(text[], int) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION get_entity_edges(text[], int) TO authenticated, service_role;
