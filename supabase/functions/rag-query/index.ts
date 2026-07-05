import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: CORS_HEADERS });
  }

  try {
    const { query, top_k = 80, entity_k = 20, relation_k = 20, graph_k = 40 } = await req.json();

    if (!query || typeof query !== "string") {
      return new Response(
        JSON.stringify({ error: "query is required" }),
        { status: 400, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
      );
    }

    const openaiKey = Deno.env.get("OPENAI_API_KEY");
    if (!openaiKey) {
      return new Response(
        JSON.stringify({ error: "Service configuration error" }),
        { status: 500, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
      );
    }

    // 1. Embed query using OpenAI text-embedding-3-small (must match indexing model)
    const embedRes = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${openaiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: query,
        dimensions: 1536,
      }),
    });

    if (!embedRes.ok) {
      const errText = await embedRes.text();
      console.error("[rag-query] OpenAI embedding error:", embedRes.status, errText);
      return new Response(
        JSON.stringify({ response: null }),
        { headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
      );
    }

    const embedData = await embedRes.json();
    const embedding: number[] = embedData?.data?.[0]?.embedding;

    if (!embedding || embedding.length === 0) {
      console.error("[rag-query] Empty embedding returned");
      return new Response(
        JSON.stringify({ response: null }),
        { headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
      );
    }

    // 2. Phase 1: pgvector similarity search — chunks, entities, relationships in parallel
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
    );

    const [chunksRes, entitiesRes, relationsRes] = await Promise.all([
      supabase.rpc("match_chunks", {
        query_embedding: embedding,
        match_count: top_k,
      }),
      supabase.rpc("match_entities", {
        query_embedding: embedding,
        match_count: entity_k,
      }),
      supabase.rpc("match_relations", {
        query_embedding: embedding,
        match_count: relation_k,
      }),
    ]);

    if (chunksRes.error) {
      console.error("[rag-query] chunks error:", chunksRes.error);
    }
    if (entitiesRes.error) {
      console.error("[rag-query] entities error:", entitiesRes.error);
    }
    if (relationsRes.error) {
      console.error("[rag-query] relations error:", relationsRes.error);
    }

    const chunks = chunksRes.data ?? [];
    const entities: { entity_name: string; content: string; similarity: number }[] = entitiesRes.data ?? [];
    const relations: { source_id: string; target_id: string; content: string; similarity: number }[] = relationsRes.data ?? [];

    // 3. Phase 2: 1-hop graph traversal from top vector-matched entities
    //    This discovers structurally connected neighbors that may not rank
    //    high in vector similarity but are directly linked in the knowledge graph.
    const entityNames = entities.map((e) => e.entity_name);
    let graphEdges: { source_entity: string; target_entity: string; edge_description: string; neighbor_name: string; neighbor_description: string }[] = [];

    if (entityNames.length > 0 && graph_k > 0) {
      const graphRes = await supabase.rpc("get_entity_edges", {
        entity_names: entityNames,
        max_edges: graph_k,
      });
      if (graphRes.error) {
        console.error("[rag-query] graph traversal error:", graphRes.error);
      } else {
        graphEdges = graphRes.data ?? [];
      }
    }

    // 4. Merge vector results + graph traversal results (deduplicate)
    const entitySet = new Set(entityNames);
    const allEntities: { name: string; description: string }[] = entities.map((e) => ({
      name: e.entity_name,
      description: e.content,
    }));

    // Add graph neighbor entities not already in vector results
    for (const edge of graphEdges) {
      if (edge.neighbor_name && !entitySet.has(edge.neighbor_name)) {
        entitySet.add(edge.neighbor_name);
        allEntities.push({
          name: edge.neighbor_name,
          description: edge.neighbor_description || "",
        });
      }
    }

    // Deduplicate relations: existing vector relations + graph edges
    const relationKeySet = new Set(
      relations.map((r) => `${r.source_id}||${r.target_id}`)
    );
    const allRelations: { source: string; target: string; description: string }[] = relations.map((r) => ({
      source: r.source_id,
      target: r.target_id,
      description: r.content,
    }));

    for (const edge of graphEdges) {
      const key = `${edge.source_entity}||${edge.target_entity}`;
      const keyRev = `${edge.target_entity}||${edge.source_entity}`;
      if (!relationKeySet.has(key) && !relationKeySet.has(keyRev)) {
        relationKeySet.add(key);
        allRelations.push({
          source: edge.source_entity,
          target: edge.target_entity,
          description: edge.edge_description,
        });
      }
    }

    if (chunks.length === 0 && allEntities.length === 0 && allRelations.length === 0) {
      return new Response(
        JSON.stringify({ response: "" }),
        { headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
      );
    }

    // 5. Build combined context with structured sections
    const sections: string[] = [];

    if (chunks.length > 0) {
      const chunkText = chunks
        .map((row: { content: string }) => row.content)
        .join("\n\n---\n\n");
      sections.push(`-----Chunks-----\n${chunkText}`);
    }

    if (allEntities.length > 0) {
      const entityText = allEntities
        .map((e) => `${e.name}: ${e.description}`)
        .join("\n");
      sections.push(`-----Entities-----\n${entityText}`);
    }

    if (allRelations.length > 0) {
      const relationText = allRelations
        .map((r) => `${r.source} → ${r.target}: ${r.description}`)
        .join("\n");
      sections.push(`-----Relationships-----\n${relationText}`);
    }

    const context = sections.join("\n\n");

    return new Response(
      JSON.stringify({ response: context }),
      { headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
    );

  } catch (err) {
    console.error("[rag-query] Unexpected error:", err);
    return new Response(
      JSON.stringify({ response: null }),
      { headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
    );
  }
});
