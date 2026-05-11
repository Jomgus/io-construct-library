-- Hybrid search index setup for master_constructs (Neon/Postgres + pgvector)
-- Run after data load/migration.

CREATE EXTENSION IF NOT EXISTS vector;

-- Vector ANN index for cosine distance.
-- Requires pgvector with HNSW support.
CREATE INDEX IF NOT EXISTS master_constructs_embedding_hnsw_idx
ON master_constructs
USING hnsw (embedding vector_cosine_ops);

-- Full-text index matching the expression used in worker query.
CREATE INDEX IF NOT EXISTS master_constructs_textsearch_gin_idx
ON master_constructs
USING gin (
  (
    setweight(to_tsvector('english', coalesce("Construct_Name", '')), 'A') ||
    setweight(to_tsvector('english', coalesce("Description", '')), 'B')
  )
);

-- Filter/sort helper for source + evidence threshold.
CREATE INDEX IF NOT EXISTS master_constructs_source_papercount_idx
ON master_constructs ("Source", "Paper_Count" DESC);

ANALYZE master_constructs;
