## Cloudflare Worker Setup

This worker requires one secret:

- `POSTGRES_URL` (Neon connection string)
  - `SEARCH_RANKING_MODE` is a non-secret runtime variable (`hybrid` or `vector_only`)

Set it with:

```bash
wrangler secret put POSTGRES_URL
```

Then deploy with your normal wrangler workflow.

To switch ranking mode instantly (no code change), update:

```toml
[vars]
SEARCH_RANKING_MODE = "hybrid"      # or "vector_only"
```

Then redeploy worker.

## Search Strategy

The worker uses hybrid retrieval:

- Semantic candidates from pgvector cosine distance
- Lexical candidates from PostgreSQL full-text search (`websearch_to_tsquery`)
- Reciprocal rank fusion (RRF) to combine both rankings

Rollback mode:

- `vector_only` bypasses lexical retrieval and RRF fusion.

Method details:

- `docs/SEARCH_METHODOLOGY.md`
- `docs/EVALUATION_PROTOCOL.md`

## Database Indexes (Required for Performance)

Run this script against Neon after loading `master_constructs`:

`scripts/neon_hybrid_indexes.sql`

It creates:

- HNSW vector index on `embedding`
- GIN full-text index matching the worker's lexical expression
- B-tree helper index for source/evidence threshold filters
