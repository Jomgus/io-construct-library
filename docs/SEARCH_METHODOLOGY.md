# Search Methodology

## Objective

Retrieve and rank I/O-relevant constructs for free-text user queries while:
- preserving precision on narrow construct queries,
- supporting semantic recall on paraphrased queries,
- handling ambiguous queries with useful top-k coverage,
- enabling safe runtime rollback.

## System Architecture

1. Frontend sends query + filters to `app/api/search/route.ts`.
2. API route forwards request to Cloudflare Worker.
3. Worker:
- computes query embedding with Cloudflare AI (`@cf/baai/bge-base-en-v1.5`),
- executes ranking SQL on Neon/Postgres `master_constructs`,
- returns ranked results and related constructs.

## Retrieval/Ranking Modes

Configured with `SEARCH_RANKING_MODE` in worker env:

- `hybrid` (default):
  - semantic candidate ranking from pgvector cosine distance,
  - lexical candidate ranking from PostgreSQL FTS (`websearch_to_tsquery`, `ts_rank_cd`),
  - weighted Reciprocal Rank Fusion (RRF).
- `vector_only`:
  - semantic ranking only + exact-match boost.

## Rerank Layer (Scaffolded)

Configured in Next API env:
- `SEARCH_RERANK_MODE=off|cross_encoder_small`
- `SEARCH_RERANK_CANDIDATE_LIMIT` (default `30`)

Current behavior:
1. Worker retrieves top-N candidates (`N = candidate limit` when rerank mode is on).
2. `GET /api/search` calls `POST /api/rerank` over HTTP with timeout.
3. `/api/rerank` calls external cross-encoder inference service.
4. Fail-open fallback returns worker-ranked results if rerank fails or times out.

Endpoints:
- Search orchestration: `GET /api/search`
- Rerank stub endpoint: `POST /api/rerank`

Note:
- `cross_encoder_small` uses external service contract (`CROSS_ENCODER_ENDPOINT`).
- Keep `SEARCH_RERANK_MODE=off` in production until quality + latency eval pass.

## Current Hybrid Logic

- Candidate pool: `max(60, min(500, limit * 10))`
- RRF `k`: `50`
- Weighted fusion:
  - semantic weight: from tuned params (`w_vec`)
  - lexical weight: from tuned params (`w_lex`)
- Tie-breaking:
  - exact name match,
  - fused score,
  - similarity score,
  - paper count,
  - stable row id.

## Data Constraints

Primary benchmark generation and tuning are constrained to approved I/O constructs from:
- `data/processed/openalex_enriched.csv`
- whitelisted journals only:
  - Journal of Applied Psychology
  - Personnel Psychology
  - Journal of Organizational Behavior
  - Journal of Occupational and Organizational Psychology
  - Academy of Management Journal
  - Journal of Management
  - Applied Psychology: An International Review
  - Journal of Vocational Behavior

## Indexing Requirements

Required for production performance:
- HNSW index on vector embeddings,
- GIN index on FTS expression,
- B-tree index on `("Source", "Paper_Count" DESC)`.

SQL script: `scripts/neon_hybrid_indexes.sql`.

## Rollback and Safety

- Set `SEARCH_RANKING_MODE="vector_only"` for immediate rollback path.
- Keep hybrid default for broader relevance coverage once validated.
- Evaluate both quality and latency before promoting any heavier reranking stage.
