# Construct Library

Search tool for I/O psychology constructs built with `Next.js`, `Cloudflare Workers`, and `Neon/Postgres + pgvector`.

## What It Does

The app helps a user search for an I/O construct and get back a definition, source links, and related constructs.

It combines:

- a construct database built from OpenAlex and O*NET
- keyword and vector search
- optional reranking for harder queries
- benchmark and holdout evaluation

## Why I Built It

In I/O psychology, similar constructs often appear under different names, and exact keyword search misses too much.

The goal here was to make three things easier:

- exact construct lookups
- broader researcher queries
- finding nearby or overlapping constructs in the same result flow

## Current Search Approach

The live stack uses:

- `Next.js` frontend in `app/`
- Next API route in [`app/api/search/route.ts`](app/api/search/route.ts)
- `Cloudflare Worker` retrieval and ranking in [`cf-worker/src/index.ts`](cf-worker/src/index.ts)
- `Neon/Postgres + pgvector` for lexical and vector retrieval

Search modes:

- `hybrid`
  lexical search plus semantic search, merged with Reciprocal Rank Fusion
- `vector_only`
  semantic retrieval only

Optional reranking:

- `SEARCH_RERANK_MODE=off|cross_encoder_small`
- `cross_encoder_small` applies a second-pass cross-encoder reranker to top candidates

## Results Snapshot

Latest 120-query end-to-end benchmark:

- hybrid only (`OFF`): `Hit@5 = 0.9417`, `MRR@5 = 0.8172`
- hybrid + rerank (`ON`): `Hit@5 = 0.9750`, `MRR@5 = 0.9278`

The biggest gain showed up on ambiguous queries.

See:

- [`docs/SEARCH_METHODOLOGY.md`](docs/SEARCH_METHODOLOGY.md)
- [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md)
- [`data/eval/ranking_eval_e2e_v2_report.json`](data/eval/ranking_eval_e2e_v2_report.json)

## Repo Guide

- `app/`: frontend and API routes
- `cf-worker/`: Cloudflare Worker retrieval layer
- `scripts/`: ingestion, tuning, evaluation, and benchmark utilities
- `data/processed/`: merged construct database artifacts
- `data/eval/`: benchmark, audit, and evaluation outputs
- `docs/`: methodology and evaluation notes

## Quick Start

```bash
npm run dev
```

Then open `http://localhost:3000`.

## Key Files

- [`app/page.tsx`](app/page.tsx): search UI
- [`app/api/search/route.ts`](app/api/search/route.ts): search orchestration and rerank control
- [`app/api/rerank/route.ts`](app/api/rerank/route.ts): rerank API
- [`cf-worker/src/index.ts`](cf-worker/src/index.ts): hybrid and vector retrieval logic
- [`scripts/openalex_step1_pipeline.py`](scripts/openalex_step1_pipeline.py): OpenAlex ingestion
- [`scripts/tune_hybrid_search.py`](scripts/tune_hybrid_search.py): hybrid tuning
- [`scripts/evaluate_search_api_e2e.py`](scripts/evaluate_search_api_e2e.py): end-to-end evaluation

## Notes

- Production should keep `SEARCH_RERANK_MODE=off` until a deployment passes quality and latency checks.
- Holdout review is tracked separately from development benchmarks so later corpus changes do not blur earlier evaluation results.
