# Evaluation Protocol

## Scope

Defines how benchmark data is generated, reviewed, tuned against, and used for ranking decisions.

## Gold Benchmark Draft

Generator:
- `scripts/generate_benchmark_io_gold_v1.py`

Outputs:
- `data/eval/benchmark_io_gold_v1.jsonl`
- `data/eval/review_sheet.csv`

Required benchmark structure:
- 120 queries total
- 4 strata (30 each):
  - `broad`
  - `narrow`
  - `ambiguous`
  - `scale_focused`
- each query has:
  - `proposed_relevant_constructs` (3-8),
  - one-line rationale,
  - `io_relevance_confidence >= 0.8`.

## Hard Validation Rules

Before writing benchmark files, generator must:
- reject rows containing denylisted science/meta terms in query or targets,
- print:
  - rejected row count,
  - final row count per stratum,
  - top 20 most frequent target constructs,
- fail if any denylisted term remains.

Note:
- `io_relevance_confidence` is heuristic QC, not human ground truth.

## Human Review Gate

`review_sheet.csv` columns:
- `query`
- `proposed_relevant_constructs`
- `rationale`
- `reviewer_keep/edit`

Metrics are provisional until reviewer-approved labels are finalized.

## Tuning

Script:
- `scripts/tune_hybrid_search.py`

Inputs:
- `data/processed/cleaned_master_database.csv`
- `data/eval/benchmark_io_gold_v1.jsonl`

Output:
- `data/eval/tuned_hybrid_params.json`

Tuning objective:
- Primary: maximize `MRR@5`
- Secondary: maintain/improve `Hit@5`.

## Mode Evaluation with CIs

Script:
- `scripts/evaluate_ranking_modes.py`

Compared modes:
- `lexical_only`
- `vector_only`
- `hybrid`

Output:
- `data/eval/ranking_eval_report.json`

Required stats:
- `Hit@5`, `MRR@5` for each mode,
- bootstrap 95% CIs for each mode,
- paired delta CI for `hybrid - vector_only`.

## End-to-End API Evaluation with CIs

Script:
- `scripts/evaluate_search_api_e2e.py`

Purpose:
- evaluate through live `GET /api/search` path (includes runtime rerank/fallback behavior),
- compare `SEARCH_RERANK_MODE=off` vs `cross_encoder_small` deployments with paired bootstrap CIs.

Output:
- `data/eval/ranking_eval_e2e_report.json`

## Rerank Latency Gate

Script:
- `scripts/evaluate_rerank_latency.py`

Required env:
- `CROSS_ENCODER_ENDPOINT`
- optional auth/model timeout vars

Required latency report prior to enabling rerank by default:
- request success rate
- timeout/error rate
- p50/p95 latency

If endpoint is not configured, rerank remains disabled (`SEARCH_RERANK_MODE=off`).

## Winner Rule

Use this deterministic decision rule:
1. Primary metric: highest `MRR@5`.
2. Guardrail: `Hit@5` must not drop materially vs runner-up.
3. Confidence: prefer mode whose bootstrap CI is clearly better, or at least not worse.

If CIs overlap heavily, winner is operational/provisional rather than statistically decisive.

## Reproducible Command Set

```bash
python3 scripts/generate_benchmark_io_gold_v1.py
python3 scripts/tune_hybrid_search.py
python3 scripts/evaluate_ranking_modes.py
python3 scripts/evaluate_search_api_e2e.py --off-url <preview-off-url> --on-url <preview-on-url> --use-vercel-curl
```
