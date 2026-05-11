#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="python3"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <OFF_URL> <ON_URL>"
  exit 2
fi

OFF_URL="$1"
ON_URL="$2"

cd "$ROOT"

"$PY" scripts/evaluate_search_api_e2e.py \
  --off-url "$OFF_URL" \
  --on-url "$ON_URL" \
  --use-vercel-curl \
  --output data/eval/ambiguous_upgrade_report.json

"$PY" scripts/build_ambiguous_hard_negatives.py \
  --output data/eval/ambiguous_hard_negatives.jsonl

echo "wrote=data/eval/ambiguous_upgrade_report.json"
echo "wrote=data/eval/ambiguous_hard_negatives.jsonl"
