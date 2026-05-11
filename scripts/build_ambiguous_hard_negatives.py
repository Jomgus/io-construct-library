#!/usr/bin/env python3
"""
Build ambiguous hard-negative set from latest e2e diagnostics.

Output:
- data/eval/ambiguous_hard_negatives.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
BENCH = EVAL_DIR / "benchmark_io_gold_v1.jsonl"
OUT = EVAL_DIR / "ambiguous_hard_negatives.jsonl"


def norm(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def load_benchmark() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with BENCH.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = (item.get("query") or "").strip()
            if not query:
                continue
            rows[query] = {
                "query": query,
                "stratum": str(item.get("stratum") or "unknown").replace("-", "_"),
                "positive_constructs": [str(x) for x in (item.get("proposed_relevant_constructs") or [])],
            }
    return rows


def latest_diag_on(path_override: str | None) -> Path:
    if path_override:
        return Path(path_override)
    candidates = sorted(EVAL_DIR.glob("e2e_diagnostics_on*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("No ON diagnostics file found (expected data/eval/e2e_diagnostics_on*.jsonl).")
    return candidates[0]


def id_to_name(cid: str) -> str:
    if "::" in cid:
        return cid.split("::", 1)[1]
    return cid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-on", default="")
    ap.add_argument("--rr-threshold", type=float, default=0.5)
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()

    bench = load_benchmark()
    diag_path = latest_diag_on(args.diag_on or None)
    output = Path(args.output)

    kept = []
    with diag_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query = str(row.get("query") or "")
            b = bench.get(query)
            if not b:
                continue
            if b["stratum"] != "ambiguous":
                continue
            rr = float(row.get("rr_at_5") or 0.0)
            if rr >= args.rr_threshold:
                continue
            positives = b["positive_constructs"]
            pos_norm = [norm(p) for p in positives]
            hard_negatives = []
            for cid in row.get("top5_ids") or []:
                name = id_to_name(str(cid))
                n = norm(name)
                if any(p in n or n in p for p in pos_norm):
                    continue
                hard_negatives.append(name)
            if not hard_negatives:
                continue
            kept.append(
                {
                    "query": query,
                    "positive_constructs": positives,
                    "hard_negative_constructs": hard_negatives[:8],
                    "stratum": "ambiguous",
                    "notes": f"on_rr_at_5={rr:.3f} below threshold={args.rr_threshold:.3f}",
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "diag_on": str(diag_path),
                "output": str(output),
                "rows": len(kept),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
