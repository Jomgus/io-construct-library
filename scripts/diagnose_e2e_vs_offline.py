#!/usr/bin/env python3
"""
Per-query diagnostics for OFF/ON e2e search, plus offline-vs-e2e diffs.

Outputs:
- data/eval/e2e_diagnostics_off.jsonl
- data/eval/e2e_diagnostics_on.jsonl
- data/eval/offline_vs_e2e_diff.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cleaned_master_database.csv"
BENCH = ROOT / "data" / "eval" / "benchmark_io_gold_v1.jsonl"
TUNED = ROOT / "data" / "eval" / "tuned_hybrid_params.json"
OUT_OFF = ROOT / "data" / "eval" / "e2e_diagnostics_off.jsonl"
OUT_ON = ROOT / "data" / "eval" / "e2e_diagnostics_on.jsonl"
OUT_DIFF = ROOT / "data" / "eval" / "offline_vs_e2e_diff.csv"


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def construct_id(source: str, name: str) -> str:
    return f"{norm(source)}::{norm(name)}"


def relevance_hit_and_rr(ranked_ids: list[str], relevant: list[str], k: int = 5) -> tuple[float, float]:
    rel = [norm(r) for r in relevant]
    for i, cid in enumerate(ranked_ids[:k], start=1):
        name = cid.split("::", 1)[-1]
        if any(r in name or name in r for r in rel):
            return 1.0, 1.0 / i
    return 0.0, 0.0


@dataclass
class Row:
    name: str
    source: str
    text_n: str


def load_rows() -> list[Row]:
    rows: list[Row] = []
    with DATASET.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            name = (raw.get("Construct_Name") or "").strip()
            if not name:
                continue
            source = (raw.get("Source") or "").strip()
            definition = (raw.get("Definition_Text") or "").strip()
            rows.append(Row(name=name, source=source, text_n=norm(f"{name} {definition}")))
    return rows


def load_benchmark() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with BENCH.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = (item.get("query") or "").strip()
            rel = item.get("proposed_relevant_constructs") or []
            if not query or not isinstance(rel, list) or not rel:
                continue
            out.append(
                {
                    "query": query,
                    "stratum": item.get("stratum", "unknown"),
                    "relevant_constructs": [str(x) for x in rel],
                }
            )
    return out


def offline_hybrid_top5(rows: list[Row], bench: list[dict[str, Any]]) -> list[list[str]]:
    docs = [r.text_n for r in rows]
    names = [r.name for r in rows]
    sources = [r.source for r in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    doc_tfidf = vectorizer.fit_transform(docs)

    q_tfidf = vectorizer.transform([b["query"] for b in bench])
    # Offline proxy for semantics using TF-IDF itself (deterministic, no external model).
    vec_scores = (q_tfidf @ doc_tfidf.T).toarray()
    lex_scores = vec_scores.copy()

    candidate_pool = 200
    rrf_k = 50.0
    sem_w = 0.67
    lex_w = 0.33
    if TUNED.exists():
        tuned = json.loads(TUNED.read_text(encoding="utf-8"))
        best = tuned.get("best_hybrid", {})
        sem_w = float(best.get("w_vec", sem_w))
        lex_w = float(best.get("w_lex", lex_w))

    all_top5: list[list[str]] = []
    for i, item in enumerate(bench):
        query_n = norm(item["query"])
        exact_bonus = np.array([1.0 if norm(n) == query_n else 0.0 for n in names], dtype=np.float64)
        sem_order = np.argsort(-(vec_scores[i] + exact_bonus))[:candidate_pool]
        lex_order = np.argsort(-lex_scores[i])[:candidate_pool]
        rrf = np.zeros(len(rows), dtype=np.float64)
        for rank, idx in enumerate(sem_order, start=1):
            rrf[idx] += sem_w / (rrf_k + rank)
        for rank, idx in enumerate(lex_order, start=1):
            rrf[idx] += lex_w / (rrf_k + rank)
        idxs = np.argsort(-rrf)[:5]
        all_top5.append([construct_id(sources[j], names[j]) for j in idxs])
    return all_top5


def fetch_e2e(deployment_url: str, query: str) -> dict[str, Any]:
    path = f"/api/search?q={urllib.parse.quote(query)}&limit=5"
    cmd = [
        "vercel",
        "curl",
        path,
        "--deployment",
        deployment_url,
        "--",
        "--silent",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "vercel curl failed")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)
    raise RuntimeError(f"no JSON response for query={query}")


def run_mode(deployment_url: str, bench: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(bench):
        resp = fetch_e2e(deployment_url, item["query"])
        results = resp.get("results") or []
        top5_ids = [
            construct_id(str(r.get("source") or ""), str(r.get("constructName") or ""))
            for r in results[:5]
        ]
        hit, rr = relevance_hit_and_rr(top5_ids, item["relevant_constructs"], k=5)
        rows.append(
            {
                "idx": i,
                "query": item["query"],
                "stratum": item["stratum"],
                "candidateCount": int(resp.get("candidateCount") or 0),
                "rerankApplied": bool(resp.get("rerankApplied")),
                "rerankError": resp.get("rerankError"),
                "top5_ids": top5_ids,
                "hit_at_5": hit,
                "rr_at_5": rr,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-url", required=True)
    ap.add_argument("--on-url", required=True)
    args = ap.parse_args()

    bench = load_benchmark()
    rows = load_rows()
    offline_top5 = offline_hybrid_top5(rows, bench)
    off = run_mode(args.off_url, bench)
    on = run_mode(args.on_url, bench)

    write_jsonl(OUT_OFF, off)
    write_jsonl(OUT_ON, on)

    with OUT_DIFF.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "query",
                "stratum",
                "offline_top5_ids",
                "offline_rr_at_5",
                "e2e_off_top5_ids",
                "e2e_off_candidateCount",
                "e2e_off_rerankApplied",
                "e2e_off_rerankError",
                "e2e_off_rr_at_5",
                "e2e_on_top5_ids",
                "e2e_on_candidateCount",
                "e2e_on_rerankApplied",
                "e2e_on_rerankError",
                "e2e_on_rr_at_5",
                "delta_off_minus_offline_rr",
                "delta_on_minus_offline_rr",
            ],
        )
        w.writeheader()
        for i, b in enumerate(bench):
            offline_rr = relevance_hit_and_rr(offline_top5[i], b["relevant_constructs"], k=5)[1]
            off_rr = off[i]["rr_at_5"]
            on_rr = on[i]["rr_at_5"]
            w.writerow(
                {
                    "idx": i,
                    "query": b["query"],
                    "stratum": b["stratum"],
                    "offline_top5_ids": "|".join(offline_top5[i]),
                    "offline_rr_at_5": offline_rr,
                    "e2e_off_top5_ids": "|".join(off[i]["top5_ids"]),
                    "e2e_off_candidateCount": off[i]["candidateCount"],
                    "e2e_off_rerankApplied": off[i]["rerankApplied"],
                    "e2e_off_rerankError": off[i]["rerankError"],
                    "e2e_off_rr_at_5": off_rr,
                    "e2e_on_top5_ids": "|".join(on[i]["top5_ids"]),
                    "e2e_on_candidateCount": on[i]["candidateCount"],
                    "e2e_on_rerankApplied": on[i]["rerankApplied"],
                    "e2e_on_rerankError": on[i]["rerankError"],
                    "e2e_on_rr_at_5": on_rr,
                    "delta_off_minus_offline_rr": off_rr - offline_rr,
                    "delta_on_minus_offline_rr": on_rr - offline_rr,
                }
            )

    # concise console summary
    off_rrs = np.array([r["rr_at_5"] for r in off], dtype=np.float64)
    on_rrs = np.array([r["rr_at_5"] for r in on], dtype=np.float64)
    print(
        json.dumps(
            {
                "queries": len(bench),
                "off_mean_rr_at_5": float(np.mean(off_rrs)),
                "on_mean_rr_at_5": float(np.mean(on_rrs)),
                "off_diag": str(OUT_OFF),
                "on_diag": str(OUT_ON),
                "diff_csv": str(OUT_DIFF),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
