#!/usr/bin/env python3
"""
End-to-end ranking evaluation through /api/search.

Outputs:
- aggregate metrics with CIs
- per-stratum and per-intent metrics
- per-query diagnostics (candidate depth, rerank flags, top-5 ids)
- acceptance gate pass/fail
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "data" / "eval" / "benchmark_io_gold_v1.jsonl"
OUT_PATH = ROOT / "data" / "eval" / "ranking_eval_e2e_report.json"
OFF_DIAG_PATH = ROOT / "data" / "eval" / "e2e_diagnostics_off.jsonl"
ON_DIAG_PATH = ROOT / "data" / "eval" / "e2e_diagnostics_on.jsonl"
BASELINE_PATH = ROOT / "data" / "eval" / "ranking_eval_e2e_report.json"


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def construct_id(source: str, name: str) -> str:
    return f"{norm(source)}::{norm(name)}"


def relevance_hit_and_rr(ranked_names: list[str], relevant: list[str], k: int = 5) -> tuple[float, float]:
    rel = [norm(r) for r in relevant]
    for i, name in enumerate(ranked_names[:k], start=1):
        n = norm(name)
        if any(r in n or n in r for r in rel):
            return 1.0, 1.0 / i
    return 0.0, 0.0


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(values)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(values[idx]))
    low = float(np.percentile(samples, 100 * (alpha / 2)))
    high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return low, high


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(a)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(a[idx] - b[idx]))
    low = float(np.percentile(samples, 100 * (alpha / 2)))
    high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return low, high


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = (item.get("query") or "").strip()
            rel = item.get("proposed_relevant_constructs") or []
            stratum = (item.get("stratum") or "unknown").strip().replace("-", "_")
            if not query or not isinstance(rel, list) or not rel:
                continue
            out.append(
                {
                    "query": query,
                    "relevant_constructs": [str(x) for x in rel],
                    "stratum": stratum,
                }
            )
    if not out:
        raise RuntimeError(f"No benchmark rows loaded from {path}")
    return out


def fetch_json_http(base_url: str, query: str, limit: int) -> dict[str, Any]:
    qs = urllib.parse.urlencode({"q": query, "limit": str(limit)})
    url = f"{base_url.rstrip('/')}/api/search?{qs}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)


def fetch_json_vercel_curl(deployment_url: str, query: str, limit: int) -> dict[str, Any]:
    path = f"/api/search?q={urllib.parse.quote(query)}&limit={limit}"
    cmd = [
        "vercel",
        "curl",
        path,
        "--deployment",
        deployment_url,
        "--",
        "--silent",
        "--max-time",
        "30",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=45)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "vercel curl failed")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)
    raise RuntimeError(f"Could not parse JSON from vercel curl output: {proc.stdout[:500]}")


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"hit_at_5": 0.0, "mrr_at_5": 0.0}
    hits = np.array([r["hit_at_5"] for r in rows], dtype=np.float64)
    rrs = np.array([r["rr_at_5"] for r in rows], dtype=np.float64)
    return {
        "hit_at_5": float(np.mean(hits)),
        "mrr_at_5": float(np.mean(rrs)),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def evaluate_mode(
    mode_name: str,
    base_url: str,
    bench: list[dict[str, Any]],
    limit: int,
    use_vercel_curl: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fetcher = fetch_json_vercel_curl if use_vercel_curl else fetch_json_http

    for i, item in enumerate(bench):
        error_msg = ""
        timeout = False
        try:
            resp = fetcher(base_url, item["query"], limit)
            raw_results = resp.get("results") or []
            names = [str(r.get("constructName") or "") for r in raw_results]
            top5_ids = [
                construct_id(str(r.get("source") or ""), str(r.get("constructName") or ""))
                for r in raw_results[:limit]
            ]
            h, rr = relevance_hit_and_rr(names, item["relevant_constructs"], k=limit)
            rerank_error = resp.get("rerankError")
            rerank_error_text = str(rerank_error) if rerank_error else ""
            if "timeout" in rerank_error_text.lower() or "abort" in rerank_error_text.lower():
                timeout = True
            row = {
                "idx": i,
                "query": item["query"],
                "stratum": item["stratum"],
                "detected_intent": str(resp.get("detectedIntent") or "unknown"),
                "rewritten_query": str(resp.get("rewrittenQuery") or item["query"]),
                "candidate_depth_requested": int(resp.get("candidateDepthRequested") or 0),
                "candidate_count": int(resp.get("candidateCount") or 0),
                "rerank_mode": str(resp.get("rerankMode") or ""),
                "rerank_applied": bool(resp.get("rerankApplied")),
                "rerank_error": rerank_error_text,
                "timeout": timeout,
                "top5_ids": top5_ids,
                "hit_at_5": h,
                "rr_at_5": rr,
            }
        except Exception as exc:
            error_msg = str(exc)
            if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                timeout = True
            row = {
                "idx": i,
                "query": item["query"],
                "stratum": item["stratum"],
                "detected_intent": "unknown",
                "rewritten_query": item["query"],
                "candidate_depth_requested": 0,
                "candidate_count": 0,
                "rerank_mode": mode_name,
                "rerank_applied": False,
                "rerank_error": error_msg,
                "timeout": timeout,
                "top5_ids": [],
                "hit_at_5": 0.0,
                "rr_at_5": 0.0,
            }
        rows.append(row)

    hits = np.array([r["hit_at_5"] for r in rows], dtype=np.float64)
    rrs = np.array([r["rr_at_5"] for r in rows], dtype=np.float64)
    per_stratum_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_intent_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    depth_hist: dict[str, int] = defaultdict(int)
    for row in rows:
        per_stratum_map[row["stratum"]].append(row)
        per_intent_map[row["detected_intent"]].append(row)
        depth_hist[str(row["candidate_count"])] += 1

    timeout_rate = float(np.mean(np.array([1.0 if r["timeout"] else 0.0 for r in rows], dtype=np.float64)))
    return {
        "mode": mode_name,
        "base_url": base_url,
        "num_queries": len(bench),
        "hit_at_5": float(np.mean(hits)),
        "mrr_at_5": float(np.mean(rrs)),
        "hit_at_5_ci95": list(bootstrap_ci(hits)),
        "mrr_at_5_ci95": list(bootstrap_ci(rrs)),
        "timeout_rate": timeout_rate,
        "errors_count": int(sum(1 for r in rows if r["rerank_error"])),
        "per_stratum": {k: summarize_group(v) for k, v in per_stratum_map.items()},
        "per_intent": {k: summarize_group(v) for k, v in per_intent_map.items()},
        "candidate_depth_histogram": dict(sorted(depth_hist.items(), key=lambda kv: int(kv[0]))),
        "raw": {"hits": hits.tolist(), "rrs": rrs.tolist(), "rows": rows},
    }


def load_on_baseline(path: Path) -> dict[str, float | None]:
    out = {"overall_mrr": None, "broad_mrr": None}
    if not path.exists():
        return out
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        mode = obj.get("mode_cross_encoder_small", {})
        out["overall_mrr"] = float(mode.get("mrr_at_5"))
        out["broad_mrr"] = float(mode.get("per_stratum", {}).get("broad", {}).get("mrr_at_5"))
        return out
    except Exception:
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-url", required=True)
    ap.add_argument("--on-url", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--output", default=str(OUT_PATH))
    ap.add_argument("--benchmark", default=str(BENCH_PATH))
    ap.add_argument("--use-vercel-curl", action="store_true")
    args = ap.parse_args()

    bench_path = Path(args.benchmark)
    bench = load_benchmark(bench_path)
    baseline = load_on_baseline(BASELINE_PATH)

    off = evaluate_mode("off", args.off_url, bench, args.limit, args.use_vercel_curl)
    on = evaluate_mode("cross_encoder_small", args.on_url, bench, args.limit, args.use_vercel_curl)

    off_hits = np.array(off["raw"]["hits"], dtype=np.float64)
    off_rrs = np.array(off["raw"]["rrs"], dtype=np.float64)
    on_hits = np.array(on["raw"]["hits"], dtype=np.float64)
    on_rrs = np.array(on["raw"]["rrs"], dtype=np.float64)

    delta_hit_ci = bootstrap_ci_diff(on_hits, off_hits)
    delta_mrr_ci = bootstrap_ci_diff(on_rrs, off_rrs)

    ambiguous_on = on["per_stratum"].get("ambiguous", {"hit_at_5": 0.0, "mrr_at_5": 0.0})
    broad_on = on["per_stratum"].get("broad", {"hit_at_5": 0.0, "mrr_at_5": 0.0})
    broad_baseline = baseline.get("broad_mrr")
    broad_regression = (
        (float(broad_baseline) - broad_on["mrr_at_5"]) if broad_baseline is not None else 0.0
    )
    gates = {
        "ambiguous_hit_at_5_ge_0_75": bool(ambiguous_on["hit_at_5"] >= 0.75),
        "ambiguous_mrr_at_5_ge_0_55": bool(ambiguous_on["mrr_at_5"] >= 0.55),
        "broad_mrr_no_worse_than_minus_0_01_vs_baseline_on": bool(broad_regression <= 0.01),
        "timeout_rate_lt_0_02": bool(on["timeout_rate"] < 0.02),
    }
    gates["all_pass"] = all(gates.values())

    report = {
        "benchmark_path": str(bench_path),
        "num_queries": len(bench),
        "baseline_on_mrr_at_5": baseline.get("overall_mrr"),
        "baseline_on_broad_mrr_at_5": baseline.get("broad_mrr"),
        "mode_off": {k: v for k, v in off.items() if k != "raw"},
        "mode_cross_encoder_small": {k: v for k, v in on.items() if k != "raw"},
        "delta_cross_encoder_minus_off": {
            "hit_at_5": float(np.mean(on_hits - off_hits)),
            "mrr_at_5": float(np.mean(on_rrs - off_rrs)),
            "hit_at_5_ci95": list(delta_hit_ci),
            "mrr_at_5_ci95": list(delta_mrr_ci),
        },
        "acceptance_gates": gates,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_jsonl(OFF_DIAG_PATH, off["raw"]["rows"])
    write_jsonl(ON_DIAG_PATH, on["raw"]["rows"])

    print(json.dumps(report, indent=2))
    print(f"wrote={output}")
    print(f"wrote={OFF_DIAG_PATH}")
    print(f"wrote={ON_DIAG_PATH}")


if __name__ == "__main__":
    main()
