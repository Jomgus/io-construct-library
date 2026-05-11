#!/usr/bin/env python3
"""
Latency evaluation for cross-encoder rerank inference service.

Requires env:
- CROSS_ENCODER_ENDPOINT
Optional:
- CROSS_ENCODER_API_KEY
- CROSS_ENCODER_MODEL
- CROSS_ENCODER_TIMEOUT_MS (default 2500)
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "eval" / "benchmark_io_gold_v1.jsonl"
DATA = ROOT / "data" / "processed" / "cleaned_master_database.csv"
REPORT = ROOT / "data" / "eval" / "rerank_latency_report.json"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[max(0, min(len(values) - 1, idx))]


def is_huggingface_endpoint(endpoint: str) -> bool:
    return "api-inference.huggingface.co" in endpoint or "router.huggingface.co" in endpoint


def is_cloudflare_endpoint(endpoint: str) -> bool:
    return "api.cloudflare.com" in endpoint


def build_payload(
    endpoint: str,
    query: str,
    candidates: list[dict],
    model: str,
) -> dict:
    if is_huggingface_endpoint(endpoint):
        return {
            "inputs": [
                [query, f"{c['constructName']}. {c['definitionText']}"] for c in candidates
            ],
            "options": {"wait_for_model": True},
        }
    if is_cloudflare_endpoint(endpoint):
        return {
            "query": query,
            "contexts": [
                {
                    "id": idx,
                    "text": f"{c['constructName']}. {c['definitionText']}",
                }
                for idx, c in enumerate(candidates)
            ],
        }
    return {
        "query": query,
        "candidates": [
            {
                "index": idx,
                "title": c["constructName"],
                "text": f"{c['constructName']}. {c['definitionText']}",
                "source": c["source"],
            }
            for idx, c in enumerate(candidates)
        ],
        "model": model,
    }


def load_candidates() -> list[dict]:
    rows = []
    with DATA.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("Construct_Name") or "").strip()
            if not name:
                continue
            rows.append(
                {
                    "constructName": name,
                    "source": (r.get("Source") or "OpenAlex").strip(),
                    "definitionText": (r.get("Definition_Text") or "").strip(),
                    "paperCount": int(float((r.get("Paper_Count") or "0").strip() or 0)),
                    "referenceUrls": [],
                }
            )
    return rows


def load_queries() -> list[dict]:
    out = []
    with BENCH.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            out.append(
                {
                    "query": item["query"],
                    "targets": item.get("proposed_relevant_constructs", []),
                }
            )
    return out


def main() -> None:
    endpoint = (os.getenv("CROSS_ENCODER_ENDPOINT") or "").strip()
    if not endpoint:
        print("Missing CROSS_ENCODER_ENDPOINT. Cannot run latency evaluation.")
        raise SystemExit(2)

    api_key = (os.getenv("CROSS_ENCODER_API_KEY") or "").strip()
    model = (os.getenv("CROSS_ENCODER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
    timeout_ms = int(float(os.getenv("CROSS_ENCODER_TIMEOUT_MS") or "2500"))
    timeout_s = max(0.2, timeout_ms / 1000.0)

    random.seed(42)
    all_candidates = load_candidates()
    queries = load_queries()
    subset = queries[: min(60, len(queries))]

    durations = []
    timeouts = 0
    failures = 0
    successes = 0
    status_counts: dict[str, int] = {}
    error_examples: list[dict[str, str]] = []

    for q in subset:
        # Simulate top-30 candidate rerank payload.
        candidates = random.sample(all_candidates, k=min(30, len(all_candidates)))
        payload = build_payload(endpoint, q["query"], candidates, model)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            method="POST",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                _ = resp.read()
                if resp.status != 200:
                    failures += 1
                    key = str(resp.status)
                    status_counts[key] = status_counts.get(key, 0) + 1
                else:
                    successes += 1
                    durations.append((time.perf_counter() - t0) * 1000.0)
                    status_counts["200"] = status_counts.get("200", 0) + 1
        except urllib.error.HTTPError as e:
            failures += 1
            key = str(getattr(e, "code", "http_error"))
            status_counts[key] = status_counts.get(key, 0) + 1
            if len(error_examples) < 5:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    body = "<failed to decode error body>"
                error_examples.append({"status": key, "body": body})
        except urllib.error.URLError as e:
            failures += 1
            msg = str(e.reason).lower()
            status_counts["url_error"] = status_counts.get("url_error", 0) + 1
            if "timed out" in msg or "timeout" in msg:
                timeouts += 1
            if len(error_examples) < 5:
                error_examples.append({"status": "url_error", "body": str(e.reason)[:300]})
        except TimeoutError:
            failures += 1
            timeouts += 1
            status_counts["timeout"] = status_counts.get("timeout", 0) + 1
            if len(error_examples) < 5:
                error_examples.append({"status": "timeout", "body": "TimeoutError"})
        except Exception:
            failures += 1
            status_counts["exception"] = status_counts.get("exception", 0) + 1

    report = {
        "endpoint": endpoint,
        "model": model,
        "timeout_ms": timeout_ms,
        "requests": len(subset),
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "latency_ms": {
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "mean": statistics.mean(durations) if durations else 0.0,
        },
        "status_counts": status_counts,
        "error_examples": error_examples,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote={REPORT}")


if __name__ == "__main__":
    main()
