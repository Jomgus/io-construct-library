#!/usr/bin/env python3
"""
Evaluate vector_only vs hybrid retrieval with stratified labeled queries.

Outputs:
- data/eval/ranking_eval_report.json
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cleaned_master_database.csv"
EVAL_DIR = ROOT / "data" / "eval"
BENCH_PATH = EVAL_DIR / "benchmark_io_gold_v1.jsonl"
REPORT_PATH = EVAL_DIR / "ranking_eval_report.json"
TUNED_PARAMS_PATH = EVAL_DIR / "tuned_hybrid_params.json"


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = (item.get("query") or "").strip()
            relevant = item.get("proposed_relevant_constructs") or item.get("relevant_constructs") or []
            stratum = (item.get("stratum") or "unknown").strip()
            if not query or not isinstance(relevant, list) or not relevant:
                continue
            out.append({"query": query, "relevant_constructs": [str(x) for x in relevant], "stratum": stratum})
    if not out:
        raise RuntimeError(f"No benchmark rows loaded from {path}")
    return out


@dataclass
class Row:
    idx: int
    name: str
    source: str
    definition: str
    paper_count: int
    name_n: str
    text_n: str


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            name = (raw.get("Construct_Name") or "").strip()
            if not name:
                continue
            try:
                paper = int(float((raw.get("Paper_Count") or "0").strip()))
            except Exception:
                paper = 0
            source = (raw.get("Source") or "").strip()
            definition = (raw.get("Definition_Text") or "").strip()
            name_n = norm(name)
            text_n = f"{name} {definition}"
            rows.append(
                Row(
                    idx=len(rows),
                    name=name,
                    source=source,
                    definition=definition,
                    paper_count=paper,
                    name_n=name_n,
                    text_n=norm(text_n),
                )
            )
    return rows


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


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def main() -> None:
    rows = load_rows(DATASET)
    bench = load_benchmark(BENCH_PATH)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    docs = [r.text_n for r in rows]
    names = [r.name for r in rows]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    doc_tfidf = vectorizer.fit_transform(docs)

    semantic_mode = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        doc_emb = np.asarray(model.encode(docs, normalize_embeddings=True, show_progress_bar=False))
        query_emb = np.asarray(model.encode([b["query"] for b in bench], normalize_embeddings=True, show_progress_bar=False))
        vec_scores = query_emb @ doc_emb.T
    except Exception:
        semantic_mode = "lsa_tfidf_fallback"
        q_tfidf_tmp = vectorizer.transform([b["query"] for b in bench])
        dim = max(32, min(256, q_tfidf_tmp.shape[1] - 1))
        svd = TruncatedSVD(n_components=dim, random_state=42)
        doc_lsa = svd.fit_transform(doc_tfidf)
        q_lsa = svd.transform(q_tfidf_tmp)
        doc_lsa = doc_lsa / (np.linalg.norm(doc_lsa, axis=1, keepdims=True) + 1e-12)
        q_lsa = q_lsa / (np.linalg.norm(q_lsa, axis=1, keepdims=True) + 1e-12)
        vec_scores = q_lsa @ doc_lsa.T

    q_tfidf = vectorizer.transform([b["query"] for b in bench])
    lex_scores = (q_tfidf @ doc_tfidf.T).toarray()

    # Mode params mirroring worker defaults.
    limit = 5
    candidate_pool = 200
    rrf_k = 50.0
    sem_w = 0.67
    lex_w = 0.33
    if TUNED_PARAMS_PATH.exists():
        tuned = json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))
        best = tuned.get("best_hybrid", {})
        sem_w = float(best.get("w_vec", sem_w))
        lex_w = float(best.get("w_lex", lex_w))

    lex_hits, lex_rrs = [], []
    vec_hits, vec_rrs = [], []
    hyb_hits, hyb_rrs = [], []
    per_stratum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for i, item in enumerate(bench):
        query_n = norm(item["query"])
        exact_bonus = np.array([1.0 if norm(n) == query_n else 0.0 for n in names], dtype=np.float64)

        # Lexical-only.
        lex = lex_scores[i] + (10.0 * exact_bonus)
        lex_idx = np.argsort(-lex)[:limit]
        lex_ranked = [names[j] for j in lex_idx]
        h0, rr0 = relevance_hit_and_rr(lex_ranked, item["relevant_constructs"], k=limit)
        lex_hits.append(h0)
        lex_rrs.append(rr0)

        # Vector-only.
        vec = vec_scores[i] + (10.0 * exact_bonus)
        vec_idx = np.argsort(-vec)[:limit]
        vec_ranked = [names[j] for j in vec_idx]
        h, rr = relevance_hit_and_rr(vec_ranked, item["relevant_constructs"], k=limit)
        vec_hits.append(h)
        vec_rrs.append(rr)

        # Hybrid weighted RRF.
        sem_order = np.argsort(-(vec_scores[i] + exact_bonus))[:candidate_pool]
        lex_order = np.argsort(-lex_scores[i])[:candidate_pool]

        rrf = np.zeros(len(rows), dtype=np.float64)
        for rank, idx in enumerate(sem_order, start=1):
            rrf[idx] += sem_w / (rrf_k + rank)
        for rank, idx in enumerate(lex_order, start=1):
            rrf[idx] += lex_w / (rrf_k + rank)
        hyb_idx = np.argsort(-rrf)[:limit]
        hyb_ranked = [names[j] for j in hyb_idx]
        h2, rr2 = relevance_hit_and_rr(hyb_ranked, item["relevant_constructs"], k=limit)
        hyb_hits.append(h2)
        hyb_rrs.append(rr2)

        s = item["stratum"]
        per_stratum[s]["lex_hit"].append(h0)
        per_stratum[s]["lex_mrr"].append(rr0)
        per_stratum[s]["vec_hit"].append(h)
        per_stratum[s]["vec_mrr"].append(rr)
        per_stratum[s]["hyb_hit"].append(h2)
        per_stratum[s]["hyb_mrr"].append(rr2)

    lex_hits_a = np.array(lex_hits, dtype=np.float64)
    lex_rrs_a = np.array(lex_rrs, dtype=np.float64)
    vec_hits_a = np.array(vec_hits, dtype=np.float64)
    vec_rrs_a = np.array(vec_rrs, dtype=np.float64)
    hyb_hits_a = np.array(hyb_hits, dtype=np.float64)
    hyb_rrs_a = np.array(hyb_rrs, dtype=np.float64)

    lex_hit_ci = bootstrap_ci(lex_hits_a)
    lex_mrr_ci = bootstrap_ci(lex_rrs_a)
    vec_hit_ci = bootstrap_ci(vec_hits_a)
    vec_mrr_ci = bootstrap_ci(vec_rrs_a)
    hyb_hit_ci = bootstrap_ci(hyb_hits_a)
    hyb_mrr_ci = bootstrap_ci(hyb_rrs_a)
    diff_hit_ci = bootstrap_ci_diff(hyb_hits_a, vec_hits_a)
    diff_mrr_ci = bootstrap_ci_diff(hyb_rrs_a, vec_rrs_a)

    report = {
        "semantic_mode": semantic_mode,
        "num_queries": len(bench),
        "strata_counts": {k: sum(1 for b in bench if b["stratum"] == k) for k in sorted({b["stratum"] for b in bench})},
        "lexical_only": {
            "hit_at_5": float(np.mean(lex_hits_a)),
            "mrr_at_5": float(np.mean(lex_rrs_a)),
            "hit_at_5_ci95": list(lex_hit_ci),
            "mrr_at_5_ci95": list(lex_mrr_ci),
        },
        "vector_only": {
            "hit_at_5": float(np.mean(vec_hits_a)),
            "mrr_at_5": float(np.mean(vec_rrs_a)),
            "hit_at_5_ci95": list(vec_hit_ci),
            "mrr_at_5_ci95": list(vec_mrr_ci),
        },
        "hybrid": {
            "hit_at_5": float(np.mean(hyb_hits_a)),
            "mrr_at_5": float(np.mean(hyb_rrs_a)),
            "hit_at_5_ci95": list(hyb_hit_ci),
            "mrr_at_5_ci95": list(hyb_mrr_ci),
        },
        "delta_hybrid_minus_vector": {
            "hit_at_5_ci95": list(diff_hit_ci),
            "mrr_at_5_ci95": list(diff_mrr_ci),
        },
        "per_stratum": {
            s: {
                "vector_hit_at_5": float(np.mean(v["vec_hit"])),
                "vector_mrr_at_5": float(np.mean(v["vec_mrr"])),
                "hybrid_hit_at_5": float(np.mean(v["hyb_hit"])),
                "hybrid_mrr_at_5": float(np.mean(v["hyb_mrr"])),
                "lexical_hit_at_5": float(np.mean(v["lex_hit"])),
                "lexical_mrr_at_5": float(np.mean(v["lex_mrr"])),
            }
            for s, v in per_stratum.items()
        },
        "hybrid_params": {
            "semantic_weight": sem_w,
            "lexical_weight": lex_w,
            "rrf_k": rrf_k,
            "candidate_pool": candidate_pool,
        },
        "selection_rule": {
            "primary_metric": "highest_mrr_at_5",
            "guardrail": "hit_at_5 must not drop materially vs top alternative",
            "confidence": "prefer mode with bootstrap CI clearly better or at least not worse",
        },
        "paths": {
            "benchmark": str(BENCH_PATH),
            "report": str(REPORT_PATH),
        },
    }

    # Winner selection by requested rule.
    candidates = {
        "lexical_only": report["lexical_only"],
        "vector_only": report["vector_only"],
        "hybrid": report["hybrid"],
    }
    winner = max(candidates.items(), key=lambda kv: (kv[1]["mrr_at_5"], kv[1]["hit_at_5"]))[0]
    winner_vs_best_alt = sorted(candidates.items(), key=lambda kv: kv[1]["mrr_at_5"], reverse=True)[:2]
    if len(winner_vs_best_alt) == 2:
        alt_name, alt = winner_vs_best_alt[1]
        win = candidates[winner]
        hit_drop = alt["hit_at_5"] - win["hit_at_5"]
        guardrail_pass = hit_drop <= 0.02
        report["winner"] = {
            "mode": winner,
            "runner_up": alt_name,
            "guardrail_pass": guardrail_pass,
            "hit_drop_vs_runner_up": hit_drop,
        }
    else:
        report["winner"] = {"mode": winner}

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"semantic_mode={semantic_mode}")
    print(f"queries={len(bench)} strata={report['strata_counts']}")
    print(
        "lexical_only hit@5={:.3f} mrr@5={:.3f}".format(
            report["lexical_only"]["hit_at_5"], report["lexical_only"]["mrr_at_5"]
        )
    )
    print(
        "vector_only hit@5={:.3f} mrr@5={:.3f}".format(
            report["vector_only"]["hit_at_5"], report["vector_only"]["mrr_at_5"]
        )
    )
    print(
        "hybrid      hit@5={:.3f} mrr@5={:.3f}".format(
            report["hybrid"]["hit_at_5"], report["hybrid"]["mrr_at_5"]
        )
    )
    print(f"vector_only hit@5 ci95={tuple(report['vector_only']['hit_at_5_ci95'])}")
    print(f"hybrid      hit@5 ci95={tuple(report['hybrid']['hit_at_5_ci95'])}")
    print(f"lexical_only hit@5 ci95={tuple(report['lexical_only']['hit_at_5_ci95'])}")
    print(f"vector_only mrr@5 ci95={tuple(report['vector_only']['mrr_at_5_ci95'])}")
    print(f"hybrid      mrr@5 ci95={tuple(report['hybrid']['mrr_at_5_ci95'])}")
    print(f"lexical_only mrr@5 ci95={tuple(report['lexical_only']['mrr_at_5_ci95'])}")
    print(f"delta(hybrid-vector) hit@5 ci95={tuple(report['delta_hybrid_minus_vector']['hit_at_5_ci95'])}")
    print(f"delta(hybrid-vector) mrr@5 ci95={tuple(report['delta_hybrid_minus_vector']['mrr_at_5_ci95'])}")
    print(f"winner={report['winner']}")
    print(f"benchmark={BENCH_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
