#!/usr/bin/env python3
"""
Tune hybrid retrieval weights for construct search.

Runs an offline benchmark over `data/processed/cleaned_master_database.csv` and compares:
- old lexical-style scoring
- vector-only scoring
- weighted hybrid scoring (grid search)

Usage:
  python3 scripts/tune_hybrid_search.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cleaned_master_database.csv"
BENCHMARK_PATH = ROOT / "data" / "eval" / "benchmark_io_gold_v1.jsonl"
TUNED_PARAMS_PATH = ROOT / "data" / "eval" / "tuned_hybrid_params.json"


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Row:
    idx: int
    name: str
    source: str
    definition: str
    paper_count: int
    name_n: str
    def_n: str
    text_n: str


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as f:
        for idx, raw in enumerate(csv.DictReader(f)):
            name = (raw.get("Construct_Name") or "").strip()
            if not name:
                continue
            source = (raw.get("Source") or "").strip()
            definition = (raw.get("Definition_Text") or "").strip()
            try:
                paper_count = int(float((raw.get("Paper_Count") or "0").strip()))
            except Exception:
                paper_count = 0
            name_n = norm(name)
            def_n = norm(definition)
            rows.append(
                Row(
                    idx=len(rows),
                    name=name,
                    source=source,
                    definition=definition,
                    paper_count=paper_count,
                    name_n=name_n,
                    def_n=def_n,
                    text_n=f"{name_n} {def_n}".strip(),
                )
            )
    return rows


def load_benchmark(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = (item.get("query") or "").strip()
            targets = item.get("proposed_relevant_constructs") or item.get("relevant_constructs") or []
            if not query or not isinstance(targets, list) or not targets:
                continue
            rows.append({"q": query, "targets": [str(t) for t in targets]})
    if not rows:
        raise RuntimeError(f"No benchmark rows loaded from {path}")
    return rows


def target_exists(rows: Iterable[Row], targets: list[str]) -> bool:
    target_norm = [norm(t) for t in targets]
    for row in rows:
        if any(t in row.name_n for t in target_norm):
            return True
    return False


def evaluate_ranked(ranked_names: list[str], targets: list[str], k: int = 5) -> tuple[float, float]:
    target_norm = [norm(t) for t in targets]
    for i, name in enumerate(ranked_names[:k], start=1):
        name_n = norm(name)
        if any(t in name_n for t in target_norm):
            return 1.0, 1.0 / i
    return 0.0, 0.0


def old_lexical_score(row: Row, q_norm: str, tokens: list[str]) -> float:
    if not q_norm:
        return math.log10(row.paper_count + 1) * 3
    score = 0.0
    if row.name_n == q_norm:
        score += 100
    if row.name_n.startswith(q_norm):
        score += 35
    if q_norm in row.name_n:
        score += 25
    if q_norm in row.def_n:
        score += 12
    for token in tokens:
        if token in row.name_n:
            score += 10
        if token in row.def_n:
            score += 4
    score += math.log10(row.paper_count + 1) * 5
    return score


def rank_old_lexical(rows: list[Row], query: str, k: int = 5, min_papers: int = 25) -> list[str]:
    q_norm = norm(query)
    tokens = [t for t in q_norm.split() if t]
    scored: list[tuple[float, str]] = []
    for row in rows:
        if row.source == "OpenAlex" and row.paper_count < min_papers:
            continue
        if tokens and not any(t in row.text_n for t in tokens):
            continue
        scored.append((old_lexical_score(row, q_norm, tokens), row.name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored[:k]]


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.matmul(a_norm, b_norm.T)


def softmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = np.sum(exp) + 1e-12
    return exp / denom


def main() -> None:
    rows = read_rows(DATASET)
    if not rows:
        raise RuntimeError(f"No rows loaded from {DATASET}")

    benchmark = load_benchmark(BENCHMARK_PATH)
    feasible = [x for x in benchmark if target_exists(rows, x["targets"])]  # type: ignore[arg-type]
    print(f"Rows loaded: {len(rows)}")
    print(f"Benchmark file: {BENCHMARK_PATH}")
    print(f"Benchmark queries: {len(benchmark)} | feasible: {len(feasible)}")

    # Build lexical matrix.
    docs = [r.text_n for r in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    doc_tfidf = vectorizer.fit_transform(docs)

    # Build semantic matrix.
    # Prefer local cached MiniLM; fallback to LSA if unavailable.
    semantic_mode = "sentence-transformers/all-MiniLM-L6-v2"
    model = None
    doc_emb = None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        doc_emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        semantic_mode = "lsa-tfidf-fallback"

    # Evaluate baselines.
    old_hits = old_mrr = 0.0
    vec_hits = vec_mrr = 0.0
    lex_hits = lex_mrr = 0.0

    query_texts = [q["q"] for q in feasible]  # type: ignore[index]
    q_tfidf = vectorizer.transform(query_texts)
    tfidf_scores = (q_tfidf @ doc_tfidf.T).toarray()
    if model is not None and doc_emb is not None:
        q_emb = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)
        vec_scores = cosine_sim_matrix(np.asarray(q_emb), np.asarray(doc_emb))
    else:
        dim = max(32, min(256, q_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=dim, random_state=42)
        doc_lsa = svd.fit_transform(doc_tfidf)
        q_lsa = svd.transform(q_tfidf)
        vec_scores = cosine_sim_matrix(np.asarray(q_lsa), np.asarray(doc_lsa))

    paper_bonus = np.array([math.log10(r.paper_count + 1) for r in rows], dtype=np.float64)
    openalex_mask = np.array([1.0 if r.source == "OpenAlex" else 0.0 for r in rows], dtype=np.float64)
    min_papers_mask = np.array([1.0 if (r.source != "OpenAlex" or r.paper_count >= 25) else 0.0 for r in rows], dtype=np.float64)

    for i, q in enumerate(feasible):
        targets = q["targets"]  # type: ignore[index]
        # old lexical
        old_ranked = rank_old_lexical(rows, q["q"], k=5, min_papers=25)  # type: ignore[index]
        h, rr = evaluate_ranked(old_ranked, targets, k=5)  # type: ignore[arg-type]
        old_hits += h
        old_mrr += rr

        # vector-only (with evidence mask)
        vec = vec_scores[i].copy()
        vec[min_papers_mask < 0.5] = -1e9
        top_vec_idx = np.argsort(-vec)[:5]
        vec_ranked = [rows[j].name for j in top_vec_idx]
        h, rr = evaluate_ranked(vec_ranked, targets, k=5)  # type: ignore[arg-type]
        vec_hits += h
        vec_mrr += rr

        # lexical-only tfidf
        lex = tfidf_scores[i].copy()
        lex[min_papers_mask < 0.5] = -1e9
        top_lex_idx = np.argsort(-lex)[:5]
        lex_ranked = [rows[j].name for j in top_lex_idx]
        h, rr = evaluate_ranked(lex_ranked, targets, k=5)  # type: ignore[arg-type]
        lex_hits += h
        lex_mrr += rr

    n = max(1, len(feasible))
    print("\nBaselines (feasible set)")
    print(f"semantic_mode={semantic_mode}")
    print(f"old_lexical   hit@5={old_hits/n:.3f} mrr@5={old_mrr/n:.3f}")
    print(f"tfidf_lexical hit@5={lex_hits/n:.3f} mrr@5={lex_mrr/n:.3f}")
    print(f"vector_only   hit@5={vec_hits/n:.3f} mrr@5={vec_mrr/n:.3f}")

    # Grid search weighted hybrid.
    best = None
    for w_lex in np.arange(0.1, 1.01, 0.1):
        for w_vec in np.arange(0.1, 1.01, 0.1):
            for w_paper in [0.0, 0.05, 0.1, 0.2]:
                for w_exact in [0.0, 0.5, 1.0]:
                    for w_openalex in [0.0, 0.05, 0.1]:
                        hit_sum = 0.0
                        mrr_sum = 0.0
                        for i, q in enumerate(feasible):
                            q_norm = norm(q["q"])  # type: ignore[index]
                            exact = np.array([1.0 if r.name_n == q_norm else 0.0 for r in rows], dtype=np.float64)

                            lex = softmax_norm(tfidf_scores[i])
                            vec = softmax_norm(vec_scores[i])

                            fused = (
                                w_lex * lex
                                + w_vec * vec
                                + w_paper * paper_bonus
                                + w_exact * exact
                                + w_openalex * openalex_mask
                            )
                            fused[min_papers_mask < 0.5] = -1e9

                            top_idx = np.argsort(-fused)[:5]
                            ranked = [rows[j].name for j in top_idx]
                            h, rr = evaluate_ranked(ranked, q["targets"], k=5)  # type: ignore[arg-type,index]
                            hit_sum += h
                            mrr_sum += rr

                        hit = hit_sum / n
                        mrr = mrr_sum / n
                        # Primary metric is MRR, guardrail metric is Hit@5.
                        key = (round(mrr, 6), round(hit, 6))
                        if best is None or key > best["key"]:
                            best = {
                                "key": key,
                                "hit": hit,
                                "mrr": mrr,
                                "weights": {
                                    "w_lex": float(w_lex),
                                    "w_vec": float(w_vec),
                                    "w_paper": float(w_paper),
                                    "w_exact": float(w_exact),
                                    "w_openalex": float(w_openalex),
                                },
                            }

    assert best is not None
    TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PARAMS_PATH.write_text(
        json.dumps(
            {
                "benchmark": str(BENCHMARK_PATH),
                "num_queries": len(feasible),
                "semantic_mode": semantic_mode,
                "best_hybrid": {
                    "hit_at_5": best["hit"],
                    "mrr_at_5": best["mrr"],
                    **best["weights"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nBest Hybrid")
    print(
        f"hit@5={best['hit']:.3f} mrr@5={best['mrr']:.3f} "
        f"weights={best['weights']}"
    )
    print(f"Wrote tuned params: {TUNED_PARAMS_PATH}")


if __name__ == "__main__":
    main()
