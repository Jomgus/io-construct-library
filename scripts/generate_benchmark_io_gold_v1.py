#!/usr/bin/env python3
"""
Generate IO gold benchmark v1 for human review.

Outputs:
- data/eval/benchmark_io_gold_v1.jsonl
- data/eval/review_sheet.csv
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "openalex_enriched.csv"
OUT_JSONL = ROOT / "data" / "eval" / "benchmark_io_gold_v1.jsonl"
OUT_REVIEW = ROOT / "data" / "eval" / "review_sheet.csv"

WHITELISTED_JOURNALS = {
    "Journal of Applied Psychology",
    "Personnel Psychology",
    "Journal of Organizational Behavior",
    "Journal of Occupational and Organizational Psychology",
    "Academy of Management Journal",
    "Journal of Management",
    "Applied Psychology: An International Review",
    "Journal of Vocational Behavior",
}

EXCLUDE_KEYWORDS = {
    "cancer",
    "tumor",
    "oncology",
    "gene",
    "genetic",
    "genome",
    "dna",
    "rna",
    "protein",
    "enzyme",
    "metabolism",
    "biochem",
    "neurology",
    "neurosurgery",
    "pathology",
    "infectious",
    "virus",
    "covid",
    "epidemiology",
    "cardio",
    "geology",
    "astronomy",
    "geography",
    "seismic",
    "volcano",
    "quantum",
    "nanotechnology",
    "astrophysics",
    "polymer",
    "crystallography",
    "alloy",
    "botany",
    "zoology",
    "molecule",
    "semiconductor",
    "climate change",
    "global warming",
    "geoengineering",
    "climate justice",
    "web design",
    "inventory turnover",
    "asian psychology",
    "design management",
}

SCALE_HINTS = {
    "scale",
    "scales",
    "measurement",
    "measure",
    "questionnaire",
    "inventory",
    "validity",
    "reliability",
    "psychometric",
    "factor analysis",
    "construct validity",
}

EXCLUDE_NAME_SUBSTRINGS = {
    "climate",
    "geoengineering",
    "global warming",
    "inventory turnover",
    "web design",
    "asian psychology",
    "statistical evidence",
    "systematic review",
    "evidence based",
    "evidence-based",
    "design management",
    "eyewitness identification",
}

GENERIC_AMBIGUOUS_TOKENS = {
    "change",
    "evidence",
    "design",
    "review",
    "policy",
}

FINAL_DENYLIST_TERMS = {
    "population",
    "statistics",
    "statistical",
    "methodology",
    "methodological",
    "meta analysis",
    "systematic review",
    "sampling",
    "sample size",
    "regression",
    "epidemiology",
    "climate change",
    "global warming",
    "geoengineering",
    "web design",
    "inventory turnover",
    "asian psychology",
}


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Construct:
    name: str
    paper_count: int
    definition: str
    journals: list[str]
    name_n: str
    text_n: str


def load_pool() -> list[Construct]:
    out: list[Construct] = []
    with SRC.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("Construct_Name") or "").strip()
            if not name:
                continue
            definition = (row.get("Definition_Text") or "").strip()
            name_n = norm(name)
            journals = [j.strip() for j in (row.get("Journals") or "").split(";") if j.strip()]
            if not journals:
                continue
            if any(j not in WHITELISTED_JOURNALS for j in journals):
                continue
            text_n = norm(f"{name} {definition}")
            if any(k in text_n for k in EXCLUDE_KEYWORDS):
                continue
            if any(k in name_n for k in EXCLUDE_NAME_SUBSTRINGS):
                continue
            try:
                pc = int(float((row.get("Total_Paper_Count") or "0").strip()))
            except Exception:
                pc = 0
            out.append(
                Construct(
                    name=name,
                    paper_count=pc,
                    definition=definition,
                    journals=journals,
                    name_n=name_n,
                    text_n=text_n,
                )
            )
    # de-duplicate by normalized name
    dedup: dict[str, Construct] = {}
    for c in out:
        if c.name_n not in dedup or c.paper_count > dedup[c.name_n].paper_count:
            dedup[c.name_n] = c
    return list(dedup.values())


def similar_constructs(
    query: str,
    pool: list[Construct],
    vectorizer: TfidfVectorizer,
    doc_matrix,
    min_k: int = 3,
    max_k: int = 8,
    anchor: str | None = None,
) -> tuple[list[str], float]:
    qv = vectorizer.transform([query])
    scores = (qv @ doc_matrix.T).toarray()[0]
    order = np.argsort(-scores)
    picked: list[str] = []
    picked_scores: list[float] = []
    if anchor:
        picked.append(anchor)
    for idx in order:
        cand = pool[int(idx)].name
        if cand in picked:
            continue
        if scores[int(idx)] <= 0:
            continue
        picked.append(cand)
        picked_scores.append(float(scores[int(idx)]))
        if len(picked) >= max_k:
            break
    if len(picked) < min_k:
        for idx in order:
            cand = pool[int(idx)].name
            if cand in picked:
                continue
            picked.append(cand)
            if len(picked) >= min_k:
                break
    picked = [
        x
        for x in picked[:max_k]
        if not any(b in norm(x) for b in EXCLUDE_NAME_SUBSTRINGS)
    ]
    return picked[:max_k], float(np.mean(picked_scores[: min(4, len(picked_scores))])) if picked_scores else 0.0


def generate() -> list[dict]:
    random.seed(42)
    np.random.seed(42)

    pool = load_pool()
    pool_sorted = sorted(pool, key=lambda x: (-x.paper_count, x.name_n))
    docs = [c.text_n for c in pool_sorted]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    doc_matrix = vectorizer.fit_transform(docs)

    # strata source pools
    broad_pool = [c for c in pool_sorted if c.paper_count >= 120]
    narrow_pool = [c for c in pool_sorted if 8 <= c.paper_count <= 45]
    scale_pool = [c for c in pool_sorted if any(h in c.text_n for h in SCALE_HINTS)]

    # ambiguous groups by shared token in construct name
    token_groups: dict[str, list[Construct]] = defaultdict(list)
    stop = {"work", "job", "employee", "organizational", "organization", "psychology", "theory", "model"}
    for c in pool_sorted:
        for t in set(c.name_n.split()):
            if len(t) < 5 or t in stop:
                continue
            token_groups[t].append(c)
    ambiguous_groups = []
    for token, vals in token_groups.items():
        uniq = {}
        for v in vals:
            uniq[v.name_n] = v
        group = sorted(uniq.values(), key=lambda x: (-x.paper_count, x.name_n))
        if 3 <= len(group) <= 10:
            ambiguous_groups.append((token, group))
    ambiguous_groups.sort(key=lambda x: (-len(x[1]), x[0]))

    rows: list[dict] = []
    used_queries = set()
    desired_candidates_per_stratum = 90

    def add_row(stratum: str, query: str, rationale: str, rel: list[str], conf: float) -> None:
        if query in used_queries:
            return
        # Confidence is heuristic quality-control score, not human validation.
        if conf < 0.8:
            return
        if not (3 <= len(rel) <= 8):
            return
        if any(any(b in norm(x) for b in EXCLUDE_NAME_SUBSTRINGS) for x in rel):
            return
        rows.append(
            {
                "id": f"io_gold_v1_{len(rows)+1:03d}",
                "stratum": stratum,
                "query": query,
                "proposed_relevant_constructs": rel,
                "rationale": rationale,
                "io_relevance_confidence": round(conf, 3),
            }
        )
        used_queries.add(query)

    # broad (30)
    broad_templates = [
        "overall {name} in workplace settings",
        "evidence on {name} in io psychology",
        "organizational outcomes linked to {name}",
    ]
    i = 0
    for c in broad_pool:
        q = broad_templates[i % len(broad_templates)].format(name=c.name)
        rel, base = similar_constructs(q, pool_sorted, vectorizer, doc_matrix, anchor=c.name)
        conf = 0.93 if len(rel) >= 5 else 0.86
        rationale = f"Broad I/O intent centered on high-frequency construct {c.name} and adjacent constructs from whitelisted-journal corpus."
        add_row("broad", q, rationale, rel, conf)
        i += 1
        if len([r for r in rows if r["stratum"] == "broad"]) >= desired_candidates_per_stratum:
            break

    # narrow (30)
    narrow_templates = [
        "specific construct definition for {name}",
        "how {name} differs from related constructs",
        "nomological placement of {name}",
    ]
    i = 0
    for c in narrow_pool:
        q = narrow_templates[i % len(narrow_templates)].format(name=c.name)
        rel, base = similar_constructs(q, pool_sorted, vectorizer, doc_matrix, anchor=c.name)
        conf = 0.9 if len(rel) >= 5 else 0.84
        rationale = f"Narrow intent targets a specific lower-frequency I/O construct ({c.name}) with nearest shortlist neighbors."
        add_row("narrow", q, rationale, rel, conf)
        i += 1
        if len([r for r in rows if r["stratum"] == "narrow"]) >= desired_candidates_per_stratum:
            break

    # ambiguous (30)
    for token, group in ambiguous_groups:
        if token in GENERIC_AMBIGUOUS_TOKENS:
            continue
        q = f"{token} construct in work context"
        rel = [g.name for g in group[:8]]
        rel = [x for x in rel if not any(b in norm(x) for b in EXCLUDE_NAME_SUBSTRINGS)]
        conf = 0.88 if len(rel) >= 4 else 0.81
        rationale = f"Ambiguous token '{token}' maps to multiple plausible I/O constructs in approved shortlist, requiring disambiguation."
        add_row("ambiguous", q, rationale, rel, conf)
        if len([r for r in rows if r["stratum"] == "ambiguous"]) >= desired_candidates_per_stratum:
            break

    # scale_focused (30)
    scale_templates = [
        "validated scale or measurement for {name}",
        "questionnaire and psychometric evidence for {name}",
        "reliability and validity indicators for {name}",
    ]
    i = 0
    for c in scale_pool:
        q = scale_templates[i % len(scale_templates)].format(name=c.name)
        rel, base = similar_constructs(q, pool_sorted, vectorizer, doc_matrix, anchor=c.name)
        conf = 0.92 if len(rel) >= 5 else 0.85
        rationale = f"Scale-focused intent seeks measurement instruments/psychometric neighbors for {c.name} in I/O literature."
        add_row("scale_focused", q, rationale, rel, conf)
        i += 1
        if len([r for r in rows if r["stratum"] == "scale_focused"]) >= desired_candidates_per_stratum:
            break

    # enforce exact 120 with 30 per stratum
    target = {"broad": 30, "narrow": 30, "ambiguous": 30, "scale_focused": 30}
    per_stratum_candidates: dict[str, list[dict]] = {}
    for s in ["broad", "narrow", "ambiguous", "scale_focused"]:
        picks = [r for r in rows if r["stratum"] == s and r["io_relevance_confidence"] >= 0.8]
        if len(picks) < target[s]:
            raise RuntimeError(f"Not enough rows for stratum={s}: have {len(picks)}, need {target[s]}")
        # Keep extra candidates for backfill after hard validation rejects rows.
        per_stratum_candidates[s] = picks[: max(target[s] * 3, 90)]
    out_unvalidated: list[dict] = []
    for s in ["broad", "narrow", "ambiguous", "scale_focused"]:
        out_unvalidated.extend(per_stratum_candidates[s])

    # Final hard validator before write.
    def has_denylisted_term(text: str) -> bool:
        t = norm(text)
        return any(term in t for term in FINAL_DENYLIST_TERMS)

    out: list[dict] = []
    rejected = 0
    for row in out_unvalidated:
        query_bad = has_denylisted_term(row["query"])
        targets_bad = any(has_denylisted_term(x) for x in row["proposed_relevant_constructs"])
        if query_bad or targets_bad:
            rejected += 1
            continue
        out.append(row)

    # If rejections created gaps, backfill from remaining candidates in same stratum.
    if rejected > 0:
        for s in ["broad", "narrow", "ambiguous", "scale_focused"]:
            have = len([r for r in out if r["stratum"] == s])
            need = target[s] - have
            if need <= 0:
                continue
            pool_s = [
                r
                for r in per_stratum_candidates[s]
                if r["id"] not in {x["id"] for x in out}
            ]
            for cand in pool_s:
                if has_denylisted_term(cand["query"]):
                    continue
                if any(has_denylisted_term(x) for x in cand["proposed_relevant_constructs"]):
                    continue
                out.append(cand)
                need -= 1
                if need == 0:
                    break

    # Trim to exact per-stratum target after validation/backfill.
    final_out: list[dict] = []
    for s in ["broad", "narrow", "ambiguous", "scale_focused"]:
        kept = [r for r in out if r["stratum"] == s]
        final_out.extend(kept[: target[s]])
    out = final_out

    # Hard fail if stratum balance is broken.
    for s in ["broad", "narrow", "ambiguous", "scale_focused"]:
        cnt = len([r for r in out if r["stratum"] == s])
        if cnt != target[s]:
            raise RuntimeError(f"Final validation left stratum={s} with {cnt} rows (expected {target[s]})")

    # Hard fail if any denylisted term remains.
    remaining = []
    for row in out:
        bad_targets = [t for t in row["proposed_relevant_constructs"] if has_denylisted_term(t)]
        if has_denylisted_term(row["query"]) or bad_targets:
            remaining.append({"id": row["id"], "query": row["query"], "bad_targets": bad_targets})
    if remaining:
        raise RuntimeError(f"Denylisted terms remain after validation: {remaining[:5]}")

    # attach summary metadata for main()
    generate.rejected_rows = rejected  # type: ignore[attr-defined]

    return out


def write_outputs(rows: list[dict]) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with OUT_REVIEW.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "proposed_relevant_constructs", "rationale", "reviewer_keep/edit"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "query": r["query"],
                    "proposed_relevant_constructs": "; ".join(r["proposed_relevant_constructs"]),
                    "rationale": r["rationale"],
                    "reviewer_keep/edit": "",
                }
            )


def main() -> None:
    rows = generate()
    write_outputs(rows)
    counts = defaultdict(int)
    target_counts = defaultdict(int)
    for r in rows:
        counts[r["stratum"]] += 1
        for t in r["proposed_relevant_constructs"]:
            target_counts[t] += 1

    rejected_rows = getattr(generate, "rejected_rows", 0)
    print(f"Rejected rows (final validator): {rejected_rows}")
    print(f"Wrote {len(rows)} rows to {OUT_JSONL}")
    print("Strata:", dict(counts))
    print("Top 20 most frequent target constructs:")
    for name, n in sorted(target_counts.items(), key=lambda kv: (-kv[1], kv[0].casefold()))[:20]:
        print(f"- {name}: {n}")
    print(f"Wrote review sheet: {OUT_REVIEW}")


if __name__ == "__main__":
    main()
