#!/usr/bin/env python3
"""
Build benchmark_io_gold_v2 by pruning/rebuilding ambiguous rows from v1.

Outputs:
- data/eval/benchmark_io_gold_v2.jsonl
- data/eval/review_sheet_v2.csv
- data/eval/removed_queries_v2.csv
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "eval" / "benchmark_io_gold_v1.jsonl"
DB = ROOT / "data" / "processed" / "cleaned_master_database.csv"
OUT = ROOT / "data" / "eval" / "benchmark_io_gold_v2.jsonl"
REVIEW = ROOT / "data" / "eval" / "review_sheet_v2.csv"
REMOVED = ROOT / "data" / "eval" / "removed_queries_v2.csv"

DENYLIST = {
    "climate change",
    "global warming",
    "geoengineering",
    "web design",
    "inventory turnover",
    "asian psychology",
    "medical",
    "oncology",
    "biology",
    "genetics",
    "geography",
    "meteorology",
    "astrophysics",
    "methodology",
    "meta analysis",
    "systematic review",
    "population",
    "statistics",
    "statistical",
}


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_denylist(text: str) -> str | None:
    t = norm(text)
    for term in DENYLIST:
        if term in t:
            return term
    return None


def load_v1() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with V1.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_construct_pool() -> list[dict[str, Any]]:
    rows = []
    with DB.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("Construct_Name") or "").strip()
            if not name:
                continue
            definition = (r.get("Definition_Text") or "").strip()
            try:
                pc = int(float((r.get("Paper_Count") or "0").strip() or 0))
            except Exception:
                pc = 0
            rows.append(
                {
                    "name": name,
                    "name_n": norm(name),
                    "text_n": norm(f"{name} {definition}"),
                    "paper_count": pc,
                }
            )
    dedup = {}
    for r in rows:
        key = r["name_n"]
        if key not in dedup or r["paper_count"] > dedup[key]["paper_count"]:
            dedup[key] = r
    return sorted(dedup.values(), key=lambda x: (-x["paper_count"], x["name_n"]))


AMBIGUOUS_V2_SPECS = [
    {"query": "justice perceptions at work", "ambiguity_type": "polysemy", "seeds": ["organizational justice", "procedural justice", "distributive justice", "interactional justice"], "keep_reason": "Common I/O query where justice facets are easily conflated."},
    {"query": "commitment at work", "ambiguity_type": "polysemy", "seeds": ["organizational commitment", "affective commitment", "continuance commitment", "normative commitment"], "keep_reason": "Research users often omit commitment facet in first-pass query."},
    {"query": "burnout in employees", "ambiguity_type": "polysemy", "seeds": ["burnout", "occupational burnout", "emotional exhaustion"], "keep_reason": "Ambiguous between syndrome-level and subdimension constructs."},
    {"query": "citizenship behavior at work", "ambiguity_type": "scale_alias", "seeds": ["organizational citizenship behavior", "organizational commitment", "contextual performance"], "keep_reason": "Alias-heavy area (OCB terms and close neighbors)."},
    {"query": "counterproductive behavior at work", "ambiguity_type": "scale_alias", "seeds": ["counterproductive work behavior", "workplace deviance", "abuse of power"], "keep_reason": "Multiple near-synonymous labels used across papers."},
    {"query": "work family conflict construct", "ambiguity_type": "scale_alias", "seeds": ["work family conflict", "work-family conflict", "work life balance"], "keep_reason": "Alias and directionality confusion (WFC vs balance)."},
    {"query": "leadership style effects at work", "ambiguity_type": "scope", "seeds": ["leadership style", "transformational leadership", "shared leadership", "authoritarian leadership style"], "keep_reason": "Broad leadership query without explicit theory narrows poorly."},
    {"query": "job attitude construct", "ambiguity_type": "scale_alias", "seeds": ["job attitude", "job satisfaction", "job dissatisfaction"], "keep_reason": "Frequent synonym usage between job attitude and satisfaction family."},
    {"query": "employee voice and silence", "ambiguity_type": "polysemy", "seeds": ["voice behavior", "employee silence", "psychological safety"], "keep_reason": "Voice/silence pairing is often underspecified in practical search."},
    {"query": "psychological safety in teams", "ambiguity_type": "scope", "seeds": ["psychological safety", "workplace safety", "teamwork"], "keep_reason": "Users conflate psychosocial and physical safety constructs."},
    {"query": "engagement at work", "ambiguity_type": "polysemy", "seeds": ["work engagement", "employee engagement", "job involvement"], "keep_reason": "Engagement has multiple neighboring constructs and scales."},
    {"query": "motivation at work", "ambiguity_type": "scope", "seeds": ["motivation", "intrinsic motivation", "extrinsic motivation"], "keep_reason": "High-frequency but multi-family construct cluster in I/O."},
    {"query": "stress at work", "ambiguity_type": "polysemy", "seeds": ["occupational stress", "job stress", "work stress"], "keep_reason": "General stress queries need facet disambiguation."},
    {"query": "turnover construct in organizations", "ambiguity_type": "polysemy", "seeds": ["turnover", "turnover intention", "job insecurity"], "keep_reason": "Intention vs behavior ambiguity is common in search logs."},
    {"query": "performance at work", "ambiguity_type": "scope", "seeds": ["job performance", "contextual performance", "task performance"], "keep_reason": "Performance is too broad unless dimension is specified."},
    {"query": "personality and job outcomes", "ambiguity_type": "scope", "seeds": ["big five personality traits", "conscientiousness", "extraversion and introversion", "neuroticism"], "keep_reason": "Trait-level ambiguity needs controlled candidate set."},
    {"query": "selection test quality", "ambiguity_type": "method_vs_construct", "seeds": ["test validity", "criterion validity", "construct validity", "content validity"], "keep_reason": "Users mix method language with validation constructs."},
    {"query": "scale reliability for leadership", "ambiguity_type": "method_vs_construct", "seeds": ["leadership style", "transformational leadership", "inter-rater reliability"], "keep_reason": "Method and construct terms co-occur in realistic eval queries."},
    {"query": "organizational support construct", "ambiguity_type": "polysemy", "seeds": ["perceived organizational support", "organizational commitment", "emotional support"], "keep_reason": "Support terms overlap social and organizational levels."},
    {"query": "fairness at work", "ambiguity_type": "scale_alias", "seeds": ["organizational justice", "procedural justice", "distributive justice"], "keep_reason": "Fairness is a lay alias for justice dimensions."},
    {"query": "team climate and cohesion", "ambiguity_type": "polysemy", "seeds": ["team climate", "group cohesion", "teamwork"], "keep_reason": "Closely related team constructs are frequently conflated."},
    {"query": "telecommuting outcomes", "ambiguity_type": "scope", "seeds": ["telecommuting", "work life balance", "job satisfaction"], "keep_reason": "Remote-work queries are often broad and outcome-ambiguous."},
    {"query": "workload and cognitive load at work", "ambiguity_type": "polysemy", "seeds": ["workload", "cognitive load", "working memory"], "keep_reason": "Workload vs cognitive-load confusion appears in applied search."},
    {"query": "emotional labor at work", "ambiguity_type": "polysemy", "seeds": ["emotional labor", "emotional exhaustion", "burnout"], "keep_reason": "Emotional labor is commonly mixed with burnout indicators."},
    {"query": "job control and autonomy", "ambiguity_type": "scale_alias", "seeds": ["job control", "autonomy", "job characteristic theory"], "keep_reason": "Autonomy/control aliases map to adjacent constructs."},
    {"query": "safety climate questionnaire", "ambiguity_type": "method_vs_construct", "seeds": ["organizational safety", "workplace safety", "psychological safety"], "keep_reason": "Scale-focused but still construct-ambiguous phrasing."},
    {"query": "organizational culture versus climate", "ambiguity_type": "scope", "seeds": ["organizational culture", "team climate", "organizational effectiveness"], "keep_reason": "Culture vs climate distinction is a persistent ambiguity."},
    {"query": "diversity climate in organizations", "ambiguity_type": "polysemy", "seeds": ["diversity training", "team climate", "organizational justice"], "keep_reason": "Diversity-climate terms often overlap adjacent fairness constructs."},
    {"query": "well-being construct at work", "ambiguity_type": "scale_alias", "seeds": ["well-being", "subjective well-being", "occupational stress"], "keep_reason": "Well-being queries are common but construct boundaries vary."},
    {"query": "learning climate at work", "ambiguity_type": "scope", "seeds": ["team learning", "workplace learning", "organizational effectiveness"], "keep_reason": "Learning climate searches often omit level-of-analysis cues."},
]


def build_ambiguous_rows(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = [r["text_n"] for r in pool]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(docs)
    name_to_row = {r["name_n"]: r for r in pool}
    rows = []
    for i, spec in enumerate(AMBIGUOUS_V2_SPECS, start=1):
        query = spec["query"]
        picked = []
        for seed in spec["seeds"]:
            s = norm(seed)
            exact = name_to_row.get(s)
            if exact and exact["name"] not in picked:
                picked.append(exact["name"])
                continue
            for r in pool:
                if s in r["name_n"] or r["name_n"] in s:
                    if r["name"] not in picked:
                        picked.append(r["name"])
                    if len(picked) >= 8:
                        break
                if len(picked) >= 8:
                    break
        qv = vec.transform([norm(query)])
        scores = (qv @ mat.T).toarray()[0]
        order = np.argsort(-scores)
        for idx in order:
            if len(picked) >= 8:
                break
            cand = pool[int(idx)]["name"]
            if cand in picked:
                continue
            if contains_denylist(cand):
                continue
            picked.append(cand)
        picked = picked[:8]
        if len(picked) < 3:
            continue
        confidence = 0.9 if len(set(norm(s) for s in spec["seeds"])) >= 3 else 0.84
        rows.append(
            {
                "id": f"io_gold_v2_amb_{i:03d}",
                "stratum": "ambiguous",
                "query": query,
                "proposed_relevant_constructs": picked,
                "rationale": f"Ambiguous I/O query requiring disambiguation across closely related constructs: {', '.join(spec['seeds'][:3])}.",
                "io_relevance_confidence": round(confidence, 3),
                "ambiguity_type": spec["ambiguity_type"],
                "keep_reason": spec["keep_reason"],
            }
        )
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 ambiguous v2 rows, got {len(rows)}")
    return rows


def query_pattern(row: dict[str, Any]) -> str:
    q = norm(row["query"])
    s = row["stratum"]
    if s == "broad":
        if q.startswith("overall "):
            return "broad:overall {construct} in workplace settings"
        if q.startswith("evidence on "):
            return "broad:evidence on {construct} in io psychology"
        if q.startswith("organizational outcomes linked to "):
            return "broad:organizational outcomes linked to {construct}"
        return "broad:other"
    if s == "narrow":
        if q.startswith("specific construct definition for "):
            return "narrow:specific construct definition for {construct}"
        if q.startswith("how ") and " differs from related constructs" in q:
            return "narrow:how {construct} differs from related constructs"
        if q.startswith("nomological placement of "):
            return "narrow:nomological placement of {construct}"
        return "narrow:other"
    if s == "scale_focused":
        if q.startswith("validated scale or measurement for "):
            return "scale:validated scale or measurement for {construct}"
        if q.startswith("questionnaire and psychometric evidence for "):
            return "scale:questionnaire and psychometric evidence for {construct}"
        if q.startswith("reliability and validity indicators for "):
            return "scale:reliability and validity indicators for {construct}"
        return "scale:other"
    return f"ambiguous:{row.get('ambiguity_type','unknown')}"


def strict_validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 120:
        raise RuntimeError(f"Expected 120 rows, got {len(rows)}")
    by = Counter(r["stratum"] for r in rows)
    for stratum in ("broad", "narrow", "ambiguous", "scale_focused"):
        if by.get(stratum, 0) != 30:
            raise RuntimeError(f"Stratum {stratum} expected 30, got {by.get(stratum,0)}")

    for r in rows:
        if not (3 <= len(r["proposed_relevant_constructs"]) <= 8):
            raise RuntimeError(f"Invalid relevant count for query={r['query']}")
        if float(r["io_relevance_confidence"]) < 0.8:
            raise RuntimeError(f"Low confidence for query={r['query']}")
        if not r.get("ambiguity_type"):
            raise RuntimeError(f"Missing ambiguity_type for query={r['query']}")
        if not r.get("keep_reason"):
            raise RuntimeError(f"Missing keep_reason for query={r['query']}")

        term = contains_denylist(r["query"])
        if term:
            raise RuntimeError(f"Denylisted term '{term}' in query: {r['query']}")
        for t in r["proposed_relevant_constructs"]:
            term = contains_denylist(str(t))
            if term:
                raise RuntimeError(f"Denylisted term '{term}' in target: {t}")


def main() -> None:
    v1 = load_v1()
    pool = load_construct_pool()

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []

    # Keep non-ambiguous rows (already balanced at 30 each in v1 benchmark file).
    by_stratum = defaultdict(list)
    for r in v1:
        by_stratum[r["stratum"]].append(r)

    for stratum in ("broad", "narrow", "scale_focused"):
        for r in by_stratum[stratum][:30]:
            out = {
                "id": r["id"].replace("io_gold_v1_", "io_gold_v2_"),
                "stratum": stratum,
                "query": r["query"],
                "proposed_relevant_constructs": r["proposed_relevant_constructs"][:8],
                "rationale": r["rationale"],
                "io_relevance_confidence": r["io_relevance_confidence"],
                "ambiguity_type": "scope" if stratum != "scale_focused" else "scale_alias",
                "keep_reason": f"Retained from v1 {stratum} set; query is plausible and I/O-relevant.",
            }
            kept.append(out)

    # Remove all v1 ambiguous rows and replace with curated v2 ambiguous rows.
    for r in by_stratum["ambiguous"]:
        removed.append(
            {
                "query": r["query"],
                "stratum": "ambiguous",
                "removal_reason": "replaced_with_curated_v2_ambiguous_set",
            }
        )
    kept.extend(build_ambiguous_rows(pool))

    # Validate and write outputs
    strict_validate(kept)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with REVIEW.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "stratum",
                "ambiguity_type",
                "proposed_relevant_constructs",
                "rationale",
                "keep_reason",
                "reviewer_keep/edit",
            ],
        )
        w.writeheader()
        for r in kept:
            w.writerow(
                {
                    "query": r["query"],
                    "stratum": r["stratum"],
                    "ambiguity_type": r["ambiguity_type"],
                    "proposed_relevant_constructs": "; ".join(r["proposed_relevant_constructs"]),
                    "rationale": r["rationale"],
                    "keep_reason": r["keep_reason"],
                    "reviewer_keep/edit": "",
                }
            )

    with REMOVED.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "stratum", "removal_reason"])
        w.writeheader()
        for r in removed:
            w.writerow(r)

    q_patterns = Counter(query_pattern(r) for r in kept)
    target_counts = Counter()
    for r in kept:
        for t in r["proposed_relevant_constructs"]:
            target_counts[t] += 1

    print(f"wrote={OUT}")
    print(f"wrote={REVIEW}")
    print(f"wrote={REMOVED}")
    print("top_20_query_patterns")
    for k, v in q_patterns.most_common(20):
        print(f"{k}\t{v}")
    print("top_20_target_constructs")
    for k, v in target_counts.most_common(20):
        print(f"{k}\t{v}")
    print("final_row_count_per_stratum")
    for s in ("broad", "narrow", "ambiguous", "scale_focused"):
        print(f"{s}\t{sum(1 for r in kept if r['stratum']==s)}")


if __name__ == "__main__":
    main()
