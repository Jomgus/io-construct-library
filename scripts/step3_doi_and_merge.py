#!/usr/bin/env python3
"""
Step 3: DOI enrichment + final merge.

Outputs:
- data/processed/io_construct_library_master_database.csv

Workflow:
1) Load OpenAlex enriched constructs and fetch top 3 DOI URLs per construct
   (highest cited works, restricted to 8 whitelisted journals).
2) Load O*NET constructs from unified master and map each to ONET Online URL.
3) Standardize schemas and concatenate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple

import requests


JOURNAL_SOURCE_IDS = {
    "Journal of Applied Psychology": "https://openalex.org/S166002381",
    "Personnel Psychology": "https://openalex.org/S84664706",
    "Journal of Organizational Behavior": "https://openalex.org/S160573970",
    "Journal of Occupational and Organizational Psychology": "https://openalex.org/S87328381",
    "Academy of Management Journal": "https://openalex.org/S117778295",
    "Journal of Management": "https://openalex.org/S122767448",
    "Applied Psychology: An International Review": "https://openalex.org/S2898155583",
    "Journal of Vocational Behavior": "https://openalex.org/S25746158",
}

OPENALEX_API = "https://api.openalex.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3 DOI enrichment and final merge.")
    parser.add_argument("--openalex-input", default="data/processed/openalex_enriched.csv")
    parser.add_argument("--unified-input", default="data/raw/unified_master_constructs.csv")
    parser.add_argument("--output", default="data/processed/io_construct_library_master_database.csv")
    parser.add_argument("--cache-json", default="data/processed/openalex_doi_cache.json")
    parser.add_argument("--checkpoint-csv", default="data/processed/io_construct_library_master_database.checkpoint.csv")
    parser.add_argument("--mailto", default="jng6114@mavs.uta.edu")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-openalex-rows", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    return parser.parse_args()


def to_int(value: str) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        try:
            return int(float(str(value).strip()))
        except Exception:
            return 0


def normalize(value: str) -> str:
    return (value or "").strip().casefold()


def atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def openalex_get(
    endpoint: str,
    params: Dict[str, str],
    timeout: float,
    max_retries: int,
    backoff_seconds: float,
) -> Dict:
    url = f"{OPENALEX_API}/{endpoint}"
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                else:
                    wait = backoff_seconds * (2 ** min(attempt, 6))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            wait = backoff_seconds * (2 ** min(attempt, 6))
            time.sleep(wait)
            continue
    raise RuntimeError(f"OpenAlex rate-limited after retries: endpoint={endpoint} params={params}")


def resolve_concept_id(
    construct_name: str,
    mailto: str,
    timeout: float,
    max_retries: int,
    backoff_seconds: float,
) -> str:
    params = {
        "search": construct_name,
        "per-page": "25",
        "mailto": mailto,
    }
    payload = openalex_get("concepts", params, timeout, max_retries, backoff_seconds)
    results = payload.get("results", [])
    if not results:
        return ""

    target = normalize(construct_name)
    best = None
    best_score = -1
    target_tokens = set(target.split())
    for row in results:
        display = row.get("display_name", "")
        display_n = normalize(display)
        score = 0
        if display_n == target:
            score += 100
        tokens = set(display_n.split())
        if target_tokens and target_tokens.issubset(tokens):
            score += 25
        score += min(len(target_tokens & tokens) * 3, 21)
        score += min(int(row.get("works_count") or 0) // 5000, 10)
        if score > best_score:
            best_score = score
            best = row
    return (best or {}).get("id", "")


def top_3_doi_urls_for_concept(
    concept_id: str,
    mailto: str,
    timeout: float,
    max_retries: int,
    backoff_seconds: float,
) -> List[str]:
    if not concept_id:
        return []
    source_ids_joined = "|".join(JOURNAL_SOURCE_IDS.values())
    params = {
        "filter": f"concepts.id:{concept_id},primary_location.source.id:{source_ids_joined},has_doi:true",
        "sort": "cited_by_count:desc",
        "per-page": "50",
        "mailto": mailto,
    }
    payload = openalex_get("works", params, timeout, max_retries, backoff_seconds)
    results = payload.get("results", [])
    urls: List[str] = []
    seen = set()
    for work in results:
        doi = (work.get("doi") or "").strip()
        if not doi:
            continue
        doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        doi_url = doi_url.replace("http://doi.org/", "https://doi.org/")
        if doi_url in seen:
            continue
        seen.add(doi_url)
        urls.append(doi_url)
        if len(urls) == 3:
            break
    return urls


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def onet_reference_url(construct_name: str) -> str:
    query = urllib.parse.quote_plus(construct_name)
    return f"https://www.onetonline.org/find/result?s={query}"


def load_openalex_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_onet_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("Source") or "").strip() == "O*NET":
                rows.append(row)
    return rows[:614]


def standardize_openalex(row: Dict[str, str], reference_urls: List[str]) -> Dict[str, str]:
    return {
        "Construct_Name": (row.get("Construct_Name") or "").strip(),
        "Source": "OpenAlex",
        "Definition_Text": (row.get("Definition_Text") or "").strip(),
        "Paper_Count": str(to_int(row.get("Total_Paper_Count", "0"))),
        "Reference_URLs": ",".join(reference_urls),
    }


def standardize_onet(row: Dict[str, str]) -> Dict[str, str]:
    name = (row.get("Construct_Name") or "").strip()
    return {
        "Construct_Name": name,
        "Source": "O*NET",
        "Definition_Text": (row.get("Description") or "").strip(),
        "Paper_Count": "0",
        "Reference_URLs": onet_reference_url(name),
    }


def main() -> None:
    args = parse_args()
    openalex_input = Path(args.openalex_input)
    unified_input = Path(args.unified_input)
    output_path = Path(args.output)
    cache_path = Path(args.cache_json)
    checkpoint_path = Path(args.checkpoint_csv)

    if not openalex_input.exists():
        raise FileNotFoundError(f"Missing OpenAlex input: {openalex_input}")
    if not unified_input.exists():
        raise FileNotFoundError(f"Missing unified input: {unified_input}")

    doi_cache = load_json(cache_path)
    openalex_rows = load_openalex_rows(openalex_input)
    if args.max_openalex_rows > 0:
        openalex_rows = openalex_rows[: args.max_openalex_rows]
    onet_rows = load_onet_rows(unified_input)

    out_rows: List[Dict[str, str]] = []
    cache_hits = 0
    api_calls = 0

    for idx, row in enumerate(openalex_rows, start=1):
        name = (row.get("Construct_Name") or "").strip()
        key = normalize(name)

        if key in doi_cache:
            payload = doi_cache[key]
            doi_urls = payload.get("doi_urls", [])
            cache_hits += 1
        else:
            concept_id = ""
            doi_urls: List[str] = []
            error_msg = ""
            try:
                concept_id = resolve_concept_id(
                    name,
                    args.mailto,
                    args.timeout,
                    args.max_retries,
                    args.backoff_seconds,
                )
                doi_urls = top_3_doi_urls_for_concept(
                    concept_id,
                    args.mailto,
                    args.timeout,
                    args.max_retries,
                    args.backoff_seconds,
                )
            except Exception as exc:
                # Fail-open: keep construct in dataset with empty DOI list.
                error_msg = str(exc)
            doi_cache[key] = {
                "Construct_Name": name,
                "concept_id": concept_id,
                "doi_urls": doi_urls,
                "error": error_msg,
            }
            api_calls += 1
            time.sleep(max(0.0, args.sleep_seconds))

        out_rows.append(standardize_openalex(row, doi_urls))

        if idx % max(1, args.checkpoint_every) == 0:
            interim = out_rows + [standardize_onet(r) for r in onet_rows]
            write_csv(
                checkpoint_path,
                interim,
                ["Construct_Name", "Source", "Definition_Text", "Paper_Count", "Reference_URLs"],
            )
            atomic_write_json(cache_path, doi_cache)
            print(
                f"[checkpoint] openalex={idx}/{len(openalex_rows)} api_calls={api_calls} cache_hits={cache_hits}"
            )

    out_rows.extend(standardize_onet(r) for r in onet_rows)
    write_csv(
        output_path,
        out_rows,
        ["Construct_Name", "Source", "Definition_Text", "Paper_Count", "Reference_URLs"],
    )
    write_csv(
        checkpoint_path,
        out_rows,
        ["Construct_Name", "Source", "Definition_Text", "Paper_Count", "Reference_URLs"],
    )
    atomic_write_json(cache_path, doi_cache)

    print(f"OpenAlex rows processed: {len(openalex_rows)}")
    print(f"O*NET rows merged: {len(onet_rows)}")
    print(f"API calls: {api_calls}")
    print(f"Cache hits: {cache_hits}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
