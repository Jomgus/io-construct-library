"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

type SourceType = "OpenAlex" | "O*NET";

type SearchResult = {
  constructName: string;
  source: SourceType;
  definitionText: string;
  paperCount: number;
  referenceUrls: string[];
  noDoiYet: boolean;
  related: string[];
};

const faqItems = [
  {
    question: "What is this site for?",
    answer:
      "It is a search tool for I/O psychology constructs. Enter a term and the site returns matching constructs, linked sources, and nearby terms.",
  },
  {
    question: "What sources are included?",
    answer:
      "The library uses journal-derived records from OpenAlex and occupational content from O*NET. The source filter lets you limit results to papers or O*NET records.",
  },
  {
    question: "How are results ranked?",
    answer:
      "Search combines keyword matching and semantic matching. A second reranking pass can be applied to improve ordering on harder queries.",
  },
  {
    question: "What does paper count mean?",
    answer:
      "Paper count is the number of linked OpenAlex records tied to a construct in the current library build. It shows coverage, not quality.",
  },
  {
    question: "What are related constructs?",
    answer:
      "Related constructs are nearby concepts surfaced by the library’s similarity structure. Use them to find adjacent terms, overlap, and alternate phrasing.",
  },
];

const citations = [
  { label: "OpenAlex", href: "https://openalex.org/" },
  { label: "O*NET", href: "https://www.onetonline.org/" },
  { label: "BEIR (Thakur et al., 2021)", href: "https://arxiv.org/abs/2104.08663" },
  { label: "SPLADE v2 (Formal et al., 2021)", href: "https://arxiv.org/abs/2109.10086" },
  { label: "ColBERTv2 (Santhanam et al., 2022)", href: "https://arxiv.org/abs/2112.01488" },
  { label: "Passage Re-ranking with BERT (Nogueira and Cho, 2019)", href: "https://arxiv.org/abs/1901.04085" },
];

export default function Home() {
  const sourceOptions: Array<"all" | SourceType> = ["all", "OpenAlex", "O*NET"];
  const thresholdOptions = [0, 5, 25, 100, 300];
  const sourceLabels: Record<"all" | SourceType, string> = {
    all: "All",
    OpenAlex: "Papers",
    "O*NET": "O*NET",
  };

  const [query, setQuery] = useState("");
  const [source, setSource] = useState<"all" | SourceType>("all");
  const [minPaperCount, setMinPaperCount] = useState(25);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [openFaq, setOpenFaq] = useState<number>(0);
  const [debouncedQuery, setDebouncedQuery] = useState(query);

  const resultsRef = useRef<HTMLElement | null>(null);
  const resultCount = useMemo(() => results.length, [results]);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 250);
    return () => clearTimeout(timer);
  }, [query]);

  async function runSearch(nextQuery: string, nextSource: "all" | SourceType, nextMinPaperCount: number) {
    setLoading(true);
    setError("");

    const params = new URLSearchParams({
      q: nextQuery,
      source: nextSource,
      minPaperCount: String(nextMinPaperCount),
      limit: "15",
    });

    try {
      const response = await fetch(`/api/search?${params.toString()}`);
      if (!response.ok) throw new Error("Search request failed.");
      const payload = (await response.json()) as { results: SearchResult[] };
      setResults(payload.results || []);
      setExpanded({});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([]);
      setError("");
      setLoading(false);
      return;
    }
    runSearch(debouncedQuery, source, minPaperCount);
  }, [debouncedQuery, source, minPaperCount]);

  return (
    <main className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <section
        className={`px-6 py-8 sm:px-10 lg:px-14 ${
          debouncedQuery.trim() ? "pb-4" : "min-h-screen"
        }`}
      >
        <div className="border-b border-[var(--border)] pb-4 text-[11px] uppercase tracking-[0.24em] text-[var(--muted)]">
          IO Construct Library
        </div>

        <div
          className={`mx-auto flex w-full max-w-5xl flex-col justify-center ${
            debouncedQuery.trim() ? "pt-16 sm:pt-24" : "pt-16 sm:pt-20"
          }`}
        >
          <div className="mb-8">
            <h1 className="font-display text-5xl tracking-tight text-[var(--foreground)] sm:text-6xl">
              Construct Search
            </h1>
            <p className="mt-3 text-sm text-[var(--muted)]">
              Search I/O constructs, source records, and related terms.
            </p>
          </div>

          <section
            className={`grid gap-3 border-[var(--border)] py-5 sm:grid-cols-12 ${
              debouncedQuery.trim() ? "border-t" : "border-y"
            }`}
          >
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search construct, scale, or term..."
              className="sm:col-span-5 bg-transparent px-0 py-3 text-lg outline-none placeholder:text-[#9b8f7f]"
            />
            <div className="sm:col-span-3">
              <p className="mb-2 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">Source</p>
              <div className="flex flex-wrap items-center gap-2">
              {sourceOptions.map((option) => {
                const isActive = source === option;
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => setSource(option)}
                    className={`rounded-full border px-4 py-2 text-sm transition ${
                      isActive
                        ? "border-[var(--foreground)] bg-[var(--foreground)] text-[var(--surface)]"
                        : "border-[var(--border)] text-[var(--foreground)] hover:border-[var(--foreground)]"
                    }`}
                  >
                    {sourceLabels[option]}
                  </button>
                );
              })}
              </div>
            </div>
            <div className="sm:col-span-4">
              <p className="mb-2 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">Paper count</p>
              <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap">
                {thresholdOptions.map((value) => {
                  const isActive = minPaperCount === value;
                  return (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setMinPaperCount(value)}
                      className={`rounded-full border px-3 py-2 text-sm transition ${
                        isActive
                          ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--surface)]"
                          : "border-[var(--border)] text-[var(--foreground)] hover:border-[var(--accent)]"
                      }`}
                      title="OpenAlex paper count threshold"
                    >
                      {value === 0 ? "Any" : `${value}+`}
                    </button>
                  );
                })}
              </div>
            </div>
          </section>

          {error ? <p className="mt-6 text-sm text-red-700">{error}</p> : null}

          {!debouncedQuery.trim() ? (
            <p className="mt-8 text-center text-[var(--muted)]">
              Enter a term like <span className="font-semibold text-[var(--accent)]">Job Satisfaction</span> or{" "}
              <span className="font-semibold text-[var(--accent)]">Burnout</span> to explore the network.
            </p>
          ) : null}
        </div>
      </section>

      <section ref={resultsRef} className="px-6 pb-20 sm:px-10 lg:px-14">
        <div
          className={`mx-auto max-w-5xl border-t border-[var(--border)] ${
            debouncedQuery.trim() ? "pt-6" : "pt-10"
          }`}
        >
          <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
            {debouncedQuery.trim() ? (
              <div className="flex w-full flex-wrap items-center justify-between gap-4 text-sm">
                <p className="text-[var(--muted)]">
                  {loading ? "Searching..." : `${resultCount} result${resultCount === 1 ? "" : "s"}`}
                </p>
                <Link
                  href={`/map?q=${encodeURIComponent(debouncedQuery.trim())}`}
                  className="text-[11px] uppercase tracking-[0.24em] text-[var(--accent)] underline underline-offset-4"
                >
                  Map Visual
                </Link>
              </div>
            ) : null}
          </div>

          <section className="grid gap-8">
            {results.map((item, index) => {
              const key = `${item.source}-${item.constructName}-${index}`;
              const hasExtras = item.referenceUrls.length > 1;
              const primary = item.referenceUrls[0] || "";
              const extraRefs = item.referenceUrls.slice(1);
              const isExpanded = expanded[key] || false;

              return (
                <article key={key} className="border-t border-[var(--border)] py-7">
                  <div className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
                    <div>
                      <div className="flex flex-wrap items-center gap-3">
                        <h3 className="text-3xl tracking-tight text-[var(--foreground)]">{item.constructName}</h3>
                        <span className="rounded-full border border-[var(--border)] px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-[var(--muted)]">
                          {item.source}
                        </span>
                        <span className="rounded-full border border-[var(--accent-soft)] bg-[var(--accent-soft)] px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-[var(--accent)]">
                          {item.paperCount} papers
                        </span>
                      </div>

                      <p className="mt-4 max-w-2xl text-base leading-8 text-[var(--muted)]">
                        {item.definitionText}
                      </p>
                    </div>

                    <div className="space-y-5 border-t border-[var(--border)] pt-5 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
                      {item.related && item.related.length > 0 ? (
                        <div>
                          <p className="text-[11px] uppercase tracking-[0.2em] text-[var(--muted)]">Related constructs</p>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {item.related.map((name) => (
                              <button
                                key={name}
                                type="button"
                                onClick={() => setQuery(name)}
                                className="rounded-full border border-[var(--border)] px-3 py-1.5 text-sm text-[var(--foreground)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]"
                              >
                                {name}
                              </button>
                            ))}
                          </div>
                        </div>
                      ) : null}

                      <div className="space-y-3 text-sm">
                        {primary ? (
                          <div className="space-y-1">
                            <p className="text-[11px] uppercase tracking-[0.2em] text-[var(--muted)]">
                              {item.source === "O*NET" ? "Record" : "Primary DOI"}
                            </p>
                            <a
                              href={primary}
                              target="_blank"
                              rel="noreferrer"
                              className="block break-all text-[var(--accent)] underline underline-offset-4"
                            >
                              {primary}
                            </a>
                          </div>
                        ) : null}

                        {item.noDoiYet ? <span className="text-[var(--muted)]">No DOI available</span> : null}

                        {hasExtras ? (
                          <button
                            type="button"
                            onClick={() => setExpanded((prev) => ({ ...prev, [key]: !prev[key] }))}
                            className="text-[var(--muted)] underline decoration-dotted underline-offset-4"
                          >
                            {isExpanded ? "Hide extra sources" : `Show ${extraRefs.length} more source${extraRefs.length === 1 ? "" : "s"}`}
                          </button>
                        ) : null}
                      </div>

                      {isExpanded && extraRefs.length > 0 ? (
                        <ul className="space-y-2 text-sm text-[var(--muted)]">
                          {extraRefs.map((url) => (
                            <li key={url}>
                              <a
                                href={url}
                                target="_blank"
                                rel="noreferrer"
                                className="break-all underline underline-offset-2"
                              >
                                {url}
                              </a>
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  </div>
                </article>
              );
            })}
          </section>
        </div>
      </section>

      <section className="px-6 pb-20 sm:px-10 lg:px-14">
        <div className="mx-auto max-w-5xl border-t border-[var(--border)] pt-10">
          <div className="grid gap-14">
            <div className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
              <div>
                <p className="text-[11px] uppercase tracking-[0.24em] text-[var(--accent)]">About</p>
                <h2 className="font-display mt-3 text-4xl tracking-tight">What this site does</h2>
              </div>
              <div className="space-y-6 text-[15px] leading-8 text-[var(--muted)]">
                <p>
                  The Construct Library is a search tool for I/O psychology constructs. It takes a term
                  or phrase and returns construct definitions, source links, and nearby concepts.
                </p>
                <p>
                  The library combines journal-derived construct records with occupational content from
                  O*NET, so the same search can cover research language and workplace terms.
                </p>
                <p>
                  Results can be filtered by source and paper count, then opened through related
                  constructs and linked references.
                </p>
              </div>
            </div>

            <div className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
              <div>
                <p className="text-[11px] uppercase tracking-[0.24em] text-[var(--accent)]">FAQ</p>
                <h2 className="font-display mt-3 text-4xl tracking-tight">Common questions</h2>
              </div>
              <div className="space-y-4">
                {faqItems.map((item, index) => {
                  const isOpen = openFaq === index;
                  return (
                    <article
                      key={item.question}
                      className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] px-5 py-5"
                    >
                      <button
                        type="button"
                        onClick={() => setOpenFaq(isOpen ? -1 : index)}
                        className="flex w-full items-start justify-between gap-6 text-left"
                        aria-expanded={isOpen}
                      >
                        <span className="pr-4 text-[1.35rem] leading-8 tracking-tight text-[var(--foreground)]">
                          {item.question}
                        </span>
                        <span className="pt-1 text-2xl leading-none text-[var(--muted)]">
                          {isOpen ? "−" : "+"}
                        </span>
                      </button>
                      {isOpen ? (
                        <p className="mt-5 max-w-4xl text-[15px] leading-7 text-[var(--muted)]">
                          {item.answer}
                        </p>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            </div>

            <div className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
              <div>
                <p className="text-[11px] uppercase tracking-[0.24em] text-[var(--accent)]">Citations</p>
                <h2 className="font-display mt-3 text-4xl tracking-tight">Sources and methods</h2>
              </div>
              <div className="grid gap-3 text-[15px] leading-7 text-[var(--muted)]">
                {citations.map((item) => (
                  <a
                    key={item.label}
                    href={item.href}
                    target="_blank"
                    rel="noreferrer"
                    className="border-b border-[var(--border)] pb-3 text-[var(--foreground)] transition hover:text-[var(--accent)]"
                  >
                    {item.label}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
