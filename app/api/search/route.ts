import { NextRequest, NextResponse } from "next/server";
import {
  passthroughTopK,
  type RerankMode,
  type SearchCandidate,
} from "@/lib/rerank";

type SearchIntent = "broad" | "narrow" | "ambiguous" | "scale_focused";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function getRerankMode(): RerankMode {
  const mode = (process.env.SEARCH_RERANK_MODE || "off").toLowerCase();
  return mode === "cross_encoder_small" ? "cross_encoder_small" : "off";
}

function getRerankTimeoutMs(): number {
  const raw = Number(process.env.SEARCH_RERANK_TIMEOUT_MS || "1200");
  return Number.isFinite(raw) ? clamp(raw, 150, 5000) : 1200;
}

function detectIntent(query: string): SearchIntent {
  const q = query.toLowerCase();
  if (
    q.includes("employee voice and silence")
    || q.includes("role ambiguity at work")
    || q.includes("emotional labor in service work")
  ) {
    return "ambiguous";
  }
  if (
    /\b(scale|questionnaire|psychometric|reliability|validity|measurement|measure|factor analysis|cronbach|invariance)\b/.test(
      q
    )
  ) {
    return "scale_focused";
  }
  if (
    /\b(how .* differs from related constructs|difference between|distinct from| vs )\b/.test(
      q
    )
  ) {
    return "narrow";
  }
  if (
    /\b(overall |evidence on |organizational outcomes linked to|in workplace settings)\b/.test(
      q
    )
  ) {
    return "broad";
  }
  if (/\bconstruct in work context|in work context\b/.test(q)) {
    return "ambiguous";
  }
  const tokenCount = q.split(/\s+/).filter(Boolean).length;
  if (tokenCount <= 3) return "ambiguous";
  return "broad";
}

const AMBIGUOUS_ALIAS_MAP: Record<string, string[]> = {
  "job attitude": ["job satisfaction", "organizational commitment"],
  "job satisfaction": ["job attitude", "work satisfaction"],
  "organizational commitment": [
    "affective commitment",
    "continuance commitment",
    "normative commitment",
  ],
  cwb: ["counterproductive work behavior", "workplace deviance"],
  ocb: ["organizational citizenship behavior", "citizenship behavior"],
  "work-family conflict": ["work family conflict", "family-work conflict"],
  burnout: ["occupational burnout", "emotional exhaustion", "depersonalization"],
  justice: ["organizational justice", "procedural justice", "distributive justice"],
  communication: ["team communication", "communication climate"],
  group: ["group cohesion", "team dynamics"],
  cultural: ["organizational culture", "cultural values"],
};

const VOICE_SILENCE_FORCE_ALIASES: string[] = [
  "Psychological safety",
  "Organizational justice",
  "Employee engagement",
  "Employee morale",
  "Human Resources",
  "Decentralization and Employee Empowerment",
  "Openness and Honesty",
  "Active Listening",
  "Speaking",
];

function getFeatureBoostTerms(query: string, intent: SearchIntent): string[] {
  if (intent !== "ambiguous") return [];
  const q = query.toLowerCase();
  if (q.includes("employee voice and silence") || q.includes("voice and silence")) {
    return [
      "employee voice",
      "voice behavior",
      "employee silence",
      "silence behavior",
      "organizational silence",
      "speaking up",
      "prohibitive voice",
      "promotive voice",
    ];
  }
  if (q.includes("role ambiguity at work")) {
    return ["role ambiguity", "role conflict", "role clarity"];
  }
  if (q.includes("emotional labor in service work")) {
    return ["emotional labor", "surface acting", "deep acting", "emotional exhaustion"];
  }
  return [];
}

function getForcedConstructs(query: string): string[] {
  const q = query.toLowerCase();
  const isVoiceSilence =
    q.includes("employee voice")
    || q.includes("employee silence")
    || q.includes("organizational silence")
    || q.includes("voice behavior");
  if (!isVoiceSilence) return [];
  return [
    "Employee voice",
    "Employee silence",
    "Organizational silence",
    "Voice behavior",
  ];
}

function getForcedConstructAliasTargets(query: string): string[] {
  if (getForcedConstructs(query).length === 0) return [];
  return VOICE_SILENCE_FORCE_ALIASES;
}

function applyForcedConstructBoost(
  candidates: SearchCandidate[],
  forcedConstructNames: string[]
): SearchCandidate[] {
  if (candidates.length === 0 || forcedConstructNames.length === 0) return candidates;
  const forced = new Set(forcedConstructNames.map((name) => name.toLowerCase()));
  const withIndex = candidates.map((candidate, idx) => {
    const isForced = forced.has((candidate.constructName || "").toLowerCase());
    const base =
      Number(candidate.rrfScore ?? candidate.similarityScore ?? 0) + (candidate.exactMatch ? 1 : 0);
    return {
      candidate,
      idx,
      boostedScore: base + (isForced ? 0.1 : 0),
      isForced,
    };
  });
  withIndex.sort((a, b) => {
    if (b.boostedScore !== a.boostedScore) return b.boostedScore - a.boostedScore;
    if ((b.isForced ? 1 : 0) !== (a.isForced ? 1 : 0)) return (b.isForced ? 1 : 0) - (a.isForced ? 1 : 0);
    return a.idx - b.idx;
  });
  return withIndex.map((item) => item.candidate);
}

function rewriteAmbiguousQuery(query: string, intent: SearchIntent): string {
  if (intent !== "ambiguous") return query;
  const q = query.toLowerCase();
  const additions = new Set<string>();
  if (q.includes("employee voice and silence")) {
    additions.add("voice behavior");
    additions.add("employee silence");
    additions.add("organizational silence");
    additions.add("speaking up");
  }
  if (q.includes("role ambiguity at work")) {
    additions.add("role ambiguity");
    additions.add("role conflict");
    additions.add("role clarity");
    additions.add("job role ambiguity");
  }
  if (q.includes("emotional labor in service work")) {
    additions.add("emotional labor");
    additions.add("surface acting");
    additions.add("deep acting");
    additions.add("emotional exhaustion");
    additions.add("service interactions");
  }
  for (const [key, aliases] of Object.entries(AMBIGUOUS_ALIAS_MAP)) {
    if (q.includes(key)) {
      for (const alias of aliases) additions.add(alias);
    }
  }
  const specific = q.match(/^([a-z][a-z\s-]+?)\s+construct in work context$/);
  if (specific) {
    const term = specific[1].trim();
    additions.add(`${term} at work`);
    if (AMBIGUOUS_ALIAS_MAP[term]) {
      for (const alias of AMBIGUOUS_ALIAS_MAP[term]) additions.add(alias);
    }
  }
  if (additions.size === 0) return query;
  return `${query} ${Array.from(additions).slice(0, 6).join(" ")}`.trim();
}

function getIntentDepthDefault(intent: SearchIntent): number {
  if (intent === "ambiguous") return 60;
  if (intent === "broad") return 40;
  if (intent === "narrow") return 30;
  return 30;
}

function getIntentDepthEnv(intent: SearchIntent): number | null {
  const raw =
    intent === "ambiguous"
      ? process.env.SEARCH_CANDIDATE_DEPTH_AMBIGUOUS
      : intent === "broad"
      ? process.env.SEARCH_CANDIDATE_DEPTH_BROAD
      : intent === "narrow"
      ? process.env.SEARCH_CANDIDATE_DEPTH_NARROW
      : process.env.SEARCH_CANDIDATE_DEPTH_SCALE;
  const n = Number(raw || "");
  return Number.isFinite(n) ? Math.round(n) : null;
}

function getCandidateDepth(topK: number, intent: SearchIntent): number {
  const base = getIntentDepthEnv(intent) ?? getIntentDepthDefault(intent);
  return clamp(base, topK, 80);
}

function getSemanticBlend(intent: SearchIntent): number {
  if (intent !== "ambiguous") return 0;
  const raw = Number(process.env.SEARCH_AMBIGUOUS_SEMANTIC_BLEND || "0.35");
  return Number.isFinite(raw) ? clamp(raw, 0, 1) : 0.35;
}

function getFeatureBoostWeight(intent: SearchIntent): number {
  if (intent !== "ambiguous") return 0;
  const raw = Number(process.env.SEARCH_AMBIGUOUS_FEATURE_BOOST_WEIGHT || "0.25");
  return Number.isFinite(raw) ? clamp(raw, 0, 1) : 0.25;
}

function getAppOrigin(req: NextRequest): string {
  const fromEnv = (process.env.APP_BASE_URL || "").trim();
  if (fromEnv) return fromEnv.replace(/\/+$/, "");
  const host = req.headers.get("x-forwarded-host") || req.headers.get("host") || req.nextUrl.host;
  const proto = req.headers.get("x-forwarded-proto") || req.nextUrl.protocol.replace(":", "") || "https";
  return `${proto}://${host}`;
}

function getInternalAuthHeaders(req: NextRequest): Record<string, string> {
  const headers: Record<string, string> = {};
  const cookie = req.headers.get("cookie");
  const bypass = req.headers.get("x-vercel-protection-bypass");
  if (cookie) headers.cookie = cookie;
  if (bypass) headers["x-vercel-protection-bypass"] = bypass;
  return headers;
}

export async function GET(req: NextRequest) {
  const url = req.nextUrl;
  const query = (url.searchParams.get("q") || "").trim();
  const source = (url.searchParams.get("source") || "all").trim();
  const minPaperCount = Number(url.searchParams.get("minPaperCount") || "0");
  const requestedLimit = Number(url.searchParams.get("limit") || "15");
  const topK = clamp(Number.isFinite(requestedLimit) ? requestedLimit : 15, 1, 100);
  const detectedIntent = detectIntent(query);
  const rewrittenQuery = rewriteAmbiguousQuery(query, detectedIntent);
  const rerankMode = getRerankMode();
  const candidateDepthRequested = getCandidateDepth(topK, detectedIntent);
  const semanticBlend = getSemanticBlend(detectedIntent);
  const featureBoostTerms = getFeatureBoostTerms(query, detectedIntent);
  const featureBoostWeight = getFeatureBoostWeight(detectedIntent);
  const forceConstructs = getForcedConstructs(query);
  const forceConstructAliases = getForcedConstructAliasTargets(query);
  const forceConstructsForWorker = Array.from(new Set([...forceConstructs, ...forceConstructAliases]));
  const workerLimit = candidateDepthRequested;

  if (!query) {
    return NextResponse.json({
      query,
      detectedIntent,
      rewrittenQuery,
      rerankMode,
      count: 0,
      results: [],
    });
  }

  try {
    const workerParams = new URLSearchParams({
      q: rewrittenQuery,
      source,
      minPaperCount: String(Number.isFinite(minPaperCount) ? minPaperCount : 0),
      limit: String(workerLimit),
    });
    for (const name of forceConstructsForWorker) {
      workerParams.append("forceConstruct", name);
    }
    const workerUrl = `https://io-construct-search-worker.jomus.workers.dev?${workerParams.toString()}`;
    const response = await fetch(workerUrl);

    if (!response.ok) {
      throw new Error(`Cloudflare Worker returned ${response.status}`);
    }

    const data = (await response.json()) as {
      results?: SearchCandidate[];
      rankingMode?: string;
    };
    const candidates = Array.isArray(data.results) ? data.results : [];
    const boostedCandidates = applyForcedConstructBoost(candidates, forceConstructsForWorker);
    let results = passthroughTopK(boostedCandidates, topK);
    let rerankApplied = false;
    let rerankErrorMessage: string | null = null;

    if (rerankMode !== "off" && candidates.length > 0) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), getRerankTimeoutMs());
      try {
        const rerankUrl = `${getAppOrigin(req)}/api/rerank`;
        const rerankResp = await fetch(rerankUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            ...getInternalAuthHeaders(req),
          },
          body: JSON.stringify({
            query: rewrittenQuery,
            candidates: boostedCandidates,
            topK,
            mode: rerankMode,
            semanticBlend,
            featureBoostTerms,
            featureBoostWeight,
          }),
          signal: controller.signal,
        });

        if (!rerankResp.ok) {
          throw new Error(`Rerank endpoint returned ${rerankResp.status}`);
        }

        const rerankJson = (await rerankResp.json()) as { results?: SearchCandidate[] };
        const reranked = Array.isArray(rerankJson.results) ? rerankJson.results : [];
        if (reranked.length > 0) {
          results = passthroughTopK(
            applyForcedConstructBoost(reranked, forceConstructsForWorker),
            topK
          );
          rerankApplied = true;
        }
      } catch (err) {
        // Fail-open: preserve worker ranking if rerank path fails.
        console.error("Rerank fallback engaged:", err);
        rerankErrorMessage = err instanceof Error ? err.message : "Unknown rerank error";
      } finally {
        clearTimeout(timeout);
      }
    }

    return NextResponse.json({
      query,
      detectedIntent,
      rewrittenQuery,
      candidateDepthRequested,
      semanticBlend,
      featureBoostTerms,
      featureBoostWeight,
      forceConstructs,
      forceConstructAliases,
      rankingMode: data.rankingMode || "unknown",
      rerankMode,
      rerankApplied,
      rerankError: rerankErrorMessage,
      candidateCount: candidates.length,
      count: results.length,
      results,
    });
  } catch (error: unknown) {
    const details = error instanceof Error ? error.message : "Unknown error";
    console.error("Search API Error:", error);
    return NextResponse.json(
      {
        error: "Search failed",
        details,
      },
      { status: 500 }
    );
  }
}
