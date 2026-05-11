import { NextRequest, NextResponse } from "next/server";
import {
  passthroughTopK,
  rerankCandidatesByScores,
  type RerankMode,
  type SearchCandidate,
} from "@/lib/rerank";

type RerankRequest = {
  query?: string;
  candidates?: SearchCandidate[];
  topK?: number;
  mode?: RerankMode;
  semanticBlend?: number;
  featureBoostTerms?: string[];
  featureBoostWeight?: number;
};

function toMode(value: string | undefined): RerankMode {
  return value === "cross_encoder_small" ? "cross_encoder_small" : "off";
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function getTimeoutMs(): number {
  const raw = Number(process.env.CROSS_ENCODER_TIMEOUT_MS || "2500");
  return Number.isFinite(raw) ? clamp(raw, 200, 10000) : 2500;
}

function isHuggingFaceEndpoint(endpoint: string): boolean {
  return endpoint.includes("api-inference.huggingface.co");
}

function isCloudflareEndpoint(endpoint: string): boolean {
  return endpoint.includes("api.cloudflare.com");
}

type CrossEncoderResponse =
  | { scores?: number[] }
  | { results?: Array<{ score?: number; index?: number }> }
  | { result?: { response?: Array<{ id?: number; score?: number }> } }
  | Array<{ score?: number; index?: number }>
  | Array<Array<{ score?: number; label?: string }>>;

function extractScores(
  response: CrossEncoderResponse,
  expectedLength: number
): number[] {
  if (Array.isArray(response)) {
    // Shape A: [{score, index?}, ...]
    if (response.length > 0 && !Array.isArray(response[0])) {
      const rows = response as Array<{ score?: number; index?: number }>;
      const scores = new Array(expectedLength).fill(Number.NEGATIVE_INFINITY);
      let assigned = 0;
      for (let i = 0; i < rows.length; i += 1) {
        const row = rows[i];
        const idx = Number.isInteger(row.index) ? (row.index as number) : i;
        if (idx < 0 || idx >= expectedLength) continue;
        scores[idx] = typeof row.score === "number" ? row.score : Number.NEGATIVE_INFINITY;
        assigned += 1;
      }
      if (assigned > 0) return scores;
    }
    // Shape B (HF pipeline for list inputs): [[{label, score}], ...]
    if (response.length > 0 && Array.isArray(response[0])) {
      const rows = response as Array<Array<{ score?: number; label?: string }>>;
      const scores = new Array(expectedLength).fill(Number.NEGATIVE_INFINITY);
      let assigned = 0;
      for (let i = 0; i < rows.length && i < expectedLength; i += 1) {
        const first = rows[i]?.[0];
        if (!first || typeof first.score !== "number") continue;
        scores[i] = first.score;
        assigned += 1;
      }
      if (assigned > 0) return scores;
    }
  }

  if (Array.isArray((response as { scores?: number[] }).scores)) {
    const scores = (response as { scores: number[] }).scores;
    if (scores.length === expectedLength) return scores;
  }

  const results = (response as { results?: Array<{ score?: number; index?: number }> }).results;
  if (!Array.isArray(results)) {
    const cf = (response as { result?: { response?: Array<{ id?: number; score?: number }> } })
      .result?.response;
    if (Array.isArray(cf)) {
      const scores = new Array(expectedLength).fill(Number.NEGATIVE_INFINITY);
      let assigned = 0;
      for (let i = 0; i < cf.length; i += 1) {
        const row = cf[i];
        const idx = Number.isInteger(row.id) ? (row.id as number) : i;
        if (idx < 0 || idx >= expectedLength) continue;
        const score = typeof row.score === "number" ? row.score : Number.NEGATIVE_INFINITY;
        scores[idx] = score;
        assigned += 1;
      }
      if (assigned > 0) return scores;
    }
    throw new Error("Inference response missing scores/results.");
  }

  const scores = new Array(expectedLength).fill(Number.NEGATIVE_INFINITY);
  let assigned = 0;
  for (let i = 0; i < results.length; i += 1) {
    const row = results[i];
    const idx = Number.isInteger(row.index) ? (row.index as number) : i;
    if (idx < 0 || idx >= expectedLength) continue;
    const score = typeof row.score === "number" ? row.score : Number.NEGATIVE_INFINITY;
    scores[idx] = score;
    assigned += 1;
  }
  if (assigned === 0) {
    throw new Error("Inference response had no usable scores.");
  }
  return scores;
}

function normalizeScores(values: number[]): number[] {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return new Array(values.length).fill(0);
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  if (hi - lo < 1e-12) return values.map(() => 1);
  return values.map((v) => (Number.isFinite(v) ? (v - lo) / (hi - lo) : 0));
}

function blendScores(
  ceScores: number[],
  candidates: SearchCandidate[],
  semanticBlend: number
): number[] {
  const alpha = clamp(semanticBlend, 0, 1);
  if (alpha <= 0) return ceScores;
  const ceNorm = normalizeScores(ceScores);
  const semRaw = candidates.map((c) =>
    typeof c.similarityScore === "number" ? c.similarityScore : 0
  );
  const semNorm = normalizeScores(semRaw);
  return ceNorm.map((v, i) => (1 - alpha) * v + alpha * semNorm[i]);
}

function applyFeatureBoost(
  baseScores: number[],
  candidates: SearchCandidate[],
  terms: string[],
  weight: number
): number[] {
  const w = clamp(weight, 0, 1);
  if (w <= 0 || terms.length === 0) return baseScores;
  const lowered = terms.map((t) => t.toLowerCase());
  const boosts = candidates.map((c) => {
    const text = `${c.constructName} ${c.definitionText}`.toLowerCase();
    let matches = 0;
    for (const term of lowered) {
      if (term && text.includes(term)) matches += 1;
    }
    return matches;
  });
  const maxMatch = Math.max(...boosts, 0);
  if (maxMatch <= 0) return baseScores;
  return baseScores.map((s, i) => s + w * (boosts[i] / maxMatch));
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as RerankRequest;
    const query = (body.query || "").trim();
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const topK =
      Number.isFinite(body.topK) && typeof body.topK === "number" ? body.topK : 15;
    const mode = toMode(body.mode);
    const semanticBlend =
      Number.isFinite(body.semanticBlend) && typeof body.semanticBlend === "number"
        ? clamp(body.semanticBlend, 0, 1)
        : 0;
    const featureBoostTerms = Array.isArray(body.featureBoostTerms)
      ? body.featureBoostTerms.map((t) => String(t || "").trim()).filter(Boolean).slice(0, 16)
      : [];
    const featureBoostWeight =
      Number.isFinite(body.featureBoostWeight) && typeof body.featureBoostWeight === "number"
        ? clamp(body.featureBoostWeight, 0, 1)
        : 0;

    if (!query) {
      return NextResponse.json(
        { error: "Missing query", details: "Provide non-empty query." },
        { status: 400 }
      );
    }
    if (candidates.length === 0) {
      return NextResponse.json({
        query,
        mode,
        count: 0,
        results: [],
      });
    }

    if (mode === "off") {
      return NextResponse.json({
        query,
        mode,
        provider: "none",
        count: Math.min(topK, candidates.length),
        results: passthroughTopK(candidates, topK),
      });
    }

    const endpoint = (process.env.CROSS_ENCODER_ENDPOINT || "").trim();
    if (!endpoint) {
      return NextResponse.json(
        {
          error: "Rerank service unavailable",
          details: "Set CROSS_ENCODER_ENDPOINT for cross_encoder_small mode.",
        },
        { status: 503 }
      );
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), getTimeoutMs());

    const model = process.env.CROSS_ENCODER_MODEL || "cross-encoder/ms-marco-MiniLM-L-6-v2";
    const payload = isHuggingFaceEndpoint(endpoint)
      ? {
          inputs: candidates.map((c) => [query, `${c.constructName}. ${c.definitionText}`]),
          options: { wait_for_model: true },
        }
      : isCloudflareEndpoint(endpoint)
      ? {
          query,
          contexts: candidates.map((c, idx) => ({
            id: idx,
            text: `${c.constructName}. ${c.definitionText}`,
          })),
        }
      : {
          query,
          candidates: candidates.map((c, idx) => ({
            index: idx,
            title: c.constructName,
            text: `${c.constructName}. ${c.definitionText}`,
            source: c.source,
          })),
          model,
        };

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    const apiKey = (process.env.CROSS_ENCODER_API_KEY || "").trim();
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    let serviceResponse: Response;
    try {
      serviceResponse = await fetch(endpoint, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeout);
    }

    if (!serviceResponse.ok) {
      throw new Error(`Cross-encoder service returned ${serviceResponse.status}`);
    }

    const serviceJson = (await serviceResponse.json()) as CrossEncoderResponse;
    const scores = extractScores(serviceJson, candidates.length);
    const blended = blendScores(scores, candidates, semanticBlend);
    const effectiveScores = applyFeatureBoost(
      blended,
      candidates,
      featureBoostTerms,
      featureBoostWeight
    );
    const results = rerankCandidatesByScores(candidates, effectiveScores, topK);

    return NextResponse.json({
      query,
      mode,
      provider: endpoint,
      model,
      semanticBlend,
      featureBoostTerms,
      featureBoostWeight,
      count: results.length,
      results,
    });
  } catch (error: unknown) {
    const details = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { error: "Rerank failed", details },
      { status: 500 }
    );
  }
}
