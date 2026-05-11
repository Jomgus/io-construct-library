export type RerankMode = "off" | "cross_encoder_small";

export type SearchCandidate = {
  constructName: string;
  source: "OpenAlex" | "O*NET";
  definitionText: string;
  paperCount: number;
  referenceUrls: string[];
  similarityScore?: number;
  rrfScore?: number;
  semanticRrf?: number;
  lexicalRrf?: number;
  exactMatch?: boolean;
  noDoiYet?: boolean;
  related?: string[];
};

type RankedCandidate = SearchCandidate & { rerankScore: number };

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function rerankCandidatesByScores(
  candidates: SearchCandidate[],
  scores: number[],
  topK: number,
): SearchCandidate[] {
  const k = clamp(topK, 1, Math.max(1, candidates.length));
  const scored: RankedCandidate[] = candidates.map((c, idx) => ({
    ...c,
    rerankScore: Number.isFinite(scores[idx]) ? scores[idx] : Number.NEGATIVE_INFINITY,
  }));

  scored.sort((a, b) => {
    if (b.rerankScore !== a.rerankScore) return b.rerankScore - a.rerankScore;
    if ((b.exactMatch ? 1 : 0) !== (a.exactMatch ? 1 : 0)) {
      return (b.exactMatch ? 1 : 0) - (a.exactMatch ? 1 : 0);
    }
    return b.paperCount - a.paperCount;
  });
  return scored.slice(0, k).map((candidate) => {
    const { rerankScore, ...rest } = candidate;
    void rerankScore;
    return rest;
  });
}

export function passthroughTopK(
  candidates: SearchCandidate[],
  topK: number
): SearchCandidate[] {
  const k = clamp(topK, 1, Math.max(1, candidates.length));
  return candidates.slice(0, k);
}
