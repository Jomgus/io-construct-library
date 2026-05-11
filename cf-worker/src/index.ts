import { neon } from '@neondatabase/serverless';

type SourceFilter = 'all' | 'OpenAlex' | 'O*NET';
type RankingMode = 'hybrid' | 'vector_only';

type EmbeddingResponse = {
	data: number[][];
};

type SearchRow = {
	id: number;
	Construct_Name: string;
	Source: 'OpenAlex' | 'O*NET';
	Description: string;
	Paper_Count: number;
	Reference_URLs: string | null;
	similarity_score: number;
	exact_match: number;
	rrf_score: number;
	semantic_rrf: number;
	lexical_rrf: number;
	forced_boost: number;
	related: string[] | null;
};

export interface Env {
	AI: {
		run(model: string, payload: { text: string[] }): Promise<EmbeddingResponse>;
	};
	POSTGRES_URL: string;
	SEARCH_RANKING_MODE?: string;
}

const FORCED_CONSTRUCT_ALIASES: Record<string, string[]> = {
	'employee voice': [
		'Psychological safety',
		'Organizational justice',
		'Employee engagement',
		'Openness and Honesty',
		'Speaking',
		'Active Listening',
	],
	'employee silence': [
		'Expressive Suppression',
		'Psychological safety',
		'Employee morale',
		'Withdrawing from others',
	],
	'organizational silence': [
		'Psychological safety',
		'Organizational justice',
		'Employee morale',
		'Human Resources',
	],
	'voice behavior': [
		'Openness and Honesty',
		'Active Listening',
		'Speaking',
		'Interpersonal communication',
		'Work behavior',
	],
};

function expandForcedConstructs(forceConstructs: string[]): string[] {
	const expanded: string[] = [];
	const seen = new Set<string>();
	const add = (value: string) => {
		const trimmed = value.trim();
		if (!trimmed) return;
		const key = trimmed.toLowerCase();
		if (seen.has(key)) return;
		seen.add(key);
		expanded.push(trimmed);
	};
	for (const forced of forceConstructs) {
		add(forced);
		const aliases = FORCED_CONSTRUCT_ALIASES[forced.toLowerCase()] || [];
		for (const alias of aliases) add(alias);
	}
	return expanded.slice(0, 40);
}

const worker = {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);
		const query = url.searchParams.get('q');
		const sourceParam = (url.searchParams.get('source') || 'all') as SourceFilter;
		const source: SourceFilter =
			sourceParam === 'OpenAlex' || sourceParam === 'O*NET' || sourceParam === 'all'
				? sourceParam
				: 'all';
		const minPaperCountRaw = Number(url.searchParams.get('minPaperCount') || '0');
		const minPaperCount = Number.isFinite(minPaperCountRaw) ? Math.max(0, minPaperCountRaw) : 0;
		const limitRaw = Number(url.searchParams.get('limit') || '15');
		const limit = Number.isFinite(limitRaw) ? Math.max(1, Math.min(100, limitRaw)) : 15;
		const forceConstructs = url.searchParams
			.getAll('forceConstruct')
			.map((x) => x.trim())
			.filter((x) => x.length > 0)
			.slice(0, 20);
		const expandedForceConstructs = expandForcedConstructs(forceConstructs);
		const forceConstructsLower = expandedForceConstructs.map((x) => x.toLowerCase());
		const candidatePool = Math.max(60, Math.min(500, limit * 10));
		const rrfK = 50;
		const rankingModeEnv = (env.SEARCH_RANKING_MODE || 'hybrid').toLowerCase();
		const rankingMode: RankingMode = rankingModeEnv === 'vector_only' ? 'vector_only' : 'hybrid';
		// Tuned offline on utaenv benchmark: semantic signal should dominate lexical.
		const semanticRrfWeight = 0.67;
		const lexicalRrfWeight = 0.33;

		if (!query) {
			return new Response(JSON.stringify({ error: "Missing query 'q'" }), {
				status: 400,
				headers: { 'Content-Type': 'application/json' },
			});
		}

		try {
			// 1. Get embedding from Cloudflare AI
			const aiResponse = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
				text: [query],
			});
			const embedding = aiResponse.data[0];

			// 2. Connect to Neon
			const sql = neon(env.POSTGRES_URL);

			let results: SearchRow[] = [];
			if (rankingMode === 'vector_only') {
				results = (await sql`
					WITH params AS (
						SELECT ${JSON.stringify(embedding)}::vector AS query_embedding
					),
					search_results AS (
						SELECT
							m.id,
							m."Construct_Name",
							m."Source",
							m."Description",
							m."Paper_Count",
							m."Reference_URLs",
							m.embedding,
							1 - (m.embedding <=> p.query_embedding) AS similarity_score,
							(CASE WHEN LOWER(m."Construct_Name") = LOWER(${query}) THEN 1 ELSE 0 END) AS exact_match,
							0.0::float8 AS rrf_score,
							0.0::float8 AS semantic_rrf,
							0.0::float8 AS lexical_rrf,
							0.0::float8 AS forced_boost
						FROM master_constructs m
						CROSS JOIN params p
						WHERE (${source} = 'all' OR m."Source" = ${source})
						  AND (m."Source" != 'OpenAlex' OR m."Paper_Count" >= ${minPaperCount})
						ORDER BY
							exact_match DESC,
							m.embedding <=> p.query_embedding ASC,
							m."Paper_Count" DESC,
							m.id ASC
						LIMIT ${limit}
					)
					SELECT *, (
						SELECT json_agg(sub."Construct_Name")
						FROM (
							SELECT "Construct_Name"
							FROM master_constructs m2
							WHERE m2.id != search_results.id
							ORDER BY m2.embedding <=> search_results.embedding
							LIMIT 3
						) sub
					) as related
					FROM search_results
				`) as SearchRow[];
			} else {
				// Hybrid retrieval:
				// - semantic candidates from pgvector distance
				// - lexical candidates from PostgreSQL full-text search
				// - reciprocal rank fusion (RRF) for robust merged ranking
				results = (await sql`
					WITH params AS (
						SELECT
							${JSON.stringify(embedding)}::vector AS query_embedding,
							websearch_to_tsquery('english', ${query}) AS ts_query
					),
					filtered AS (
						SELECT
							m.id,
							m."Construct_Name",
							m."Source",
							m."Description",
							m."Paper_Count",
							m."Reference_URLs",
							m.embedding,
							setweight(to_tsvector('english', coalesce(m."Construct_Name", '')), 'A') ||
							setweight(to_tsvector('english', coalesce(m."Description", '')), 'B') AS textsearch
						FROM master_constructs m
						WHERE (${source} = 'all' OR "Source" = ${source})
						  AND (m."Source" != 'OpenAlex' OR m."Paper_Count" >= ${minPaperCount})
					),
					semantic AS (
						SELECT
							f.id,
							row_number() OVER (
								ORDER BY
									(CASE WHEN LOWER(f."Construct_Name") = LOWER(${query}) THEN 1 ELSE 0 END) DESC,
									f.embedding <=> p.query_embedding ASC,
									f."Paper_Count" DESC,
									f.id ASC
							) AS rank_position
						FROM filtered f
						CROSS JOIN params p
						ORDER BY
							(CASE WHEN LOWER(f."Construct_Name") = LOWER(${query}) THEN 1 ELSE 0 END) DESC,
							f.embedding <=> p.query_embedding ASC,
							f."Paper_Count" DESC,
							f.id ASC
						LIMIT ${candidatePool}
					),
					lexical AS (
						SELECT
							f.id,
							row_number() OVER (
								ORDER BY
									ts_rank_cd(f.textsearch, p.ts_query, 32) DESC,
									f."Paper_Count" DESC,
									f.id ASC
							) AS rank_position
						FROM filtered f
						CROSS JOIN params p
						WHERE f.textsearch @@ p.ts_query
						ORDER BY
							ts_rank_cd(f.textsearch, p.ts_query, 32) DESC,
							f."Paper_Count" DESC,
							f.id ASC
						LIMIT ${candidatePool}
					),
					forced AS (
						SELECT
							f.id,
							row_number() OVER (
								ORDER BY
									f."Paper_Count" DESC,
									f.id ASC
							) AS rank_position
						FROM filtered f
						WHERE
							array_length(${forceConstructsLower}::text[], 1) IS NOT NULL
							AND lower(f."Construct_Name") = ANY(${forceConstructsLower}::text[])
						ORDER BY
							f."Paper_Count" DESC,
							f.id ASC
						LIMIT 20
					),
					fused AS (
						SELECT
							id,
							sum(rrf_part) AS rrf_score,
							sum(semantic_part) AS semantic_rrf,
							sum(lexical_part) AS lexical_rrf,
							sum(forced_part) AS forced_boost
						FROM (
							SELECT
								id,
								(${semanticRrfWeight}::float8) / ((${rrfK}::float8) + rank_position::float8) AS rrf_part,
								1.0::float8 / ((${rrfK}::float8) + rank_position::float8) AS semantic_part,
								0.0 AS lexical_part,
								0.0::float8 AS forced_part
							FROM semantic
							UNION ALL
							SELECT
								id,
								(${lexicalRrfWeight}::float8) / ((${rrfK}::float8) + rank_position::float8) AS rrf_part,
								0.0 AS semantic_part,
								1.0::float8 / ((${rrfK}::float8) + rank_position::float8) AS lexical_part,
								0.0::float8 AS forced_part
							FROM lexical
							UNION ALL
							SELECT
								id,
								0.1::float8 AS rrf_part,
								0.0::float8 AS semantic_part,
								0.0::float8 AS lexical_part,
								0.1::float8 AS forced_part
							FROM forced
						) ranked_parts
						GROUP BY id
					),
					search_results AS (
						SELECT
							f.id,
							f."Construct_Name",
							f."Source",
							f."Description",
							f."Paper_Count",
							f."Reference_URLs",
							f.embedding,
							1 - (f.embedding <=> p.query_embedding) AS similarity_score,
							(CASE WHEN LOWER(f."Construct_Name") = LOWER(${query}) THEN 1 ELSE 0 END) AS exact_match,
							fused.rrf_score,
							fused.semantic_rrf,
							fused.lexical_rrf,
							fused.forced_boost
						FROM fused
						JOIN filtered f ON f.id = fused.id
						CROSS JOIN params p
						ORDER BY
							exact_match DESC,
							(fused.rrf_score + fused.forced_boost) DESC,
							similarity_score DESC,
							f."Paper_Count" DESC,
							f.id ASC
						LIMIT ${limit}
					)
					SELECT *, (
						SELECT json_agg(sub."Construct_Name")
						FROM (
							SELECT "Construct_Name"
							FROM master_constructs m2
							WHERE m2.id != search_results.id
							ORDER BY m2.embedding <=> search_results.embedding
							LIMIT 3
						) sub
					) as related
					FROM search_results
				`) as SearchRow[];
			}

			// 4. Return results
			return new Response(JSON.stringify({
				query,
				rankingMode,
				results: results.map((r) => ({
					constructName: r.Construct_Name,
					source: r.Source,
					definitionText: r.Description,
					paperCount: r.Paper_Count,
					referenceUrls: r.Reference_URLs ? r.Reference_URLs.split(',') : [],
					similarityScore: r.similarity_score,
					rrfScore: r.rrf_score,
					semanticRrf: r.semantic_rrf,
					lexicalRrf: r.lexical_rrf,
					forcedBoost: r.forced_boost,
					exactMatch: !!r.exact_match,
					noDoiYet: r.Source === 'OpenAlex' && (!r.Reference_URLs || r.Reference_URLs === ''),
					related: r.related || []
				}))
			}), {
				headers: { 
					'Content-Type': 'application/json',
					'Access-Control-Allow-Origin': '*'
				},
			});

		} catch (error: unknown) {
			const message = error instanceof Error ? error.message : 'Unknown error';
			return new Response(JSON.stringify({ error: message }), {
				status: 500,
				headers: { 'Content-Type': 'application/json' },
			});
		}
	},
};

export default worker;
