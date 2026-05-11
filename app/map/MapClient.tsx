"use client";

import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from "d3-force";
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

type GraphNode = {
  id: string;
  label: string;
  tier: 0 | 1 | 2;
  source?: SourceType;
  paperCount?: number;
  fx?: number | null;
  fy?: number | null;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
};

type GraphEdge = {
  source: string | GraphNode;
  target: string | GraphNode;
};

type Viewport = {
  x: number;
  y: number;
  scale: number;
};

type SelectedNodeData = {
  label: string;
  source?: SourceType;
  paperCount?: number;
  definitionText?: string;
  referenceUrls?: string[];
  noDoiYet?: boolean;
  isApproximate?: boolean;
};

const WIDTH = 1200;
const HEIGHT = 820;
const BASE_SCALE = 1;

export default function MapClient({ initialQuery }: { initialQuery: string }) {
  const [query, setQuery] = useState(initialQuery);
  const [submittedQuery, setSubmittedQuery] = useState(initialQuery);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNodeData, setSelectedNodeData] = useState<SelectedNodeData | null>(null);
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [viewport, setViewport] = useState<Viewport>({ x: 0, y: 0, scale: BASE_SCALE });

  const svgRef = useRef<SVGSVGElement | null>(null);
  const isPanningRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0, originX: 0, originY: 0 });

  useEffect(() => {
    setQuery(initialQuery);
    setSubmittedQuery(initialQuery);
  }, [initialQuery]);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams({
          q: submittedQuery,
          source: "all",
          minPaperCount: "25",
          limit: "8",
        });
        const response = await fetch(`/api/search?${params.toString()}`);
        if (!response.ok) throw new Error("Map search request failed.");
        const payload = (await response.json()) as { results: SearchResult[] };
        setResults(payload.results || []);
        setViewport({ x: 0, y: 0, scale: BASE_SCALE });
        setSelectedNodeId(null);
        setSelectedNodeData(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unexpected error.");
        setResults([]);
      } finally {
        setLoading(false);
      }
    }

    if (submittedQuery.trim()) load();
  }, [submittedQuery]);

  const graph = useMemo(() => {
    const centerId = `query:${submittedQuery}`;
    const nodes: GraphNode[] = [
      {
        id: centerId,
        label: submittedQuery,
        tier: 0,
        x: WIDTH / 2,
        y: HEIGHT / 2,
        fx: WIDTH / 2,
        fy: HEIGHT / 2,
      },
    ];
    const edges: GraphEdge[] = [];
    const seen = new Set<string>([centerId]);

    results.slice(0, 8).forEach((result) => {
      const nodeId = `result:${result.constructName}`;
      nodes.push({
        id: nodeId,
        label: result.constructName,
        tier: 1,
        source: result.source,
        paperCount: result.paperCount,
      });
      edges.push({ source: centerId, target: nodeId });
      seen.add(nodeId);

      result.related.slice(0, 4).forEach((relatedName) => {
        const relatedId = `related:${relatedName.toLowerCase()}`;
        if (!seen.has(relatedId)) {
          nodes.push({
            id: relatedId,
            label: relatedName,
            tier: 2,
          });
          seen.add(relatedId);
        }
        edges.push({ source: nodeId, target: relatedId });
      });
    });

    const clonedNodes = nodes.map((node) => ({ ...node }));
    const clonedEdges = edges.map((edge) => ({ ...edge }));

    const simulation = forceSimulation<GraphNode>(clonedNodes)
      .force(
        "charge",
        forceManyBody<GraphNode>().strength((node) => (node.tier === 0 ? -850 : node.tier === 1 ? -420 : -180))
      )
      .force(
        "link",
        forceLink<GraphNode, GraphEdge>(clonedEdges)
          .id((node) => node.id)
          .distance((edge) => {
            const source = edge.source as GraphNode;
            const target = edge.target as GraphNode;
            return source.tier === 0 || target.tier === 0 ? 155 : 92;
          })
          .strength((edge) => {
            const source = edge.source as GraphNode;
            const target = edge.target as GraphNode;
            return source.tier === 0 || target.tier === 0 ? 0.85 : 0.5;
          })
      )
      .force(
        "collide",
        forceCollide<GraphNode>().radius((node) => (node.tier === 0 ? 48 : node.tier === 1 ? 34 : 24))
      )
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2));

    for (let i = 0; i < 220; i += 1) simulation.tick();
    simulation.stop();

    const centerNode = clonedNodes.find((node) => node.id === centerId);
    if (centerNode) {
      centerNode.x = WIDTH / 2;
      centerNode.y = HEIGHT / 2;
    }

    return { nodes: clonedNodes, edges: clonedEdges };
  }, [results, submittedQuery]);

  useEffect(() => {
    async function loadSelection(nodeId: string) {
      const node = graph.nodes.find((item) => item.id === nodeId);
      if (!node) return;

      if (node.tier === 0) {
        setSelectedNodeData({
          label: node.label,
          definitionText: "Query node",
        });
        return;
      }

      const direct = results.find((item) => item.constructName === node.label);
      if (direct) {
        setSelectedNodeData({
          label: direct.constructName,
          source: direct.source,
          paperCount: direct.paperCount,
          definitionText: direct.definitionText,
          referenceUrls: direct.referenceUrls,
          noDoiYet: direct.noDoiYet,
        });
        return;
      }

      setSelectionLoading(true);
      try {
        const params = new URLSearchParams({
          q: node.label,
          source: "all",
          minPaperCount: "0",
          limit: "1",
        });
        const response = await fetch(`/api/search?${params.toString()}`);
        if (!response.ok) throw new Error("Node detail request failed.");
        const payload = (await response.json()) as { results: SearchResult[] };
        const best = payload.results?.[0];
        if (!best) {
          setSelectedNodeData({ label: node.label });
        } else {
          setSelectedNodeData({
            label: best.constructName,
            source: best.source,
            paperCount: best.paperCount,
            definitionText: best.definitionText,
            referenceUrls: best.referenceUrls,
            noDoiYet: best.noDoiYet,
            isApproximate: best.constructName.toLowerCase() !== node.label.toLowerCase(),
          });
        }
      } catch {
        setSelectedNodeData({ label: node.label });
      } finally {
        setSelectionLoading(false);
      }
    }

    if (!selectedNodeId) return;
    loadSelection(selectedNodeId);
  }, [graph.nodes, results, selectedNodeId]);

  function screenToGraph(clientX: number, clientY: number) {
    if (!svgRef.current) return { x: WIDTH / 2, y: HEIGHT / 2 };
    const rect = svgRef.current.getBoundingClientRect();
    const sx = ((clientX - rect.left) / rect.width) * WIDTH;
    const sy = ((clientY - rect.top) / rect.height) * HEIGHT;
    return {
      x: (sx - viewport.x) / viewport.scale,
      y: (sy - viewport.y) / viewport.scale,
    };
  }

  function handleWheel(event: React.WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const direction = event.deltaY > 0 ? 0.92 : 1.08;
    const nextScale = Math.min(2.6, Math.max(0.6, viewport.scale * direction));
    const point = screenToGraph(event.clientX, event.clientY);
    setViewport((prev) => ({
      scale: nextScale,
      x: prev.x - point.x * (nextScale - prev.scale),
      y: prev.y - point.y * (nextScale - prev.scale),
    }));
  }

  function startPan(event: React.PointerEvent<SVGSVGElement>) {
    isPanningRef.current = true;
    panStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      originX: viewport.x,
      originY: viewport.y,
    };
  }

  function movePan(event: React.PointerEvent<SVGSVGElement>) {
    if (!isPanningRef.current) return;
    const dx = event.clientX - panStartRef.current.x;
    const dy = event.clientY - panStartRef.current.y;
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    setViewport((prev) => ({
      ...prev,
      x: panStartRef.current.originX + (dx / rect.width) * WIDTH,
      y: panStartRef.current.originY + (dy / rect.height) * HEIGHT,
    }));
  }

  function endPan() {
    isPanningRef.current = false;
  }

  function nudgeZoom(direction: "in" | "out") {
    const factor = direction === "in" ? 1.12 : 0.9;
    setViewport((prev) => {
      const nextScale = Math.min(2.6, Math.max(0.6, prev.scale * factor));
      const graphCenterX = (WIDTH / 2 - prev.x) / prev.scale;
      const graphCenterY = (HEIGHT / 2 - prev.y) / prev.scale;
      return {
        scale: nextScale,
        x: WIDTH / 2 - graphCenterX * nextScale,
        y: HEIGHT / 2 - graphCenterY * nextScale,
      };
    });
  }

  return (
    <main className="min-h-screen bg-[var(--background)] px-6 py-8 text-[var(--foreground)] sm:px-10 lg:px-14">
      <div className="mx-auto max-w-6xl">
        <div className="border-b border-[var(--border)] pb-4 text-[11px] uppercase tracking-[0.24em] text-[var(--muted)]">
          IO Construct Library Map
        </div>

        <div className="grid items-start gap-10 pt-10 lg:grid-cols-[320px_1fr]">
          <aside className="sticky top-8 space-y-6 self-start">
            <div>
              <h1 className="font-display text-4xl tracking-tight">Local map prototype</h1>
              <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
                Query-centered graph built from the top search results and their related constructs.
                Pan, zoom, and use this to judge the graph view before we compute a full network.
              </p>
              <p className="mt-3 text-sm text-[var(--muted)]">
                <Link href={`/?q=${encodeURIComponent(submittedQuery)}`} className="underline underline-offset-4">
                  Back to search
                </Link>
              </p>
            </div>

            <form
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                setSubmittedQuery(query.trim() || "job satisfaction");
              }}
            >
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search a construct..."
                className="w-full border-b border-[var(--foreground)] bg-transparent px-0 py-3 text-lg outline-none placeholder:text-[#9b8f7f]"
              />
              <button
                type="submit"
                className="rounded-full border border-[var(--foreground)] bg-[var(--foreground)] px-4 py-2 text-sm text-[var(--surface)]"
              >
                Build map
              </button>
            </form>

            <div className="space-y-2 text-sm text-[var(--muted)]">
              <p>Center node: query</p>
              <p>Dark nodes: top results</p>
              <p>Light nodes: related constructs</p>
              <p>Drag background to pan. Use wheel or buttons to zoom.</p>
            </div>

            {selectionLoading ? (
              <div className="max-h-[360px] min-h-[360px] overflow-y-auto rounded-3xl border border-[var(--border)] bg-[var(--surface)] px-4 py-4 text-sm text-[var(--muted)]">
                Loading node details...
              </div>
            ) : selectedNodeData ? (
              <div className="max-h-[360px] min-h-[360px] overflow-y-auto rounded-3xl border border-[var(--border)] bg-[var(--surface)] px-4 py-4 text-sm">
                <p className="font-medium text-[var(--foreground)]">{selectedNodeData.label}</p>
                {selectedNodeData.source ? (
                  <p className="mt-2 text-[var(--muted)]">
                    {selectedNodeData.source} · {selectedNodeData.paperCount ?? 0} papers
                  </p>
                ) : (
                  <p className="mt-2 text-[var(--muted)]">Node selected</p>
                )}
                {selectedNodeData.isApproximate ? (
                  <p className="mt-2 text-[var(--muted)]">Closest match shown for this node.</p>
                ) : null}
                {selectedNodeData.definitionText ? (
                  <p className="mt-3 leading-7 text-[var(--muted)]">{selectedNodeData.definitionText}</p>
                ) : null}
                {selectedNodeData.referenceUrls && selectedNodeData.referenceUrls.length > 0 ? (
                  <div className="mt-4 space-y-2">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-[var(--muted)]">Sources</p>
                    {selectedNodeData.referenceUrls.map((url) => (
                      <a
                        key={url}
                        href={url}
                        target="_blank"
                        rel="noreferrer"
                        className="block break-all text-[var(--accent)] underline underline-offset-4"
                      >
                        {url}
                      </a>
                    ))}
                  </div>
                ) : null}
                {selectedNodeData.noDoiYet ? <p className="mt-3 text-[var(--muted)]">No DOI available</p> : null}
              </div>
            ) : (
              <div className="max-h-[360px] min-h-[360px] rounded-3xl border border-[var(--border)] bg-[var(--surface)] px-4 py-4 text-sm text-[var(--muted)]">
                Click a node to inspect its details and source links.
              </div>
            )}

            {error ? <p className="text-sm text-red-700">{error}</p> : null}
            {loading ? <p className="text-sm text-[var(--muted)]">Building map...</p> : null}
          </aside>

          <section className="relative overflow-hidden rounded-[2rem] border border-[var(--border)] bg-[var(--surface)]">
            <div className="absolute right-5 top-5 z-10 flex gap-2">
              <button
                type="button"
                onClick={() => nudgeZoom("in")}
                className="h-10 w-10 rounded-full border border-[var(--border)] bg-[var(--surface)] text-lg text-[var(--foreground)]"
                aria-label="Zoom in"
              >
                +
              </button>
              <button
                type="button"
                onClick={() => nudgeZoom("out")}
                className="h-10 w-10 rounded-full border border-[var(--border)] bg-[var(--surface)] text-lg text-[var(--foreground)]"
                aria-label="Zoom out"
              >
                −
              </button>
              <button
                type="button"
                onClick={() => setViewport({ x: 0, y: 0, scale: BASE_SCALE })}
                className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 text-sm text-[var(--foreground)]"
              >
                Reset
              </button>
            </div>
            <svg
              ref={svgRef}
              viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
              className="h-[760px] w-full"
              role="img"
              aria-label="Construct map"
              onWheel={handleWheel}
              onPointerDown={startPan}
              onPointerMove={movePan}
              onPointerUp={endPan}
              onPointerLeave={endPan}
            >
              <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="#fbf6ef" />

              <g transform={`translate(${viewport.x} ${viewport.y}) scale(${viewport.scale})`}>
                {graph.edges.map((edge) => {
                  const source = edge.source as GraphNode;
                  const target = edge.target as GraphNode;
                  return (
                    <line
                      key={`${source.id}-${target.id}`}
                      x1={source.x ?? 0}
                      y1={source.y ?? 0}
                      x2={target.x ?? 0}
                      y2={target.y ?? 0}
                      stroke="rgba(96, 88, 78, 0.28)"
                      strokeWidth={source.tier === 0 || target.tier === 0 ? 1.6 : 1}
                    />
                  );
                })}

                {graph.nodes.map((node) => {
                  const radius = node.tier === 0 ? 18 : node.tier === 1 ? 11 : 7;
                  const fill =
                    node.tier === 0 ? "var(--accent)" : node.tier === 1 ? "#2b241d" : "#d8c9b4";
                  const fontSize = node.tier === 0 ? 21 : node.tier === 1 ? 12 : 10;
                  const label = node.label.length > 22 ? `${node.label.slice(0, 22)}…` : node.label;
                  const isSelected = selectedNodeId === node.id;

                  return (
                    <g
                      key={node.id}
                      onClick={(event) => {
                        event.stopPropagation();
                        setSelectedNodeId(node.id);
                      }}
                      className="cursor-pointer"
                    >
                      <circle
                        cx={node.x ?? 0}
                        cy={node.y ?? 0}
                        r={radius}
                        fill={fill}
                        stroke={isSelected ? "var(--accent)" : "transparent"}
                        strokeWidth={isSelected ? 8 : 0}
                      />
                      <text
                        x={node.x ?? 0}
                        y={(node.y ?? 0) + radius + 15}
                        textAnchor="middle"
                        fontSize={fontSize}
                        fill="var(--foreground)"
                        pointerEvents="none"
                      >
                        {label}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          </section>
        </div>
      </div>
    </main>
  );
}
