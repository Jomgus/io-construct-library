import MapClient from "./MapClient";

export default async function MapPage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>;
}) {
  const params = await searchParams;
  const initialQuery = params.q?.trim() || "job satisfaction";

  return <MapClient initialQuery={initialQuery} />;
}
