import type {
  ApiResponse, Pick, Prediction, MatchDetail, TrackRecord,
  MonthlyRecord, League, LeaguePredictions,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchApi<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    next: { revalidate: 300 },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  const json: ApiResponse<T> = await res.json();
  if (json.status === "error") {
    throw new Error((json.data as Record<string, string>)?.error || "Unknown API error");
  }
  return json.data;
}

export async function getPicksToday(): Promise<Pick[]> {
  return fetchApi<Pick[]>("/api/picks/today");
}

export async function getPredictionsToday(): Promise<Prediction[]> {
  return fetchApi<Prediction[]>("/api/predictions/today");
}

export async function getPredictionsByDate(date: string): Promise<Prediction[]> {
  return fetchApi<Prediction[]>(`/api/predictions/${date}`);
}

export async function getPicksHistory(params?: {
  tier?: string; league?: string; limit?: number; offset?: number;
}): Promise<Pick[]> {
  const searchParams = new URLSearchParams();
  if (params?.tier) searchParams.set("tier", params.tier);
  if (params?.league) searchParams.set("league", params.league);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  const qs = searchParams.toString();
  return fetchApi<Pick[]>(`/api/picks/history${qs ? `?${qs}` : ""}`);
}

export async function getTrackRecord(): Promise<TrackRecord> {
  return fetchApi<TrackRecord>("/api/track-record");
}

export async function getTrackRecordByTier(): Promise<
  Record<string, { total_picks: number; wins: number; losses: number; win_rate: number; total_profit: number; roi: number }>
> {
  return fetchApi("/api/track-record/by-tier");
}

export async function getTrackRecordByLeague(): Promise<
  Record<string, { total_picks: number; wins: number; losses: number; win_rate: number; total_profit: number; roi: number }>
> {
  return fetchApi("/api/track-record/by-league");
}

export async function getMonthlyTrackRecord(): Promise<MonthlyRecord[]> {
  return fetchApi<MonthlyRecord[]>("/api/track-record/monthly");
}

export async function getLeagues(): Promise<League[]> {
  return fetchApi<League[]>("/api/leagues");
}

export async function getLeaguePredictions(slug: string): Promise<LeaguePredictions> {
  return fetchApi<LeaguePredictions>(`/api/league/${slug}/predictions`);
}

export async function getMatchDetail(
  homeTeam: string, awayTeam: string, matchDate: string
): Promise<MatchDetail> {
  return fetchApi<MatchDetail>(
    `/api/match/${encodeURIComponent(homeTeam)}/${encodeURIComponent(awayTeam)}/${matchDate}`
  );
}
