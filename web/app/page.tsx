import { StatCard } from "@/components/shared/stat-card";
import { TierFilter } from "@/components/picks/tier-filter";
import { ResultsList } from "@/components/picks/results-list";
import { getPicksToday, getPicksHistory, getTrackRecord } from "@/lib/api";
import { formatPercent } from "@/lib/utils";

export const revalidate = 300;

export default async function HomePage() {
  let picks: any[] = [];
  let recentResults: any[] = [];
  let trackRecord: any = null;

  try {
    [picks, recentResults, trackRecord] = await Promise.all([
      getPicksToday(),
      getPicksHistory({ limit: 10 }),
      getTrackRecord(),
    ]);
  } catch {
    // Graceful degradation — show empty state
  }

  const platinumWinRate = trackRecord?.by_tier?.Platinum?.win_rate;
  const monthlyRoi = trackRecord?.overall?.roi;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <StatCard label="Today's Picks" value={String(picks.length)} />
        <StatCard label="Platinum Accuracy" value={platinumWinRate ? formatPercent(platinumWinRate) : "—"} />
        <StatCard label="Overall ROI" value={monthlyRoi != null ? `${monthlyRoi > 0 ? "+" : ""}${monthlyRoi.toFixed(1)}%` : "—"} />
      </div>

      <h2 className="text-2xl font-bold text-text-primary mb-6">Today&apos;s Picks</h2>
      <TierFilter picks={picks} />

      {recentResults.length > 0 && (
        <div className="mt-12">
          <ResultsList picks={recentResults} />
        </div>
      )}
    </div>
  );
}
