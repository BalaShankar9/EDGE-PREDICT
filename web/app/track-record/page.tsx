import type { Metadata } from "next";
import { StatsHero } from "@/components/track-record/stats-hero";
import { TierChart } from "@/components/track-record/tier-chart";
import { MonthlyChart } from "@/components/track-record/monthly-chart";
import { LeagueTable } from "@/components/track-record/league-table";
import { ResultsList } from "@/components/picks/results-list";
import {
  getTrackRecord,
  getMonthlyTrackRecord,
  getPicksHistory,
} from "@/lib/api";

export const metadata: Metadata = {
  title: "Track Record",
  description:
    "SharpEdge AI transparent prediction accuracy - win rates, ROI, and profit by tier and league.",
};

export const revalidate = 300;

export default async function TrackRecordPage() {
  let trackRecord = null;
  let monthly: any[] = [];
  let recentResults: any[] = [];

  try {
    [trackRecord, monthly, recentResults] = await Promise.all([
      getTrackRecord(),
      getMonthlyTrackRecord(),
      getPicksHistory({ limit: 20 }),
    ]);
  } catch {
    // Graceful degradation
  }

  if (!trackRecord) {
    return (
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-2xl font-bold text-text-primary mb-6">
          Track Record
        </h1>
        <p className="text-text-secondary">
          No resolved picks yet. Check back after matches complete.
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold text-text-primary mb-6">
        Track Record
      </h1>

      <StatsHero record={trackRecord} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <TierChart byTier={trackRecord.by_tier} />
        {monthly.length > 0 && <MonthlyChart data={monthly} />}
      </div>

      <div className="mt-8">
        <h2 className="text-lg font-semibold text-text-primary mb-4">
          By League
        </h2>
        <LeagueTable byLeague={trackRecord.by_league} />
      </div>

      {recentResults.length > 0 && (
        <div className="mt-8">
          <ResultsList picks={recentResults} />
        </div>
      )}
    </div>
  );
}
