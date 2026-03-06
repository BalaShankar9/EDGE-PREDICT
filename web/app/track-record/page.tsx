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
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center">
          <h1 className="text-3xl sm:text-4xl font-bold mb-4">
            <span className="gradient-text">Track Record</span>
          </h1>
          <p className="text-text-secondary max-w-md mx-auto">
            No resolved picks yet. Check back after matches complete. Every pick is tracked automatically.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero banner */}
      <div className="text-center mb-10">
        <h1 className="text-3xl sm:text-4xl font-bold mb-2">
          <span className="gradient-text">Our Numbers. No Cherry-Picking.</span>
        </h1>
        <p className="text-text-secondary">
          Every pick published before kick-off. Every result tracked automatically.
        </p>
      </div>

      {/* Stats hero with glow */}
      <StatsHero record={trackRecord} />

      {/* Streak info */}
      {(trackRecord.longest_win_streak > 0 || trackRecord.longest_loss_streak > 0) && (
        <div className="flex flex-wrap gap-4 mt-6 justify-center">
          {trackRecord.longest_win_streak > 0 && (
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-success/5 border border-success/20 text-sm">
              <span>&#128293;</span>
              <span className="text-success font-semibold">{trackRecord.longest_win_streak}W</span>
              <span className="text-text-secondary">best streak</span>
            </div>
          )}
          {trackRecord.longest_loss_streak > 0 && (
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-bg-surface border border-border text-sm">
              <span className="text-danger font-semibold">{trackRecord.longest_loss_streak}L</span>
              <span className="text-text-secondary">worst streak</span>
            </div>
          )}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-10">
        <TierChart byTier={trackRecord.by_tier} />
        {monthly.length > 0 && <MonthlyChart data={monthly} />}
      </div>

      {/* League breakdown */}
      <div className="mt-10">
        <div className="flex items-center gap-3 mb-4">
          <span className="w-1 h-5 bg-accent-primary" />
          <h2 className="text-lg font-semibold text-text-primary">By League</h2>
        </div>
        <LeagueTable byLeague={trackRecord.by_league} />
      </div>

      {/* Recent results */}
      {recentResults.length > 0 && (
        <div className="mt-10">
          <ResultsList picks={recentResults} />
        </div>
      )}
    </div>
  );
}
