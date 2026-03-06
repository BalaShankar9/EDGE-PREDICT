import { TierFilter } from "@/components/picks/tier-filter";
import { ResultsList } from "@/components/picks/results-list";
import { HeroSection } from "@/components/home/hero-section";
import { LivePredictionsTable } from "@/components/home/live-predictions-table";
import { getPicksToday, getPicksHistory, getTrackRecord } from "@/lib/api";

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

  const winRate = trackRecord?.overall?.win_rate ?? trackRecord?.win_rate ?? null;
  const roi = trackRecord?.overall?.roi ?? trackRecord?.roi ?? null;
  const totalPicks = trackRecord?.overall?.total_picks ?? trackRecord?.total_picks ?? 0;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
      {/* Hero */}
      <HeroSection totalPicks={totalPicks} winRate={winRate} roi={roi} />

      {/* Today's Picks */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-1 h-6 bg-accent-primary" />
          <h2 className="text-2xl font-bold text-text-primary">Today&apos;s Picks</h2>
          <span className="text-sm text-text-secondary ml-auto tabular-nums">{picks.length} picks</span>
        </div>
        <TierFilter picks={picks} />
      </section>

      {/* Live Predictions Table */}
      {picks.length > 0 && (
        <section className="mb-12 animate-slide-in-up-delay-3">
          <LivePredictionsTable picks={picks} />
        </section>
      )}

      {/* Recent Results */}
      {recentResults.length > 0 && (
        <section className="mb-12">
          <ResultsList picks={recentResults} />
        </section>
      )}
    </div>
  );
}
