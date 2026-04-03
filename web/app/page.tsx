import { TierFilter } from "@/components/picks/tier-filter";
import { ResultsList } from "@/components/picks/results-list";
import { HeroSection } from "@/components/home/hero-section";
import { getPicksToday, getPicksHistory, getTrackRecord, getPredictionsToday, getPredictionsUpcoming } from "@/lib/api";
import type { Prediction } from "@/lib/types";

export const revalidate = 300;

function PredictionsTable({ predictions }: { predictions: Prediction[] }) {
  if (predictions.length === 0) return null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-text-secondary text-left">
            <th className="py-3 px-2">Date</th>
            <th className="py-3 px-2">League</th>
            <th className="py-3 px-2">Match</th>
            <th className="py-3 px-2 text-center">Home</th>
            <th className="py-3 px-2 text-center">Draw</th>
            <th className="py-3 px-2 text-center">Away</th>
            <th className="py-3 px-2 text-center">Prediction</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p, i) => {
            const probs = [p.prob_home, p.prob_draw, p.prob_away];
            const maxIdx = probs.indexOf(Math.max(...probs));
            const labels = ["Home", "Draw", "Away"];
            const prediction = labels[maxIdx];
            const confidence = probs[maxIdx];

            return (
              <tr key={i} className="border-b border-border/50 hover:bg-bg-surface/50 transition-colors">
                <td className="py-3 px-2 text-text-secondary tabular-nums">{p.match_date}</td>
                <td className="py-3 px-2">
                  <span className="text-xs px-2 py-0.5 bg-bg-surface border border-border rounded">
                    {p.league}
                  </span>
                </td>
                <td className="py-3 px-2 font-medium text-text-primary">
                  {p.home_team} <span className="text-text-secondary">vs</span> {p.away_team}
                </td>
                <td className={`py-3 px-2 text-center tabular-nums ${maxIdx === 0 ? "text-accent-primary font-bold" : "text-text-secondary"}`}>
                  {(p.prob_home * 100).toFixed(0)}%
                </td>
                <td className={`py-3 px-2 text-center tabular-nums ${maxIdx === 1 ? "text-accent-primary font-bold" : "text-text-secondary"}`}>
                  {(p.prob_draw * 100).toFixed(0)}%
                </td>
                <td className={`py-3 px-2 text-center tabular-nums ${maxIdx === 2 ? "text-accent-primary font-bold" : "text-text-secondary"}`}>
                  {(p.prob_away * 100).toFixed(0)}%
                </td>
                <td className="py-3 px-2 text-center">
                  <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-1 rounded ${
                    confidence >= 0.55 ? "bg-success/10 text-success border border-success/20" :
                    confidence >= 0.45 ? "bg-warning/10 text-warning border border-warning/20" :
                    "bg-bg-surface text-text-secondary border border-border"
                  }`}>
                    {prediction} {(confidence * 100).toFixed(0)}%
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default async function HomePage() {
  let picks: any[] = [];
  let predictions: Prediction[] = [];
  let recentResults: any[] = [];
  let trackRecord: any = null;

  try {
    [picks, predictions, recentResults, trackRecord] = await Promise.all([
      getPicksToday(),
      getPredictionsUpcoming(7),  // Show next 7 days of predictions
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

      {/* Today's Predictions */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-1 h-6 bg-accent-primary" />
          <h2 className="text-2xl font-bold text-text-primary">Upcoming Predictions</h2>
          <span className="text-sm text-text-secondary ml-auto tabular-nums">
            {predictions.length} matches
          </span>
        </div>
        {predictions.length > 0 ? (
          <PredictionsTable predictions={predictions} />
        ) : (
          <div className="text-center py-12 bg-bg-surface border border-border rounded-lg">
            <p className="text-text-secondary">No predictions for today. Check back tomorrow.</p>
          </div>
        )}
      </section>

      {/* Today's Picks (filtered value bets) */}
      {picks.length > 0 && (
        <section className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <span className="w-1 h-6 bg-success" />
            <h2 className="text-2xl font-bold text-text-primary">Value Bets</h2>
            <span className="text-sm text-text-secondary ml-auto tabular-nums">{picks.length} picks</span>
          </div>
          <TierFilter picks={picks} />
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
