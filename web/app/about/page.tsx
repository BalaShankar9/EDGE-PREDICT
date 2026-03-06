import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "How It Works",
  description:
    "How SharpEdge AI predicts football matches using 15 data sources, 50-feature ML model, and walk-forward validation.",
};

const DATA_SOURCES = [
  { name: "Football-Data.org", type: "Fixtures & Results" },
  { name: "BetExplorer", type: "Bookmaker Odds" },
  { name: "Forebet", type: "Match Predictions" },
  { name: "PredictZ", type: "Match Predictions" },
  { name: "WinDrawWin", type: "Match Predictions" },
  { name: "FootyStats", type: "Advanced Stats" },
  { name: "Vitibet", type: "Match Predictions" },
  { name: "BetStudy", type: "Match Predictions" },
  { name: "Club ELO", type: "Team Ratings" },
  { name: "FBref", type: "Expected Goals (xG)" },
  { name: "Understat", type: "Shot Statistics" },
  { name: "Football-Data UK", type: "Historical Data" },
  { name: "FPL API", type: "Injuries & Form" },
  { name: "Transfermarkt", type: "Injury Reports" },
  { name: "The Odds API", type: "Live Odds" },
];

const FEATURE_GROUPS = [
  { name: "Form Features", count: 12, desc: "Recent performance, goals scored/conceded, home/away form" },
  { name: "ELO Ratings", count: 6, desc: "Team strength ratings with home advantage" },
  { name: "Head-to-Head", count: 6, desc: "Historical matchup data between teams" },
  { name: "Market Features", count: 8, desc: "Bookmaker odds and implied probabilities" },
  { name: "Context Features", count: 6, desc: "Rest days, distance, schedule congestion" },
  { name: "Shot Statistics", count: 8, desc: "xG, shots on target, shot quality" },
  { name: "Meta Predictions", count: 4, desc: "Consensus from competitor prediction sites" },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-3xl font-bold text-text-primary mb-2">
        How SharpEdge AI Works
      </h1>
      <p className="text-text-secondary mb-8">
        A transparent look at our prediction engine.
      </p>

      <section className="mb-12">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          15 Data Sources, Scraped Daily
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {DATA_SOURCES.map((source) => (
            <div
              key={source.name}
              className="bg-bg-surface border border-border p-4"
            >
              <p className="text-sm font-medium text-text-primary">
                {source.name}
              </p>
              <p className="text-xs text-text-secondary">{source.type}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          50-Feature ML Model
        </h2>
        <p className="text-sm text-text-secondary mb-4">
          Each match is described by 50 engineered features across 7 groups:
        </p>
        <div className="space-y-3">
          {FEATURE_GROUPS.map((group) => (
            <div
              key={group.name}
              className="bg-bg-surface border border-border p-4 flex items-center justify-between"
            >
              <div>
                <p className="text-sm font-medium text-text-primary">
                  {group.name}
                </p>
                <p className="text-xs text-text-secondary">{group.desc}</p>
              </div>
              <span className="text-sm text-accent-primary font-medium tabular-nums">
                {group.count}
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          Walk-Forward Validation
        </h2>
        <div className="bg-bg-surface border border-border p-6 space-y-3 text-sm text-text-secondary">
          <p>
            We use <strong className="text-text-primary">walk-forward validation</strong> - the model is
            trained on past seasons and tested on future matches it has never seen.
            This prevents data leakage and gives realistic accuracy estimates.
          </p>
          <p>
            The ensemble combines <strong className="text-text-primary">XGBoost</strong> (gradient-boosted
            trees) and <strong className="text-text-primary">Poisson regression</strong> (goal-scoring
            model) for robust probability estimates.
          </p>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          Tier System
        </h2>
        <div className="space-y-3">
          <div className="bg-bg-surface border border-tier-platinum/30 p-4">
            <p className="text-sm font-semibold text-tier-platinum">Platinum</p>
            <p className="text-xs text-text-secondary">
              Confidence &ge; 75%. Empirical accuracy: 82.1% (walk-forward, 3504 matches)
            </p>
          </div>
          <div className="bg-bg-surface border border-tier-gold/30 p-4">
            <p className="text-sm font-semibold text-tier-gold">Gold</p>
            <p className="text-xs text-text-secondary">
              Confidence 65-75%. Empirical accuracy: ~76%
            </p>
          </div>
          <div className="bg-bg-surface border border-tier-silver/30 p-4">
            <p className="text-sm font-semibold text-tier-silver">Silver</p>
            <p className="text-xs text-text-secondary">
              Confidence 50-65%. Empirical accuracy: ~68%
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          No Cherry-Picking
        </h2>
        <p className="text-sm text-text-secondary">
          Every pick is published before kick-off and tracked automatically.
          Results are resolved against actual scores. Our{" "}
          <a href="/track-record" className="text-accent-primary hover:underline">
            track record
          </a>{" "}
          is fully transparent and auditable.
        </p>
      </section>
    </div>
  );
}
