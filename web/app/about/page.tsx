import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "How It Works",
  description:
    "How SharpEdge AI predicts football matches using 15 data sources, 50-feature ML model, and walk-forward validation.",
};

const DATA_SOURCES = [
  { name: "Football-Data.org", type: "Fixtures & Results", icon: "\u26BD" },
  { name: "BetExplorer", type: "Bookmaker Odds", icon: "\uD83D\uDCCA" },
  { name: "Forebet", type: "Match Predictions", icon: "\uD83D\uDD2E" },
  { name: "PredictZ", type: "Match Predictions", icon: "\uD83D\uDD2E" },
  { name: "WinDrawWin", type: "Match Predictions", icon: "\uD83D\uDD2E" },
  { name: "FootyStats", type: "Advanced Stats", icon: "\uD83D\uDCC8" },
  { name: "Vitibet", type: "Match Predictions", icon: "\uD83D\uDD2E" },
  { name: "BetStudy", type: "Match Predictions", icon: "\uD83D\uDD2E" },
  { name: "Club ELO", type: "Team Ratings", icon: "\uD83C\uDFC6" },
  { name: "FBref", type: "Expected Goals (xG)", icon: "\uD83C\uDFAF" },
  { name: "Understat", type: "Shot Statistics", icon: "\uD83D\uDCA5" },
  { name: "Football-Data UK", type: "Historical Data", icon: "\uD83D\uDCC1" },
  { name: "FPL API", type: "Injuries & Form", icon: "\uD83E\uDE7A" },
  { name: "Transfermarkt", type: "Injury Reports", icon: "\uD83C\uDFE5" },
  { name: "The Odds API", type: "Live Odds", icon: "\uD83D\uDCB0" },
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

const PIPELINE_STEPS = [
  { label: "Collect", desc: "Scrape 15 sources daily" },
  { label: "Engineer", desc: "Build 50 features per match" },
  { label: "Predict", desc: "XGBoost + Poisson ensemble" },
  { label: "Filter", desc: "Tier by confidence level" },
  { label: "Publish", desc: "Pre-kickoff, fully tracked" },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero */}
      <div className="text-center mb-16">
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
          <span className="gradient-text">The Edge is Transparency</span>
        </h1>
        <p className="text-lg text-text-secondary max-w-2xl mx-auto">
          A complete look inside our prediction engine. No black boxes.
        </p>
        {/* Animated counter line */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3 text-sm text-text-secondary">
          <span className="px-3 py-1.5 bg-bg-surface border border-border">
            <span className="text-accent-primary font-bold">15</span> Sources
          </span>
          <span className="text-accent-primary">&times;</span>
          <span className="px-3 py-1.5 bg-bg-surface border border-border">
            <span className="text-accent-primary font-bold">50</span> Features
          </span>
          <span className="text-accent-primary">&times;</span>
          <span className="px-3 py-1.5 bg-bg-surface border border-border">
            <span className="text-accent-primary font-bold">5</span> Leagues
          </span>
          <span className="text-accent-primary">=</span>
          <span className="px-3 py-1.5 bg-accent-primary/10 border border-accent-primary/30 text-accent-primary font-semibold">
            Your Edge
          </span>
        </div>
      </div>

      {/* Data Sources */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-1 h-6 bg-accent-primary" />
          <h2 className="text-xl font-bold text-text-primary">15 Data Sources, Scraped Daily</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {DATA_SOURCES.map((source) => (
            <div
              key={source.name}
              className="group bg-bg-surface border border-border p-4 hover:border-accent-primary/20 hover:-translate-y-0.5 transition-all duration-200"
            >
              <div className="flex items-center gap-3">
                <span className="text-lg">{source.icon}</span>
                <div>
                  <p className="text-sm font-medium text-text-primary group-hover:text-accent-primary transition-colors">
                    {source.name}
                  </p>
                  <p className="text-xs text-text-secondary">{source.type}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Pipeline */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-1 h-6 bg-accent-secondary" />
          <h2 className="text-xl font-bold text-text-primary">Prediction Pipeline</h2>
        </div>
        <div className="flex flex-col sm:flex-row gap-3">
          {PIPELINE_STEPS.map((step, i) => (
            <div key={step.label} className="flex-1 relative">
              <div className="glass-card p-4 text-center h-full">
                <div className="text-xs text-accent-primary font-bold mb-1 tabular-nums">
                  {String(i + 1).padStart(2, "0")}
                </div>
                <p className="text-sm font-semibold text-text-primary mb-1">{step.label}</p>
                <p className="text-xs text-text-secondary">{step.desc}</p>
              </div>
              {i < PIPELINE_STEPS.length - 1 && (
                <div className="hidden sm:block absolute top-1/2 -right-2 transform -translate-y-1/2 z-10 text-text-secondary/30">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-1 h-6 bg-accent-primary" />
          <h2 className="text-xl font-bold text-text-primary">50-Feature ML Model</h2>
        </div>
        <p className="text-sm text-text-secondary mb-6">
          Each match is described by 50 engineered features across 7 groups:
        </p>
        <div className="space-y-2">
          {FEATURE_GROUPS.map((group) => (
            <div
              key={group.name}
              className="bg-bg-surface border border-border p-4 flex items-center justify-between hover:border-accent-primary/20 transition-colors group"
            >
              <div>
                <p className="text-sm font-medium text-text-primary group-hover:text-accent-primary transition-colors">
                  {group.name}
                </p>
                <p className="text-xs text-text-secondary">{group.desc}</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-accent-primary font-bold tabular-nums">
                  {group.count}
                </span>
                {/* Mini bar */}
                <div className="w-16 h-1.5 bg-bg-elevated overflow-hidden hidden sm:block">
                  <div
                    className="h-full bg-accent-primary/60"
                    style={{ width: `${(group.count / 12) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Validation */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-1 h-6 bg-accent-secondary" />
          <h2 className="text-xl font-bold text-text-primary">Walk-Forward Validation</h2>
        </div>
        <div className="glass-card p-6 space-y-3 text-sm text-text-secondary">
          <p>
            We use <strong className="text-text-primary">walk-forward validation</strong> &mdash; the model is
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

      {/* Tier System */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-1 h-6 bg-accent-primary" />
          <h2 className="text-xl font-bold text-text-primary">Tier System</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-bg-surface border-l-4 border-l-tier-platinum border border-border p-6 tier-platinum">
            <p className="text-lg font-bold text-tier-platinum mb-1">Platinum</p>
            <p className="text-xs text-text-secondary mb-4">
              Confidence &ge; 75%
            </p>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-tier-platinum">82.1%</span>
              <span className="text-xs text-text-secondary">accuracy</span>
            </div>
            <p className="text-[10px] text-text-secondary/60 mt-1">Walk-forward, 3504 matches</p>
          </div>
          <div className="bg-bg-surface border-l-4 border-l-tier-gold border border-border p-6 tier-gold">
            <p className="text-lg font-bold text-tier-gold mb-1">Gold</p>
            <p className="text-xs text-text-secondary mb-4">
              Confidence 65&ndash;75%
            </p>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-tier-gold">~76%</span>
              <span className="text-xs text-text-secondary">accuracy</span>
            </div>
          </div>
          <div className="bg-bg-surface border-l-4 border-l-tier-silver border border-border p-6 tier-silver">
            <p className="text-lg font-bold text-tier-silver mb-1">Silver</p>
            <p className="text-xs text-text-secondary mb-4">
              Confidence 50&ndash;65%
            </p>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-tier-silver">~68%</span>
              <span className="text-xs text-text-secondary">accuracy</span>
            </div>
          </div>
        </div>
      </section>

      {/* No Cherry-Picking */}
      <section>
        <div className="glass-card p-8 text-center">
          <h2 className="text-xl font-bold text-text-primary mb-3">
            No Cherry-Picking
          </h2>
          <p className="text-sm text-text-secondary max-w-lg mx-auto">
            Every pick is published before kick-off and tracked automatically.
            Results are resolved against actual scores. Our{" "}
            <a href="/track-record" className="text-accent-primary hover:underline font-medium">
              track record
            </a>{" "}
            is fully transparent and auditable.
          </p>
        </div>
      </section>
    </div>
  );
}
