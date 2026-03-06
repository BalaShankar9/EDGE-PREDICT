import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { TierBadge } from "@/components/picks/tier-badge";
import { ConfidenceBar } from "@/components/picks/confidence-bar";
import { ProbabilityDonut } from "@/components/match/probability-donut";
import { FeatureFactors } from "@/components/match/feature-factors";
import { OddsTable } from "@/components/match/odds-table";
import { getMatchDetail } from "@/lib/api";
import { parseMatchSlug, formatDate } from "@/lib/utils";

export const revalidate = 300;

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const parsed = parseMatchSlug(slug);
  if (!parsed) return { title: "Match Not Found" };
  return {
    title: `${parsed.homeTeam} vs ${parsed.awayTeam} Prediction`,
    description: `AI prediction for ${parsed.homeTeam} vs ${parsed.awayTeam} on ${formatDate(parsed.matchDate)}.`,
  };
}

export default async function MatchDetailPage({ params }: Props) {
  const { slug } = await params;
  const parsed = parseMatchSlug(slug);
  if (!parsed) notFound();

  let match;
  try {
    match = await getMatchDetail(parsed.homeTeam, parsed.awayTeam, parsed.matchDate);
  } catch {
    notFound();
  }

  const probs = match.probabilities;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="text-center mb-8">
        <p className="text-sm text-text-secondary mb-2">
          {match.league} &middot; {formatDate(match.match_date)}
        </p>
        <h1 className="text-3xl font-bold text-text-primary">
          {match.home_team} vs {match.away_team}
        </h1>
        {match.pick && (
          <div className="mt-4 flex items-center justify-center gap-3">
            <TierBadge tier={match.pick.tier} />
            <span className="bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-3 py-1 text-sm font-medium">
              {match.pick.selection}
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3 space-y-6">
          <div className="bg-bg-surface border border-border p-6">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Match Probabilities</h3>
            <ProbabilityDonut home={probs.home} draw={probs.draw} away={probs.away} homeTeam={match.home_team} awayTeam={match.away_team} />
          </div>

          {match.pick && (
            <div className="bg-bg-surface border border-border p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4">AI Confidence</h3>
              <ConfidenceBar value={match.pick.model_prob} label={match.pick.selection} />
              {match.pick.meta_agreement > 0 && (
                <p className="text-xs text-text-secondary mt-2">{match.pick.meta_agreement}/6 prediction sources agree</p>
              )}
            </div>
          )}

          <FeatureFactors xgboostProbs={match.xgboost_probs} poissonProbs={match.poisson_probs} ensembleWeights={match.ensemble_weights} />
        </div>

        <div className="lg:col-span-2 space-y-6">
          <OddsTable bestOdds={match.pick?.best_odds} bookmaker={match.pick?.bookmaker} edge={match.pick?.edge} probHome={probs.home} probDraw={probs.draw} probAway={probs.away} />

          {(probs.over_25 != null || probs.btts_yes != null) && (
            <div className="bg-bg-surface border border-border p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4">Markets</h3>
              <div className="space-y-3">
                {probs.over_25 != null && <ConfidenceBar value={probs.over_25} label="Over 2.5 Goals" />}
                {probs.btts_yes != null && <ConfidenceBar value={probs.btts_yes} label="BTTS Yes" />}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
