import Link from "next/link";
import { TierBadge } from "./tier-badge";
import { ConfidenceBar } from "./confidence-bar";
import { LeagueBadge } from "@/components/shared/league-badge";
import { formatOdds, formatEdge, makeMatchSlug } from "@/lib/utils";
import type { Pick } from "@/lib/types";

interface PickCardProps {
  pick: Pick;
}

export function PickCard({ pick }: PickCardProps) {
  const slug = makeMatchSlug(pick.home_team, pick.away_team, pick.match_date);

  return (
    <Link
      href={`/match/${slug}`}
      className="block bg-bg-surface border border-border hover:border-accent-primary/30 transition-colors p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <TierBadge tier={pick.tier} />
        <LeagueBadge league={pick.league} />
      </div>
      <div className="text-center mb-4">
        <p className="text-lg font-semibold text-text-primary">{pick.home_team}</p>
        <p className="text-xs text-text-secondary my-1">vs</p>
        <p className="text-lg font-semibold text-text-primary">{pick.away_team}</p>
      </div>
      <div className="text-center mb-4">
        <span className="inline-block bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-3 py-1 text-sm font-medium">
          {pick.pick_selection}
        </span>
      </div>
      <ConfidenceBar value={pick.model_prob} label="Confidence" />
      <div className="flex justify-between mt-4 text-xs">
        <div>
          <span className="text-text-secondary">Odds: </span>
          <span className="text-text-primary font-medium tabular-nums">{formatOdds(pick.best_odds)}</span>
        </div>
        <div>
          <span className="text-text-secondary">Edge: </span>
          <span className="text-success font-medium tabular-nums">{formatEdge(pick.edge)}</span>
        </div>
        {pick.meta_agreement > 0 && (
          <div>
            <span className="text-text-secondary">Agree: </span>
            <span className="text-text-primary font-medium">{pick.meta_agreement}/6</span>
          </div>
        )}
      </div>
    </Link>
  );
}
