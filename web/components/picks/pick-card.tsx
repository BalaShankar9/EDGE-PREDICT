import Link from "next/link";
import { TierBadge } from "./tier-badge";
import { ConfidenceBar } from "./confidence-bar";
import { LeagueBadge } from "@/components/shared/league-badge";
import { cn, formatOdds, formatEdge, makeMatchSlug } from "@/lib/utils";
import type { Pick } from "@/lib/types";

interface PickCardProps {
  pick: Pick;
}

const tierBorderColor: Record<string, string> = {
  Platinum: "border-l-tier-platinum",
  Gold: "border-l-tier-gold",
  Silver: "border-l-tier-silver",
};

const tierGlowClass: Record<string, string> = {
  Platinum: "tier-platinum",
  Gold: "tier-gold",
  Silver: "tier-silver",
};

export function PickCard({ pick }: PickCardProps) {
  const slug = makeMatchSlug(pick.home_team, pick.away_team, pick.match_date);
  const hasResult = pick.result === "win" || pick.result === "loss";
  const isWin = pick.result === "win";

  return (
    <Link
      href={`/match/${slug}`}
      className={cn(
        "group relative block bg-bg-surface border border-border border-l-4 p-5 transition-all duration-300",
        "hover:-translate-y-0.5 hover:border-accent-primary/20",
        tierBorderColor[pick.tier] || "border-l-border",
        tierGlowClass[pick.tier]
      )}
    >
      {/* Result overlay badge */}
      {hasResult && (
        <div className={cn(
          "absolute top-3 right-3 w-8 h-8 flex items-center justify-center text-xs font-bold z-10",
          isWin ? "bg-success/15 text-success border border-success/30" : "bg-danger/15 text-danger border border-danger/30"
        )}>
          {isWin ? "W" : "L"}
        </div>
      )}

      {/* Top: tier + league */}
      <div className="flex items-center justify-between mb-4">
        <TierBadge tier={pick.tier} />
        <LeagueBadge league={pick.league} />
      </div>

      {/* Center: teams */}
      <div className="text-center mb-4">
        <p className="text-lg font-bold text-text-primary leading-tight group-hover:text-accent-primary transition-colors">
          {pick.home_team}
        </p>
        <div className="flex items-center justify-center gap-2 my-2">
          <span className="h-px w-6 bg-border" />
          <span className="text-xs text-text-secondary font-medium tracking-wider uppercase">vs</span>
          <span className="h-px w-6 bg-border" />
        </div>
        <p className="text-lg font-bold text-text-primary leading-tight group-hover:text-accent-primary transition-colors">
          {pick.away_team}
        </p>
      </div>

      {/* Prediction badge */}
      <div className="text-center mb-4">
        <span className="inline-flex items-center gap-2 bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-4 py-1.5 text-sm font-semibold">
          {pick.pick_selection}
          <span className="text-xs opacity-70 tabular-nums">{(pick.model_prob * 100).toFixed(0)}%</span>
        </span>
      </div>

      {/* Confidence bar */}
      <ConfidenceBar value={pick.model_prob} label="Confidence" />

      {/* Bottom stats row */}
      <div className="flex justify-between mt-4 gap-2">
        <div className="flex-1 bg-bg-elevated/50 px-3 py-2 text-center">
          <p className="text-[10px] text-text-secondary uppercase tracking-wider mb-0.5">Odds</p>
          <p className="text-sm text-text-primary font-semibold tabular-nums">{formatOdds(pick.best_odds)}</p>
        </div>
        <div className="flex-1 bg-bg-elevated/50 px-3 py-2 text-center">
          <p className="text-[10px] text-text-secondary uppercase tracking-wider mb-0.5">Edge</p>
          <p className="text-sm text-success font-semibold tabular-nums">{formatEdge(pick.edge)}</p>
        </div>
        {pick.meta_agreement > 0 && (
          <div className="flex-1 bg-bg-elevated/50 px-3 py-2 text-center">
            <p className="text-[10px] text-text-secondary uppercase tracking-wider mb-0.5">Agree</p>
            <p className="text-sm text-text-primary font-semibold">{pick.meta_agreement}/6</p>
          </div>
        )}
      </div>
    </Link>
  );
}
