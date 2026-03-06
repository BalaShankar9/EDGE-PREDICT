import { cn, formatOdds, formatProfit } from "@/lib/utils";
import { StreakIndicator } from "@/components/shared/streak-indicator";
import type { Pick } from "@/lib/types";

interface ResultsListProps {
  picks: Pick[];
}

export function ResultsList({ picks }: ResultsListProps) {
  if (picks.length === 0) return null;

  return (
    <div className="space-y-6">
      {/* Streak visualizer */}
      <StreakIndicator picks={picks} />

      {/* Results list */}
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
          <span className="w-1 h-5 bg-accent-primary inline-block" />
          Recent Results
        </h3>
        <div className="space-y-2">
          {picks.map((pick, i) => (
            <div key={i} className="flex items-center justify-between py-2.5 border-b border-white/5 last:border-0 hover:bg-white/[0.02] transition-colors px-2 -mx-2">
              <div className="flex items-center gap-3">
                <span className={cn(
                  "w-7 h-7 flex items-center justify-center text-xs font-bold shrink-0",
                  pick.result === "win"
                    ? "bg-success/10 text-success border border-success/20"
                    : "bg-danger/10 text-danger border border-danger/20"
                )}>
                  {pick.result === "win" ? "W" : "L"}
                </span>
                <div>
                  <p className="text-sm text-text-primary font-medium">{pick.home_team} vs {pick.away_team}</p>
                  <p className="text-xs text-text-secondary">{pick.pick_selection} @ {formatOdds(pick.best_odds)}</p>
                </div>
              </div>
              <span className={cn(
                "text-sm font-semibold tabular-nums",
                (pick.profit_loss ?? 0) >= 0 ? "text-success" : "text-danger"
              )}>
                {formatProfit(pick.profit_loss)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
