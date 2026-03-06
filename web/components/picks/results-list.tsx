import { cn, formatOdds, formatProfit } from "@/lib/utils";
import type { Pick } from "@/lib/types";

interface ResultsListProps {
  picks: Pick[];
}

export function ResultsList({ picks }: ResultsListProps) {
  if (picks.length === 0) return null;

  return (
    <div className="bg-bg-surface border border-border p-6">
      <h3 className="text-lg font-semibold text-text-primary mb-4">Recent Results</h3>
      <div className="space-y-3">
        {picks.map((pick, i) => (
          <div key={i} className="flex items-center justify-between py-2 border-b border-border last:border-0">
            <div className="flex items-center gap-3">
              <span className={cn(
                "w-6 h-6 flex items-center justify-center text-xs font-bold",
                pick.result === "win" ? "bg-success/10 text-success" : "bg-danger/10 text-danger"
              )}>
                {pick.result === "win" ? "W" : "L"}
              </span>
              <div>
                <p className="text-sm text-text-primary">{pick.home_team} vs {pick.away_team}</p>
                <p className="text-xs text-text-secondary">{pick.pick_selection} @ {formatOdds(pick.best_odds)}</p>
              </div>
            </div>
            <span className={cn(
              "text-sm font-medium tabular-nums",
              (pick.profit_loss ?? 0) >= 0 ? "text-success" : "text-danger"
            )}>
              {formatProfit(pick.profit_loss)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
