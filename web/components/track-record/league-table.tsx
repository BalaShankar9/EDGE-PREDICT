import { cn, formatPercent, formatProfit } from "@/lib/utils";
import type { TierStats } from "@/lib/types";

interface LeagueTableProps {
  byLeague: Record<string, TierStats>;
}

export function LeagueTable({ byLeague }: LeagueTableProps) {
  const leagues = Object.entries(byLeague)
    .map(([name, stats]) => ({ name, ...stats }))
    .sort((a, b) => b.total_picks - a.total_picks);

  const maxPicks = Math.max(...leagues.map((l) => l.total_picks), 1);

  return (
    <div className="glass-card overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-text-secondary border-b border-white/5">
            <th className="text-left py-3 px-5 font-medium">League</th>
            <th className="text-right py-3 px-5 font-medium">W-L</th>
            <th className="text-right py-3 px-5 font-medium">Win Rate</th>
            <th className="text-right py-3 px-5 font-medium">Profit</th>
            <th className="text-right py-3 px-5 font-medium">ROI</th>
          </tr>
        </thead>
        <tbody>
          {leagues.map((league) => (
            <tr key={league.name} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
              <td className="py-3 px-5">
                <div>
                  <span className="text-text-primary font-medium">{league.name}</span>
                  {/* Mini bar for relative volume */}
                  <div className="w-20 h-1 bg-bg-elevated mt-1.5 overflow-hidden">
                    <div
                      className="h-full bg-accent-primary/40"
                      style={{ width: `${(league.total_picks / maxPicks) * 100}%` }}
                    />
                  </div>
                </div>
              </td>
              <td className="py-3 px-5 text-right tabular-nums text-text-primary">
                {league.wins}-{league.losses}
              </td>
              <td className="py-3 px-5 text-right tabular-nums text-text-primary">
                {formatPercent(league.win_rate)}
              </td>
              <td
                className={cn(
                  "py-3 px-5 text-right tabular-nums font-medium",
                  league.total_profit >= 0 ? "text-success" : "text-danger"
                )}
              >
                {formatProfit(league.total_profit)}
              </td>
              <td
                className={cn(
                  "py-3 px-5 text-right tabular-nums",
                  league.roi >= 0 ? "text-success" : "text-danger"
                )}
              >
                {league.roi > 0 ? "+" : ""}{league.roi.toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
