import { cn, formatPercent, formatProfit } from "@/lib/utils";
import type { TierStats } from "@/lib/types";

interface LeagueTableProps {
  byLeague: Record<string, TierStats>;
}

export function LeagueTable({ byLeague }: LeagueTableProps) {
  const leagues = Object.entries(byLeague)
    .map(([name, stats]) => ({ name, ...stats }))
    .sort((a, b) => b.total_picks - a.total_picks);

  return (
    <div className="bg-bg-surface border border-border overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-text-secondary border-b border-border">
            <th className="text-left py-3 px-4">League</th>
            <th className="text-right py-3 px-4">W-L</th>
            <th className="text-right py-3 px-4">Win Rate</th>
            <th className="text-right py-3 px-4">Profit</th>
            <th className="text-right py-3 px-4">ROI</th>
          </tr>
        </thead>
        <tbody>
          {leagues.map((league) => (
            <tr key={league.name} className="border-b border-border">
              <td className="py-3 px-4 text-text-primary font-medium">
                {league.name}
              </td>
              <td className="py-3 px-4 text-right tabular-nums text-text-primary">
                {league.wins}-{league.losses}
              </td>
              <td className="py-3 px-4 text-right tabular-nums text-text-primary">
                {formatPercent(league.win_rate)}
              </td>
              <td
                className={cn(
                  "py-3 px-4 text-right tabular-nums font-medium",
                  league.total_profit >= 0 ? "text-success" : "text-danger"
                )}
              >
                {formatProfit(league.total_profit)}
              </td>
              <td
                className={cn(
                  "py-3 px-4 text-right tabular-nums",
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
