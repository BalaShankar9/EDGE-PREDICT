import { StatCard } from "@/components/shared/stat-card";
import { formatPercent, formatProfit } from "@/lib/utils";
import type { TrackRecord } from "@/lib/types";

interface StatsHeroProps {
  record: TrackRecord;
}

export function StatsHero({ record }: StatsHeroProps) {
  const o = record.overall ?? { total_picks: 0, wins: 0, losses: 0, win_rate: 0, total_profit: 0, roi: 0, max_drawdown: 0, avg_odds: 0, longest_win_streak: 0, longest_loss_streak: 0 };
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
      <StatCard label="Total Picks" value={String(o.total_picks)} glow />
      <StatCard
        label="Win Rate"
        value={formatPercent(o.win_rate)}
        glow
        trend={o.win_rate > 0.6 ? "up" : o.win_rate < 0.5 ? "down" : "neutral"}
      />
      <StatCard
        label="ROI"
        value={`${o.roi > 0 ? "+" : ""}${o.roi.toFixed(1)}%`}
        trend={o.roi > 0 ? "up" : o.roi < 0 ? "down" : "neutral"}
      />
      <StatCard label="Profit" value={formatProfit(o.total_profit)} />
      <StatCard label="Avg Odds" value={o.avg_odds.toFixed(2)} />
    </div>
  );
}
