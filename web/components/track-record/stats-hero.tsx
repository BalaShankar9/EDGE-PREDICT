import { StatCard } from "@/components/shared/stat-card";
import { formatPercent, formatProfit } from "@/lib/utils";
import type { TrackRecord } from "@/lib/types";

interface StatsHeroProps {
  record: TrackRecord;
}

export function StatsHero({ record }: StatsHeroProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
      <StatCard label="Total Picks" value={String(record.total_picks)} glow />
      <StatCard
        label="Win Rate"
        value={formatPercent(record.win_rate)}
        glow
        trend={record.win_rate > 0.6 ? "up" : record.win_rate < 0.5 ? "down" : "neutral"}
      />
      <StatCard
        label="ROI"
        value={`${record.roi > 0 ? "+" : ""}${record.roi.toFixed(1)}%`}
        trend={record.roi > 0 ? "up" : record.roi < 0 ? "down" : "neutral"}
      />
      <StatCard label="Profit" value={formatProfit(record.total_profit)} />
      <StatCard label="Avg Odds" value={record.avg_odds.toFixed(2)} />
    </div>
  );
}
