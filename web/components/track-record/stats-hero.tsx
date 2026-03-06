import { StatCard } from "@/components/shared/stat-card";
import { formatPercent, formatProfit } from "@/lib/utils";
import type { TrackRecord } from "@/lib/types";

interface StatsHeroProps {
  record: TrackRecord;
}

export function StatsHero({ record }: StatsHeroProps) {
  const { overall } = record;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
      <StatCard label="Total Picks" value={String(overall.total_picks)} />
      <StatCard label="Win Rate" value={formatPercent(overall.win_rate)} />
      <StatCard
        label="ROI"
        value={`${overall.roi > 0 ? "+" : ""}${overall.roi.toFixed(1)}%`}
      />
      <StatCard label="Profit" value={formatProfit(overall.total_profit)} />
      <StatCard label="Avg Odds" value={overall.avg_odds.toFixed(2)} />
    </div>
  );
}
