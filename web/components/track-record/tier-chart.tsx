"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from "recharts";
import type { TierStats } from "@/lib/types";

interface TierChartProps {
  byTier: Record<string, TierStats>;
}

const TIER_COLORS: Record<string, string> = {
  Platinum: "#e5e7eb",
  Gold: "#f59e0b",
  Silver: "#9ca3af",
};

export function TierChart({ byTier }: TierChartProps) {
  const data = Object.entries(byTier).map(([tier, stats]) => ({
    tier,
    winRate: Math.round(stats.win_rate * 100),
    picks: stats.total_picks,
  }));

  return (
    <div className="glass-card p-6">
      <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
        <span className="w-1 h-4 bg-accent-secondary inline-block" />
        Accuracy by Tier
      </h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <XAxis
              dataKey="tier"
              tick={{ fill: "#f1f5f9", fontSize: 12 }}
              axisLine={{ stroke: "#1a1a2e" }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "#94a3b8", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#12121a",
                border: "1px solid rgba(255,255,255,0.06)",
                backdropFilter: "blur(16px)",
              }}
              labelStyle={{ color: "#f1f5f9" }}
              formatter={(value: number | undefined) => `${value ?? 0}%`}
            />
            <Bar dataKey="winRate" name="Win Rate" radius={[2, 2, 0, 0]}>
              {data.map((entry) => (
                <Cell key={entry.tier} fill={TIER_COLORS[entry.tier] || "#6366f1"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
