"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { MonthlyRecord } from "@/lib/types";

interface MonthlyChartProps {
  data: MonthlyRecord[];
}

export function MonthlyChart({ data }: MonthlyChartProps) {
  return (
    <div className="bg-bg-surface border border-border p-6">
      <h3 className="text-sm font-semibold text-text-primary mb-4">
        Cumulative Profit (units)
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis
              dataKey="month"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickFormatter={(v: string) => v.slice(5)}
            />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#12121a", border: "1px solid #1a1a2e" }}
              labelStyle={{ color: "#f1f5f9" }}
              formatter={(value: number | undefined) => [`${(value ?? 0).toFixed(2)}u`, "Profit"]}
            />
            <ReferenceLine y={0} stroke="#1a1a2e" />
            <Line
              type="monotone"
              dataKey="cumulative_profit"
              stroke="#00d4aa"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
