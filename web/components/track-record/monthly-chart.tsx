"use client";

import {
  AreaChart,
  Area,
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
    <div className="glass-card p-6">
      <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
        <span className="w-1 h-4 bg-accent-primary inline-block" />
        Cumulative Profit (units)
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4aa" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#00d4aa" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="month"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickFormatter={(v: string) => v.slice(5)}
              axisLine={{ stroke: "#1a1a2e" }}
              tickLine={false}
            />
            <YAxis
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
              formatter={(value: number | undefined) => [
                `${(value ?? 0).toFixed(2)}u`,
                "Profit",
              ]}
            />
            <ReferenceLine y={0} stroke="#1a1a2e" />
            <Area
              type="monotone"
              dataKey="cumulative_profit"
              stroke="#00d4aa"
              strokeWidth={2}
              fill="url(#profitGradient)"
              dot={false}
              activeDot={{ r: 4, fill: "#00d4aa", stroke: "#0a0a0f", strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
