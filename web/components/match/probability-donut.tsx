"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from "recharts";

interface ProbabilityDonutProps {
  home: number;
  draw: number;
  away: number;
  homeTeam: string;
  awayTeam: string;
}

const COLORS = ["#00d4aa", "#6366f1", "#94a3b8"];

export function ProbabilityDonut({ home, draw, away, homeTeam, awayTeam }: ProbabilityDonutProps) {
  const data = [
    { name: homeTeam, value: Math.round(home * 100) },
    { name: "Draw", value: Math.round(draw * 100) },
    { name: awayTeam, value: Math.round(away * 100) },
  ];

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius={60} outerRadius={90} dataKey="value" stroke="none">
            {data.map((_, index) => (
              <Cell key={index} fill={COLORS[index]} />
            ))}
          </Pie>
          <Legend
            formatter={(value, entry) => (
              <span className="text-text-primary text-sm">
                {value}: {(entry.payload as Record<string, unknown>)?.value}%
              </span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
