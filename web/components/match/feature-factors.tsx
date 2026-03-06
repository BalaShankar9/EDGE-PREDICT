"use client";

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

interface FeatureFactorsProps {
  xgboostProbs: Record<string, number> | null;
  poissonProbs: Record<string, number> | null;
  ensembleWeights: Record<string, number> | null;
}

export function FeatureFactors({ xgboostProbs, poissonProbs, ensembleWeights }: FeatureFactorsProps) {
  if (!xgboostProbs && !poissonProbs) return null;

  const data = [
    {
      name: "XGBoost",
      home: Math.round((xgboostProbs?.home ?? 0) * 100),
      draw: Math.round((xgboostProbs?.draw ?? 0) * 100),
      away: Math.round((xgboostProbs?.away ?? 0) * 100),
    },
    {
      name: "Poisson",
      home: Math.round((poissonProbs?.home ?? 0) * 100),
      draw: Math.round((poissonProbs?.draw ?? 0) * 100),
      away: Math.round((poissonProbs?.away ?? 0) * 100),
    },
  ];

  return (
    <div className="bg-bg-surface border border-border p-6">
      <h3 className="text-sm font-semibold text-text-primary mb-4">Model Components</h3>
      {ensembleWeights && (
        <p className="text-xs text-text-secondary mb-3">
          Weights: XGBoost {Math.round((ensembleWeights.xgboost ?? 0) * 100)}% / Poisson {Math.round((ensembleWeights.poisson ?? 0) * 100)}%
        </p>
      )}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical">
            <XAxis type="number" domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <YAxis type="category" dataKey="name" width={60} tick={{ fill: "#f1f5f9", fontSize: 12 }} />
            <Tooltip contentStyle={{ backgroundColor: "#12121a", border: "1px solid #1a1a2e" }} labelStyle={{ color: "#f1f5f9" }} />
            <Bar dataKey="home" fill="#00d4aa" name="Home" />
            <Bar dataKey="draw" fill="#6366f1" name="Draw" />
            <Bar dataKey="away" fill="#94a3b8" name="Away" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
