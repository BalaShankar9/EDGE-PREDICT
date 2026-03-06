"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { PickCard } from "./pick-card";
import type { Pick } from "@/lib/types";

const TIERS = ["All", "Platinum", "Gold", "Silver"] as const;

const tierActiveStyles: Record<string, string> = {
  All: "bg-accent-primary/10 border-accent-primary/40 text-accent-primary shadow-[0_0_12px_rgba(0,212,170,0.15)]",
  Platinum: "bg-tier-platinum/10 border-tier-platinum/40 text-tier-platinum shadow-[0_0_12px_rgba(229,231,235,0.12)]",
  Gold: "bg-tier-gold/10 border-tier-gold/40 text-tier-gold shadow-[0_0_12px_rgba(245,158,11,0.15)]",
  Silver: "bg-tier-silver/10 border-tier-silver/40 text-tier-silver shadow-[0_0_12px_rgba(156,163,175,0.12)]",
};

interface TierFilterProps {
  picks: Pick[];
}

export function TierFilter({ picks }: TierFilterProps) {
  const [activeTier, setActiveTier] = useState<string>("All");

  const filteredPicks =
    activeTier === "All" ? picks : picks.filter((p) => p.tier === activeTier);

  const tierCounts = {
    All: picks.length,
    Platinum: picks.filter((p) => p.tier === "Platinum").length,
    Gold: picks.filter((p) => p.tier === "Gold").length,
    Silver: picks.filter((p) => p.tier === "Silver").length,
  };

  return (
    <>
      <div className="flex gap-2 mb-6 flex-wrap">
        {TIERS.map((tier) => (
          <button
            key={tier}
            onClick={() => setActiveTier(tier)}
            className={cn(
              "px-4 py-2 text-sm font-medium transition-all duration-200 border",
              activeTier === tier
                ? tierActiveStyles[tier]
                : "bg-bg-surface border-border text-text-secondary hover:text-text-primary hover:border-border/80"
            )}
          >
            {tier}
            <span className="ml-2 text-xs opacity-60 tabular-nums">{tierCounts[tier]}</span>
          </button>
        ))}
      </div>
      {filteredPicks.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredPicks.map((pick, i) => (
            <PickCard key={`${pick.home_team}-${pick.away_team}-${i}`} pick={pick} />
          ))}
        </div>
      ) : (
        <div className="text-center py-16">
          <p className="text-text-secondary">No {activeTier} picks today.</p>
          <p className="text-xs text-text-secondary/60 mt-1">Check back when new predictions are published.</p>
        </div>
      )}
    </>
  );
}
