"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { PickCard } from "./pick-card";
import type { Pick } from "@/lib/types";

const TIERS = ["All", "Platinum", "Gold", "Silver"] as const;

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
      <div className="flex gap-1 mb-6">
        {TIERS.map((tier) => (
          <button
            key={tier}
            onClick={() => setActiveTier(tier)}
            className={cn(
              "px-4 py-2 text-sm font-medium transition-colors border",
              activeTier === tier
                ? "bg-accent-primary/10 border-accent-primary/30 text-accent-primary"
                : "bg-bg-surface border-border text-text-secondary hover:text-text-primary"
            )}
          >
            {tier}
            <span className="ml-2 text-xs opacity-60">{tierCounts[tier]}</span>
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
        <div className="text-center py-12 text-text-secondary">
          No {activeTier} picks today.
        </div>
      )}
    </>
  );
}
