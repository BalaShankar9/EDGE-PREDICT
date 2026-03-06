import { cn } from "@/lib/utils";
import type { Pick } from "@/lib/types";

interface StreakIndicatorProps {
  picks: Pick[];
  className?: string;
}

export function StreakIndicator({ picks, className }: StreakIndicatorProps) {
  const resolvedPicks = picks.filter((p) => p.result === "win" || p.result === "loss");
  if (resolvedPicks.length === 0) return null;

  // Calculate current streak
  let streakType = resolvedPicks[0]?.result;
  let streakCount = 0;
  for (const pick of resolvedPicks) {
    if (pick.result === streakType) {
      streakCount++;
    } else {
      break;
    }
  }

  const isWinStreak = streakType === "win";

  return (
    <div className={cn("flex items-center gap-4", className)}>
      {/* Current streak badge */}
      <div
        className={cn(
          "flex items-center gap-2 px-4 py-2 border text-sm font-semibold",
          isWinStreak
            ? "bg-success/10 border-success/30 text-success"
            : "bg-danger/10 border-danger/30 text-danger"
        )}
      >
        {isWinStreak && <span className="text-base">&#128293;</span>}
        <span>
          {streakCount}
          {isWinStreak ? "W" : "L"} streak
        </span>
      </div>

      {/* Result dots */}
      <div className="flex items-center gap-1.5">
        {resolvedPicks.slice(0, 20).map((pick, i) => (
          <div
            key={i}
            className="group relative"
          >
            <div
              className={cn(
                "w-3 h-3 transition-transform hover:scale-150",
                pick.result === "win" ? "bg-success" : "bg-danger"
              )}
              title={`${pick.home_team} vs ${pick.away_team}`}
            />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-bg-elevated border border-border text-xs text-text-primary whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
              {pick.home_team} vs {pick.away_team}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
