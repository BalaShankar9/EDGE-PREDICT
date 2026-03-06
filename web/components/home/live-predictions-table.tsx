import Link from "next/link";
import { makeMatchSlug } from "@/lib/utils";
import type { Pick } from "@/lib/types";

interface LivePredictionsTableProps {
  picks: Pick[];
}

export function LivePredictionsTable({ picks }: LivePredictionsTableProps) {
  if (picks.length === 0) return null;

  return (
    <div className="glass-card overflow-hidden">
      <div className="px-5 py-4 border-b border-white/5">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full bg-accent-primary opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 bg-accent-primary" />
          </span>
          <h3 className="text-sm font-semibold text-text-primary">Live Predictions</h3>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-text-secondary text-xs">
              <th className="text-left py-3 px-5 font-medium">Home</th>
              <th className="text-left py-3 px-5 font-medium">Away</th>
              <th className="text-left py-3 px-5 font-medium">Our Pick</th>
              <th className="text-right py-3 px-5 font-medium">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {picks.map((pick, i) => {
              const slug = makeMatchSlug(pick.home_team, pick.away_team, pick.match_date);
              return (
                <tr
                  key={i}
                  className="border-t border-white/5 hover:bg-white/[0.02] transition-colors"
                >
                  <td className="py-3 px-5">
                    <Link href={`/match/${slug}`} className="text-text-primary hover:text-accent-primary transition-colors font-medium">
                      {pick.home_team}
                    </Link>
                  </td>
                  <td className="py-3 px-5">
                    <Link href={`/match/${slug}`} className="text-text-primary hover:text-accent-primary transition-colors font-medium">
                      {pick.away_team}
                    </Link>
                  </td>
                  <td className="py-3 px-5">
                    <span className="inline-flex items-center px-2.5 py-0.5 text-xs font-semibold bg-accent-primary/10 text-accent-primary border border-accent-primary/20">
                      {pick.pick_selection}
                    </span>
                  </td>
                  <td className="py-3 px-5 text-right">
                    <span className="text-text-primary font-medium tabular-nums">
                      {(pick.model_prob * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
