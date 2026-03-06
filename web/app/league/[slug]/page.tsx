import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { TierBadge } from "@/components/picks/tier-badge";
import { getLeaguePredictions } from "@/lib/api";
import { formatDate, makeMatchSlug } from "@/lib/utils";

export const revalidate = 300;

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const name = slug.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  return {
    title: `${name} Predictions`,
    description: `AI predictions for upcoming ${name} matches.`,
  };
}

export default async function LeagueDetailPage({ params }: Props) {
  const { slug } = await params;
  let data;
  try {
    data = await getLeaguePredictions(slug);
  } catch {
    notFound();
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold text-text-primary mb-6">{data.league}</h1>
      <div className="bg-bg-surface border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-text-secondary border-b border-border">
              <th className="text-left py-3 px-4">Date</th>
              <th className="text-left py-3 px-4">Match</th>
              <th className="text-right py-3 px-4">Home</th>
              <th className="text-right py-3 px-4">Draw</th>
              <th className="text-right py-3 px-4">Away</th>
              <th className="text-center py-3 px-4">Pick</th>
            </tr>
          </thead>
          <tbody>
            {data.predictions.map((pred, i) => {
              const matchSlug = makeMatchSlug(pred.home_team, pred.away_team, pred.match_date);
              return (
                <tr key={i} className="border-b border-border hover:bg-bg-elevated transition-colors">
                  <td className="py-3 px-4 text-text-secondary">{formatDate(pred.match_date)}</td>
                  <td className="py-3 px-4">
                    <Link href={`/match/${matchSlug}`} className="text-text-primary hover:text-accent-primary">
                      {pred.home_team} vs {pred.away_team}
                    </Link>
                  </td>
                  <td className="py-3 px-4 text-right tabular-nums text-text-primary">{(pred.prob_home * 100).toFixed(0)}%</td>
                  <td className="py-3 px-4 text-right tabular-nums text-text-primary">{(pred.prob_draw * 100).toFixed(0)}%</td>
                  <td className="py-3 px-4 text-right tabular-nums text-text-primary">{(pred.prob_away * 100).toFixed(0)}%</td>
                  <td className="py-3 px-4 text-center">
                    {pred.pick ? (
                      <div className="flex items-center justify-center gap-2">
                        <TierBadge tier={pred.pick.tier} />
                        <span className="text-xs text-accent-primary">{pred.pick.selection}</span>
                      </div>
                    ) : (
                      <span className="text-text-secondary text-xs">—</span>
                    )}
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
