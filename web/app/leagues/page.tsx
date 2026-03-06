import type { Metadata } from "next";
import Link from "next/link";
import { getLeagues } from "@/lib/api";
import type { League } from "@/lib/types";

export const metadata: Metadata = {
  title: "Leagues",
  description: "Football predictions by league — Premier League, La Liga, Bundesliga, Serie A, Ligue 1.",
};

export const revalidate = 300;

export default async function LeaguesPage() {
  let leagues: League[] = [];
  try {
    leagues = await getLeagues();
  } catch {}

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold text-text-primary mb-6">Leagues</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {leagues.map((league) => (
          <Link key={league.slug} href={`/league/${league.slug}`}
            className="bg-bg-surface border border-border hover:border-accent-primary/30 transition-colors p-6">
            <h2 className="text-lg font-semibold text-text-primary mb-2">{league.name}</h2>
            <div className="flex gap-4 text-sm">
              <span className="text-text-secondary">{league.prediction_count} predictions</span>
              {league.pick_count > 0 && (
                <span className="text-accent-primary font-medium">{league.pick_count} picks</span>
              )}
            </div>
          </Link>
        ))}
      </div>
      {leagues.length === 0 && (
        <p className="text-center py-12 text-text-secondary">No leagues with upcoming predictions.</p>
      )}
    </div>
  );
}
