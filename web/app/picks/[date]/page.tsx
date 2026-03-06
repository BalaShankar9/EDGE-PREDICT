import type { Metadata } from "next";
import { TierFilter } from "@/components/picks/tier-filter";
import { getPicksToday } from "@/lib/api";
import { formatDate } from "@/lib/utils";

export const revalidate = 300;

interface Props {
  params: Promise<{ date: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { date } = await params;
  return {
    title: `Picks for ${formatDate(date)}`,
    description: `SharpEdge AI football predictions for ${formatDate(date)}.`,
  };
}

export default async function PicksByDatePage({ params }: Props) {
  const { date } = await params;
  let picks: any[] = [];

  try {
    const today = new Date().toISOString().split("T")[0];
    if (date === today) {
      picks = await getPicksToday();
    }
  } catch {
    // Graceful degradation
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold text-text-primary mb-6">
        Picks — {formatDate(date)}
      </h1>
      <TierFilter picks={picks} />
    </div>
  );
}
