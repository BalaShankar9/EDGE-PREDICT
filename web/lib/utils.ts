import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatOdds(value: number | null | undefined): string {
  if (value == null) return "—";
  return value.toFixed(2);
}

export function formatEdge(value: number | null | undefined): string {
  if (value == null) return "—";
  const pct = value * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

export function formatProfit(value: number | null | undefined): string {
  if (value == null) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}u`;
}

export function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

export function makeMatchSlug(homeTeam: string, awayTeam: string, matchDate: string): string {
  const slugify = (s: string) =>
    s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  return `${slugify(homeTeam)}-vs-${slugify(awayTeam)}-${matchDate}`;
}

export function parseMatchSlug(slug: string): {
  homeTeam: string; awayTeam: string; matchDate: string;
} | null {
  const match = slug.match(/^(.+)-vs-(.+)-(\d{4}-\d{2}-\d{2})$/);
  if (!match) return null;
  return {
    homeTeam: match[1].replace(/-/g, " "),
    awayTeam: match[2].replace(/-/g, " "),
    matchDate: match[3],
  };
}

export function tierColor(tier: string): string {
  switch (tier) {
    case "Platinum": return "text-tier-platinum";
    case "Gold": return "text-tier-gold";
    case "Silver": return "text-tier-silver";
    default: return "text-text-secondary";
  }
}

export function tierBgColor(tier: string): string {
  switch (tier) {
    case "Platinum": return "bg-tier-platinum/10 border-tier-platinum/30";
    case "Gold": return "bg-tier-gold/10 border-tier-gold/30";
    case "Silver": return "bg-tier-silver/10 border-tier-silver/30";
    default: return "bg-bg-elevated border-border";
  }
}
