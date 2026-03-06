import { cn } from "@/lib/utils";

interface LeagueBadgeProps {
  league: string;
  className?: string;
}

export function LeagueBadge({ league, className }: LeagueBadgeProps) {
  return (
    <span className={cn("text-xs text-text-secondary font-medium", className)}>
      {league}
    </span>
  );
}
