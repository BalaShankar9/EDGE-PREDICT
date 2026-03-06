import { cn, tierColor, tierBgColor } from "@/lib/utils";

interface TierBadgeProps {
  tier: string;
  className?: string;
}

export function TierBadge({ tier, className }: TierBadgeProps) {
  return (
    <span className={cn(
      "inline-flex items-center px-2 py-0.5 text-xs font-semibold border",
      tierColor(tier),
      tierBgColor(tier),
      tier === "Platinum" && "animate-pulse",
      className
    )}>
      {tier}
    </span>
  );
}
