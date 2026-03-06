import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  trend?: "up" | "down" | "neutral";
  glow?: boolean;
  className?: string;
}

export function StatCard({ label, value, trend, glow, className }: StatCardProps) {
  return (
    <div className={cn(
      "relative bg-bg-surface border border-border p-6 transition-all duration-300",
      "hover:border-accent-primary/30 hover:-translate-y-0.5",
      glow && "animate-glow",
      className
    )}>
      {/* Accent top border */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-accent-primary/60 to-accent-secondary/40" />

      {/* Glow background */}
      {glow && (
        <div className="absolute inset-0 bg-gradient-to-br from-accent-primary/5 to-transparent pointer-events-none" />
      )}

      <div className="relative">
        <p className="text-sm text-text-secondary mb-1">{label}</p>
        <div className="flex items-center gap-2">
          <p className={cn(
            "text-2xl font-bold tabular-nums",
            glow ? "text-accent-primary stat-glow" : "text-text-primary"
          )}>
            {value}
          </p>
          {trend && trend !== "neutral" && (
            <span className={cn(
              "text-xs font-medium flex items-center",
              trend === "up" ? "text-success" : "text-danger"
            )}>
              {trend === "up" ? (
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" />
                </svg>
              ) : (
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                </svg>
              )}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
