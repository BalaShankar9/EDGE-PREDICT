import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  className?: string;
}

export function StatCard({ label, value, className }: StatCardProps) {
  return (
    <div className={cn(
      "bg-bg-surface border border-border p-6",
      "hover:border-accent-primary/30 transition-colors",
      className
    )}>
      <p className="text-sm text-text-secondary mb-1">{label}</p>
      <p className="text-2xl font-bold text-text-primary tabular-nums">{value}</p>
    </div>
  );
}
