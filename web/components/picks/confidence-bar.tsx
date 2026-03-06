"use client";

import { useEffect, useState } from "react";
import { cn, formatPercent } from "@/lib/utils";

interface ConfidenceBarProps {
  value: number;
  label?: string;
  className?: string;
}

export function ConfidenceBar({ value, label, className }: ConfidenceBarProps) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setWidth(value * 100), 100);
    return () => clearTimeout(timer);
  }, [value]);

  const barColor =
    value >= 0.75
      ? "bg-gradient-to-r from-accent-primary to-accent-primary/70"
      : value >= 0.60
        ? "bg-gradient-to-r from-accent-secondary to-accent-secondary/70"
        : "bg-gradient-to-r from-text-secondary/80 to-text-secondary/50";

  return (
    <div className={cn("w-full", className)}>
      {label && (
        <div className="flex justify-between mb-1.5">
          <span className="text-xs text-text-secondary">{label}</span>
          <span className="text-xs font-semibold text-text-primary tabular-nums">
            {formatPercent(value)}
          </span>
        </div>
      )}
      <div className="h-1.5 bg-bg-elevated overflow-hidden relative">
        <div
          className={cn("h-full transition-all duration-1000 ease-out", barColor)}
          style={{ width: `${width}%` }}
        />
        {/* Shimmer overlay on the bar */}
        <div
          className="absolute inset-0 shimmer pointer-events-none"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}
