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
    value >= 0.75 ? "bg-accent-primary"
    : value >= 0.60 ? "bg-accent-secondary"
    : "bg-text-secondary";

  return (
    <div className={cn("w-full", className)}>
      {label && (
        <div className="flex justify-between mb-1">
          <span className="text-xs text-text-secondary">{label}</span>
          <span className="text-xs font-medium text-text-primary tabular-nums">
            {formatPercent(value)}
          </span>
        </div>
      )}
      <div className="h-2 bg-bg-elevated overflow-hidden">
        <div
          className={cn("h-full transition-all duration-1000 ease-out", barColor)}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}
