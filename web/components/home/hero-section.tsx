"use client";

import { AnimatedCounter } from "@/components/shared/animated-counter";

interface HeroSectionProps {
  totalPicks: number;
  winRate: number | null;
  roi: number | null;
}

export function HeroSection({ totalPicks, winRate, roi }: HeroSectionProps) {
  return (
    <section className="relative py-16 sm:py-24 overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-80 h-80 bg-accent-primary/5 blur-[120px]" />
        <div className="absolute -bottom-40 -right-40 w-80 h-80 bg-accent-secondary/5 blur-[120px]" />
      </div>

      <div className="relative z-10">
        {/* Headline */}
        <div className="text-center mb-12 animate-slide-in-up">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight mb-4">
            <span className="gradient-text">AI-Powered</span>
            <br />
            <span className="text-text-primary">Football Predictions</span>
          </h1>
          <p className="text-lg sm:text-xl text-text-secondary max-w-2xl mx-auto">
            15 data sources. 50 features. Transparent accuracy.
          </p>
        </div>

        {/* Animated stat counters */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6 max-w-3xl mx-auto animate-slide-in-up-delay-2">
          <div className="glass-card p-6 text-center group hover:-translate-y-0.5 transition-all duration-300">
            <div className="relative">
              <div className="absolute inset-0 bg-accent-primary/5 blur-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
              <p className="text-sm text-text-secondary mb-2 relative">Total Picks</p>
              <p className="text-3xl sm:text-4xl font-bold text-text-primary tabular-nums stat-glow relative">
                <AnimatedCounter end={totalPicks} duration={2000} />
              </p>
            </div>
          </div>

          <div className="glass-card p-6 text-center group hover:-translate-y-0.5 transition-all duration-300 animate-glow">
            <div className="relative">
              <div className="absolute inset-0 bg-accent-primary/5 blur-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
              <p className="text-sm text-text-secondary mb-2 relative">Win Rate</p>
              <p className="text-3xl sm:text-4xl font-bold text-accent-primary tabular-nums stat-glow relative">
                {winRate != null ? (
                  <AnimatedCounter end={winRate * 100} duration={2500} decimals={1} suffix="%" />
                ) : (
                  <span className="text-text-secondary">&mdash;</span>
                )}
              </p>
            </div>
          </div>

          <div className="glass-card p-6 text-center group hover:-translate-y-0.5 transition-all duration-300">
            <div className="relative">
              <div className="absolute inset-0 bg-accent-secondary/5 blur-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
              <p className="text-sm text-text-secondary mb-2 relative">ROI</p>
              <p className="text-3xl sm:text-4xl font-bold text-text-primary tabular-nums stat-glow relative">
                {roi != null ? (
                  <AnimatedCounter
                    end={roi}
                    duration={2500}
                    decimals={1}
                    prefix={roi > 0 ? "+" : ""}
                    suffix="%"
                  />
                ) : (
                  <span className="text-text-secondary">&mdash;</span>
                )}
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
