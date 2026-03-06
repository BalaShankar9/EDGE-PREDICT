"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { label: "Today", href: "/" },
  { label: "Leagues", href: "/leagues" },
  { label: "Track Record", href: "/track-record" },
  { label: "About", href: "/about" },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Ticker bar */}
      <div className="bg-bg-surface/80 backdrop-blur-sm border-b border-white/5 overflow-hidden">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="h-8 flex items-center text-xs text-text-secondary overflow-hidden">
            <div className="sm:hidden whitespace-nowrap ticker-animate inline-flex gap-8">
              <span>&#128293; AI-Powered Predictions</span>
              <span>&#9889; Platinum: 82.1% accuracy</span>
              <span>15 data sources &middot; 50 features</span>
              <span>&#128293; AI-Powered Predictions</span>
              <span>&#9889; Platinum: 82.1% accuracy</span>
              <span>15 data sources &middot; 50 features</span>
            </div>
            <div className="hidden sm:flex items-center gap-6">
              <span className="flex items-center gap-1.5">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full bg-accent-primary opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 bg-accent-primary" />
                </span>
                Live predictions
              </span>
              <span className="text-text-secondary/40">|</span>
              <span>&#9889; Platinum: 82.1% accuracy</span>
              <span className="text-text-secondary/40">|</span>
              <span>15 data sources &middot; 50 features &middot; Transparent results</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main nav */}
      <nav className="sticky top-0 z-50 border-b border-border bg-bg-primary/80 backdrop-blur-xl">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <Link href="/" className="flex items-center gap-2">
              {/* Lightning bolt icon */}
              <svg className="w-5 h-5 text-accent-primary" viewBox="0 0 24 24" fill="currentColor">
                <path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z" />
              </svg>
              <span className="text-xl font-bold text-accent-primary">Sharp</span>
              <span className="text-xl font-bold text-text-primary">Edge</span>
              <span className="text-xs font-medium text-accent-secondary ml-1 mt-1">AI</span>
            </Link>

            <div className="hidden sm:flex items-center gap-1">
              {NAV_ITEMS.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "relative px-3 py-2 text-sm font-medium transition-colors",
                    pathname === item.href
                      ? "text-accent-primary"
                      : "text-text-secondary hover:text-text-primary"
                  )}
                >
                  {item.label}
                  {pathname === item.href && (
                    <span className="absolute bottom-0 left-1 right-1 h-0.5 bg-gradient-to-r from-accent-primary to-accent-secondary" />
                  )}
                </Link>
              ))}
            </div>

            <div className="flex items-center gap-3">
              <a
                href="https://t.me/sharpedgeai"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden sm:inline-flex items-center gap-2 bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-4 py-2 text-sm font-medium hover:bg-accent-primary/20 transition-colors"
              >
                Join Telegram
              </a>

              {/* Mobile hamburger */}
              <button
                className="sm:hidden text-text-secondary hover:text-text-primary p-1"
                aria-label="Menu"
                onClick={() => setMobileOpen(true)}
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile drawer overlay */}
      {mobileOpen && (
        <div className="fixed inset-0 z-[60]">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
          {/* Drawer */}
          <div className="absolute right-0 top-0 bottom-0 w-72 bg-bg-surface border-l border-border p-6 flex flex-col animate-slide-in-up">
            <div className="flex justify-between items-center mb-8">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-accent-primary" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z" />
                </svg>
                <span className="text-lg font-bold text-accent-primary">Sharp</span>
                <span className="text-lg font-bold text-text-primary">Edge</span>
              </div>
              <button
                onClick={() => setMobileOpen(false)}
                className="text-text-secondary hover:text-text-primary"
                aria-label="Close menu"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex flex-col gap-1">
              {NAV_ITEMS.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "px-4 py-3 text-sm font-medium transition-colors border-l-2",
                    pathname === item.href
                      ? "text-accent-primary border-l-accent-primary bg-accent-primary/5"
                      : "text-text-secondary hover:text-text-primary border-l-transparent"
                  )}
                >
                  {item.label}
                </Link>
              ))}
            </div>
            <div className="mt-auto pt-6 border-t border-border">
              <a
                href="https://t.me/sharpedgeai"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-4 py-2.5 text-sm font-medium hover:bg-accent-primary/20 transition-colors w-full"
              >
                Join Telegram
              </a>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
