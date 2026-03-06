"use client";

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

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-bg-primary/80 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
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
                  "px-3 py-2 text-sm font-medium transition-colors",
                  pathname === item.href
                    ? "text-accent-primary"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                {item.label}
              </Link>
            ))}
          </div>
          <a
            href="https://t.me/sharpedgeai"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:inline-flex items-center gap-2 bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-4 py-2 text-sm font-medium hover:bg-accent-primary/20 transition-colors"
          >
            Join Telegram
          </a>
          <button className="sm:hidden text-text-secondary hover:text-text-primary" aria-label="Menu">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </nav>
  );
}
