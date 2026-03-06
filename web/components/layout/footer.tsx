import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-border bg-bg-surface mt-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <svg className="w-4 h-4 text-accent-primary" viewBox="0 0 24 24" fill="currentColor">
                <path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z" />
              </svg>
              <span className="text-lg font-bold text-accent-primary">Sharp</span>
              <span className="text-lg font-bold text-text-primary">Edge</span>
              <span className="text-xs font-medium text-accent-secondary ml-1">AI</span>
            </div>
            <p className="text-sm text-text-secondary mb-4">
              AI-powered football predictions with transparent accuracy tracking.
            </p>
            {/* Powered by AI badge */}
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-accent-primary/5 border border-accent-primary/20 text-xs text-accent-primary animate-glow">
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor">
                <path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z" />
              </svg>
              Powered by AI
            </div>
          </div>

          {/* Navigation */}
          <div>
            <h3 className="text-sm font-semibold text-text-primary mb-3">Navigation</h3>
            <ul className="space-y-2">
              <li><Link href="/" className="text-sm text-text-secondary hover:text-text-primary transition-colors">Today&apos;s Picks</Link></li>
              <li><Link href="/leagues" className="text-sm text-text-secondary hover:text-text-primary transition-colors">Leagues</Link></li>
              <li><Link href="/track-record" className="text-sm text-text-secondary hover:text-text-primary transition-colors">Track Record</Link></li>
              <li><Link href="/about" className="text-sm text-text-secondary hover:text-text-primary transition-colors">How It Works</Link></li>
            </ul>
          </div>

          {/* Connect */}
          <div>
            <h3 className="text-sm font-semibold text-text-primary mb-3">Connect</h3>
            <div className="space-y-2">
              <a href="https://t.me/sharpedgeai" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-text-secondary hover:text-accent-primary transition-colors">
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm4.64 6.8c-.15 1.58-.8 5.42-1.13 7.19-.14.75-.42 1-.68 1.03-.58.05-1.02-.38-1.58-.75-.88-.58-1.38-.94-2.23-1.5-.99-.65-.35-1.01.22-1.59.15-.15 2.71-2.48 2.76-2.69a.2.2 0 00-.05-.18c-.06-.05-.14-.03-.21-.02-.09.02-1.49.95-4.22 2.79-.4.27-.76.41-1.08.4-.36-.01-1.04-.2-1.55-.37-.63-.2-1.12-.31-1.08-.66.02-.18.27-.36.74-.55 2.92-1.27 4.86-2.11 5.83-2.51 2.78-1.16 3.35-1.36 3.73-1.36.08 0 .27.02.39.12.1.08.13.19.14.27-.01.06.01.24 0 .38z" /></svg>
                Telegram Channel
              </a>
            </div>
          </div>

          {/* Newsletter */}
          <div>
            <h3 className="text-sm font-semibold text-text-primary mb-3">Stay Updated</h3>
            <p className="text-xs text-text-secondary mb-3">Get daily picks in your inbox.</p>
            <div className="flex gap-2">
              <input
                type="email"
                placeholder="your@email.com"
                className="flex-1 bg-bg-primary border border-border px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary/50 focus:outline-none focus:border-accent-primary/50 transition-colors min-w-0"
              />
              <button className="bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-3 py-2 text-sm font-medium hover:bg-accent-primary/20 transition-colors shrink-0">
                Go
              </button>
            </div>
          </div>
        </div>

        <div className="mt-10 pt-8 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-text-secondary text-center sm:text-left">
            Predictions are for informational purposes only. Gambling involves risk.
            Please bet responsibly. &copy; {new Date().getFullYear()} SharpEdge AI.
          </p>
        </div>
      </div>
    </footer>
  );
}
