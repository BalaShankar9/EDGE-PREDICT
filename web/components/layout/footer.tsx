import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-border bg-bg-surface mt-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-lg font-bold text-accent-primary">Sharp</span>
              <span className="text-lg font-bold text-text-primary">Edge</span>
              <span className="text-xs font-medium text-accent-secondary ml-1">AI</span>
            </div>
            <p className="text-sm text-text-secondary">
              AI-powered football predictions with transparent accuracy tracking.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-text-primary mb-3">Navigation</h3>
            <ul className="space-y-2">
              <li><Link href="/" className="text-sm text-text-secondary hover:text-text-primary">Today&apos;s Picks</Link></li>
              <li><Link href="/track-record" className="text-sm text-text-secondary hover:text-text-primary">Track Record</Link></li>
              <li><Link href="/about" className="text-sm text-text-secondary hover:text-text-primary">How It Works</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-text-primary mb-3">Connect</h3>
            <a href="https://t.me/sharpedgeai" target="_blank" rel="noopener noreferrer"
              className="text-sm text-accent-primary hover:text-accent-primary/80">
              Telegram Channel
            </a>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-border">
          <p className="text-xs text-text-secondary text-center">
            Predictions are for informational purposes only. Gambling involves risk.
            Please bet responsibly. &copy; {new Date().getFullYear()} SharpEdge AI.
          </p>
        </div>
      </div>
    </footer>
  );
}
