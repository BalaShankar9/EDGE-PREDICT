import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
      {/* Background orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/3 left-1/3 w-64 h-64 bg-accent-primary/3 blur-[100px]" />
        <div className="absolute bottom-1/3 right-1/3 w-64 h-64 bg-accent-secondary/3 blur-[100px]" />
      </div>

      <div className="relative text-center">
        <h1 className="text-7xl sm:text-8xl font-bold gradient-text mb-4">404</h1>
        <p className="text-lg text-text-secondary mb-2">Page not found</p>
        <p className="text-sm text-text-secondary/60 mb-8">The page you&apos;re looking for doesn&apos;t exist or has been moved.</p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-6 py-2.5 text-sm font-medium hover:bg-accent-primary/20 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Home
        </Link>
      </div>
    </div>
  );
}
