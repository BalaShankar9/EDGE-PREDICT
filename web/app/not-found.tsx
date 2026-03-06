import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh]">
      <h1 className="text-4xl font-bold text-text-primary mb-2">404</h1>
      <p className="text-text-secondary mb-6">Page not found</p>
      <Link
        href="/"
        className="bg-accent-primary/10 border border-accent-primary/30 text-accent-primary px-4 py-2 text-sm font-medium hover:bg-accent-primary/20 transition-colors"
      >
        Back to Home
      </Link>
    </div>
  );
}
