"""Run historical backfill — 5 seasons, all 10 collectors."""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backfill.log"),
    ],
)

from sharpedge.orchestrator import run_backfill

if __name__ == "__main__":
    seasons = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_backfill(seasons=seasons)

    print("\n=== BACKFILL SUMMARY ===")
    total_rows = 0
    for r in results:
        status_icon = "+" if r.status == "healthy" else "~" if r.status == "degraded" else "X"
        print(f"  [{status_icon}] {r.source}: {r.rows} rows ({r.status})")
        if r.errors:
            for e in r.errors:
                print(f"     Error: {e}")
        total_rows += r.rows

    healthy = sum(1 for r in results if r.status == "healthy")
    degraded = sum(1 for r in results if r.status == "degraded")
    down = sum(1 for r in results if r.status == "down")
    print(f"\nTotal: {total_rows} rows | {healthy} healthy, {degraded} degraded, {down} down")
