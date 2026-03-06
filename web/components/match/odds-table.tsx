import { formatOdds, formatEdge } from "@/lib/utils";

interface OddsTableProps {
  bestOdds?: number;
  bookmaker?: string;
  edge?: number;
  probHome: number;
  probDraw: number;
  probAway: number;
}

export function OddsTable({ bestOdds, bookmaker, edge, probHome, probDraw, probAway }: OddsTableProps) {
  const fairHome = probHome > 0 ? 1 / probHome : 0;
  const fairDraw = probDraw > 0 ? 1 / probDraw : 0;
  const fairAway = probAway > 0 ? 1 / probAway : 0;

  return (
    <div className="bg-bg-surface border border-border p-6">
      <h3 className="text-sm font-semibold text-text-primary mb-4">Odds Analysis</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-text-secondary border-b border-border">
            <th className="text-left py-2">Market</th>
            <th className="text-right py-2">Fair Odds</th>
            <th className="text-right py-2">Probability</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-border">
            <td className="py-2 text-text-primary">Home Win</td>
            <td className="py-2 text-right tabular-nums text-text-primary">{formatOdds(fairHome)}</td>
            <td className="py-2 text-right tabular-nums text-text-secondary">{(probHome * 100).toFixed(1)}%</td>
          </tr>
          <tr className="border-b border-border">
            <td className="py-2 text-text-primary">Draw</td>
            <td className="py-2 text-right tabular-nums text-text-primary">{formatOdds(fairDraw)}</td>
            <td className="py-2 text-right tabular-nums text-text-secondary">{(probDraw * 100).toFixed(1)}%</td>
          </tr>
          <tr className="border-b border-border">
            <td className="py-2 text-text-primary">Away Win</td>
            <td className="py-2 text-right tabular-nums text-text-primary">{formatOdds(fairAway)}</td>
            <td className="py-2 text-right tabular-nums text-text-secondary">{(probAway * 100).toFixed(1)}%</td>
          </tr>
        </tbody>
      </table>
      {bestOdds && (
        <div className="mt-4 pt-4 border-t border-border">
          <div className="flex justify-between text-sm">
            <span className="text-text-secondary">Best Available</span>
            <span className="text-accent-primary font-medium tabular-nums">{formatOdds(bestOdds)} ({bookmaker})</span>
          </div>
          {edge != null && (
            <div className="flex justify-between text-sm mt-1">
              <span className="text-text-secondary">Value Edge</span>
              <span className="text-success font-medium tabular-nums">{formatEdge(edge)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
