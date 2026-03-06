// --- API Envelope ---
export interface ApiResponse<T> {
  status: "ok" | "error";
  data: T;
  meta: {
    count?: number;
    generated_at?: string;
  };
}

// --- Predictions ---
export interface Prediction {
  id?: number;
  match_date: string;
  home_team: string;
  away_team: string;
  league: string;
  prob_home: number;
  prob_draw: number;
  prob_away: number;
  prob_over: number | null;
  prob_under: number | null;
  prob_btts_yes: number | null;
  prob_btts_no: number | null;
}

// --- Picks ---
export type Tier = "Platinum" | "Gold" | "Silver";

export interface Pick {
  id?: number;
  match_date: string;
  home_team: string;
  away_team: string;
  league: string;
  pick_market: string;
  pick_selection: string;
  model_prob: number;
  best_odds: number;
  bookmaker: string;
  edge: number;
  tier: Tier;
  meta_agreement: number;
  result: string | null;
  profit_loss: number | null;
}

// --- Match Detail ---
export interface MatchDetail {
  home_team: string;
  away_team: string;
  match_date: string;
  league: string;
  probabilities: {
    home: number;
    draw: number;
    away: number;
    over_25: number | null;
    under_25: number | null;
    btts_yes: number | null;
    btts_no: number | null;
  };
  xgboost_probs: Record<string, number> | null;
  poisson_probs: Record<string, number> | null;
  ensemble_weights: Record<string, number> | null;
  pick?: {
    market: string;
    selection: string;
    model_prob: number;
    best_odds: number;
    bookmaker: string;
    edge: number;
    tier: Tier;
    meta_agreement: number;
    risk_flags: string[];
    result: string | null;
    profit_loss: number | null;
  };
}

// --- Track Record ---
export interface TierStats {
  total_picks: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_profit: number;
  roi: number;
}

export interface TrackRecord {
  overall: TierStats & {
    max_drawdown: number;
    avg_odds: number;
    longest_win_streak: number;
    longest_loss_streak: number;
  };
  by_tier: Record<string, TierStats>;
  by_league: Record<string, TierStats>;
}

export interface MonthlyRecord {
  month: string;
  picks: number;
  wins: number;
  profit: number;
  roi: number;
  cumulative_profit: number;
}

// --- Leagues ---
export interface League {
  name: string;
  slug: string;
  prediction_count: number;
  pick_count: number;
}

export interface LeaguePrediction {
  home_team: string;
  away_team: string;
  match_date: string;
  prob_home: number;
  prob_draw: number;
  prob_away: number;
  pick?: {
    selection: string;
    tier: Tier;
    best_odds: number;
    edge: number;
  };
}

export interface LeaguePredictions {
  league: string;
  slug: string;
  predictions: LeaguePrediction[];
}
