from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://localhost:5432/sharpedge"
    telegram_bot_token: str = ""
    telegram_alert_chat_id: str = ""
    environment: str = "development"

    # Scraping settings
    default_request_delay: float = 3.0
    max_retries: int = 5
    request_timeout: int = 30

    # Data directories
    cache_dir: str = "data/cache"

    # API settings
    api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Telegram bot settings
    telegram_channel_id: str = ""
    telegram_poll_interval: int = 60

    # Model settings
    model_path: str = "models/latest.pkl"

    # --- Bankroll & Execution ---
    bankroll_initial: float = 1000.0
    kelly_fraction: float = 0.25  # Quarter-Kelly (conservative)
    max_bet_pct: float = 0.03  # 3% max per bet
    max_daily_pct: float = 0.10  # 10% max per day
    max_correlated_pct: float = 0.15  # 15% max on correlated outcomes

    # --- Drift Detection ---
    drift_window_days: int = 30
    drift_calibration_threshold: float = 0.08  # ECE above this = drift
    drift_brier_threshold: float = 0.28  # Brier above this = drift
    drift_accuracy_floor: float = 0.47  # Below this = emergency retrain
    drift_check_min_predictions: int = 50

    # --- Agent Evolution ---
    agent_min_bets_for_eval: int = 30  # Min bets before judging an agent
    agent_deprecation_roi_threshold: float = -0.10  # -10% ROI = consider deprecation
    agent_promotion_clv_threshold: float = 0.02  # +2% CLV = promote
    agent_evolution_interval_days: int = 14  # Run evolution every 2 weeks

    # --- CLV Tracking ---
    clv_target_pct: float = 2.0  # Target: beat closing line by 2%+

    # --- Live Pipeline ---
    football_data_api_key: str = ""  # football-data.org API key
    odds_api_key: str = ""  # the-odds-api.com key
    pipeline_run_hour: int = 8  # Run daily pipeline at 8 AM UTC
    result_check_delay_hours: int = 3  # Check results 3h after last kickoff

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
