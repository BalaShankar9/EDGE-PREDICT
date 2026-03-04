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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
