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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
