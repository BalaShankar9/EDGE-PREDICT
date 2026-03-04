from sharpedge.config import Settings


def test_settings_defaults():
    s = Settings(database_url="sqlite:///test.db")
    assert s.environment == "development"
    assert s.default_request_delay == 3.0
    assert s.max_retries == 5
