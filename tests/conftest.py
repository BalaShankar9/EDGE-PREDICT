import pytest
from sharpedge.config import Settings


@pytest.fixture
def test_settings():
    return Settings(
        database_url="sqlite:///test.db",
        environment="test",
    )
