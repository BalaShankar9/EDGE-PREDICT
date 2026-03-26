"""Tests for WeatherImpact."""
import pytest

from sharpedge.intelligence.weather_impact import WeatherImpact


@pytest.fixture
def weather():
    return WeatherImpact()


class TestWeatherDefaults:
    def test_default_parameters(self, weather):
        feats = weather.compute_features("football")
        assert feats["wx_temperature"] == 20.0
        assert feats["wx_wind_speed"] == 10.0
        assert feats["wx_precipitation"] == 0.0
        assert feats["wx_humidity"] == 50.0

    def test_unknown_sport_base_features_only(self, weather):
        feats = weather.compute_features("cricket")
        assert "wx_temperature" in feats
        assert "wx_rain_under_signal" not in feats


class TestWeatherFootball:
    def test_rain_signal(self, weather):
        feats = weather.compute_features("football", precipitation=5.0)
        assert feats["wx_rain_under_signal"] == 1.0  # 5/5 = 1.0

    def test_heavy_rain_capped(self, weather):
        feats = weather.compute_features("football", precipitation=20.0)
        assert feats["wx_rain_under_signal"] == 1.0

    def test_extreme_cold(self, weather):
        feats = weather.compute_features("football", temperature=-5.0)
        assert feats["wx_extreme_cold"] == 1.0

    def test_normal_temp_no_cold_flag(self, weather):
        feats = weather.compute_features("football", temperature=15.0)
        assert feats["wx_extreme_cold"] == 0.0


class TestWeatherTennis:
    def test_heat_fatigue(self, weather):
        feats = weather.compute_features("tennis", temperature=35.0)
        assert feats["wx_heat_fatigue"] == 1.0

    def test_no_heat_fatigue(self, weather):
        feats = weather.compute_features("tennis", temperature=25.0)
        assert feats["wx_heat_fatigue"] == 0.0


class TestWeatherBaseball:
    def test_wind_out_signal(self, weather):
        feats = weather.compute_features("baseball", wind_speed=30.0)
        assert feats["wx_wind_out_signal"] == pytest.approx(1.0)

    def test_humidity_carry(self, weather):
        feats = weather.compute_features("baseball", humidity=80.0)
        assert feats["wx_humidity_carry"] == pytest.approx(0.8)
