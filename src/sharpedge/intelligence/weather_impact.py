"""Weather Impact Quantifier — sport-specific weather models."""


class WeatherImpact:
    """Computes weather impact features for outdoor sports."""

    # Sport-specific impact coefficients (from literature)
    FOOTBALL_RAIN_UNDER_BOOST = 0.08  # rain increases under probability by ~8%
    FOOTBALL_WIND_UNDER_BOOST = 0.05
    TENNIS_HEAT_FATIGUE = 0.03  # heat favors younger/fitter players
    BASEBALL_WIND_HR_FACTOR = 0.10  # wind out increases HR/runs

    def compute_features(self, sport: str, temperature: float = 20.0,
                         wind_speed: float = 10.0, precipitation: float = 0.0,
                         humidity: float = 50.0) -> dict[str, float]:
        """Compute weather impact features."""
        features = {
            "wx_temperature": temperature,
            "wx_wind_speed": wind_speed,
            "wx_precipitation": precipitation,
            "wx_humidity": humidity,
        }

        if sport == "football":
            features["wx_rain_under_signal"] = min(1.0, precipitation / 5.0)
            features["wx_wind_under_signal"] = min(1.0, wind_speed / 30.0)
            features["wx_extreme_cold"] = 1.0 if temperature < 0 else 0.0
        elif sport == "tennis":
            features["wx_heat_fatigue"] = 1.0 if temperature > 32 else 0.0
            features["wx_wind_disruption"] = min(1.0, wind_speed / 25.0)
        elif sport == "baseball":
            features["wx_wind_out_signal"] = max(0, wind_speed - 10) / 20.0
            features["wx_humidity_carry"] = humidity / 100.0

        return features
