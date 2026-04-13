from datetime import datetime
import random


class SensorSimulator:
    """Simulates realistic environmental sensor readings with temporal continuity."""

    def __init__(self) -> None:
        self.temperature = 26.0
        self.humidity = 52.0
        self.gas = 220.0

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def read(self) -> dict:
        # Random-walk updates keep readings smooth while still dynamic.
        self.temperature += random.uniform(-0.45, 0.45)
        self.humidity += random.uniform(-1.25, 1.25)
        self.gas += random.uniform(-8.0, 8.0)

        # Rare abnormal events simulate real incidents.
        if random.random() < 0.02:
            self.temperature += random.uniform(3.0, 8.0)
        if random.random() < 0.02:
            self.gas += random.uniform(60.0, 180.0)

        self.temperature = self._clamp(self.temperature, 16.0, 60.0)
        self.humidity = self._clamp(self.humidity, 20.0, 90.0)
        self.gas = self._clamp(self.gas, 80.0, 900.0)

        return {
            "timestamp": datetime.now(),
            "temperature": round(self.temperature, 2),
            "humidity": round(self.humidity, 2),
            "gas": round(self.gas, 2),
        }
