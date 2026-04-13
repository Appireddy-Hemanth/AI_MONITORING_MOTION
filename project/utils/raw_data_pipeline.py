"""Raw-data generation/loading and PGNN calibration utilities."""

from __future__ import annotations

import csv
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _generate_raw_records(num_samples: int = 1000, seed: int = 42) -> List[Dict[str, float]]:
    """Generate baseline physics-consistent raw records for environment and energy."""
    rng = np.random.default_rng(seed)

    temperature = 26.0
    humidity = 50.0
    gas = 220.0

    records: List[Dict[str, float]] = []
    for i in range(num_samples):
        temperature = _clamp(temperature + float(rng.uniform(-0.5, 0.5)), 18.0, 45.0)
        humidity = _clamp(humidity + float(rng.uniform(-1.0, 1.0)), 30.0, 88.0)
        gas = _clamp(gas + float(rng.uniform(-7.0, 7.0)), 90.0, 650.0)

        usage = _clamp(0.35 * (temperature / 60.0) + 0.65 * (gas / 900.0) + float(rng.uniform(-0.02, 0.02)), 0.0, 1.0)
        time_hours = 0.05 + 0.02 * i

        records.append(
            {
                "temperature": round(temperature, 3),
                "humidity": round(humidity, 3),
                "gas": round(gas, 3),
                "time_hours": round(float(time_hours), 4),
                "usage": round(float(usage), 5),
            }
        )

    return records


def generate_profile_records(num_samples: int = 1000, profile: str = "baseline", seed: int = 42) -> List[Dict[str, float]]:
    """Generate raw records with profile-specific operating patterns."""
    rng = np.random.default_rng(seed)

    profile = profile.lower().strip()
    if profile == "industrial":
        temperature = 31.0
        humidity = 42.0
        gas = 330.0
        temp_step, hum_step, gas_step = 0.9, 1.2, 12.5
    elif profile == "outdoor":
        temperature = 28.0
        humidity = 62.0
        gas = 170.0
        temp_step, hum_step, gas_step = 1.4, 2.0, 9.0
    elif profile == "critical":
        temperature = 34.0
        humidity = 48.0
        gas = 410.0
        temp_step, hum_step, gas_step = 1.2, 1.1, 17.0
    else:
        return _generate_raw_records(num_samples=num_samples, seed=seed)

    records: List[Dict[str, float]] = []
    for i in range(num_samples):
        cyc = np.sin(i / 55.0)
        temperature = _clamp(temperature + float(rng.uniform(-temp_step, temp_step)) + 0.22 * cyc, 16.0, 58.0)
        humidity = _clamp(humidity + float(rng.uniform(-hum_step, hum_step)) - 0.15 * cyc, 20.0, 95.0)
        gas = _clamp(gas + float(rng.uniform(-gas_step, gas_step)) + 2.0 * cyc, 70.0, 900.0)

        usage = _clamp(
            0.32 * (temperature / 60.0) + 0.68 * (gas / 900.0) + float(rng.uniform(-0.03, 0.03)),
            0.0,
            1.0,
        )
        time_hours = 0.05 + 0.02 * i

        records.append(
            {
                "temperature": round(temperature, 3),
                "humidity": round(humidity, 3),
                "gas": round(gas, 3),
                "time_hours": round(float(time_hours), 4),
                "usage": round(float(usage), 5),
            }
        )

    return records


def _write_records(path: Path, records: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["temperature", "humidity", "gas", "time_hours", "usage"])
        writer.writeheader()
        writer.writerows(records)


def load_or_create_raw_dataset(csv_path: str, num_samples: int = 1000) -> List[Dict[str, float]]:
    """Load existing raw dataset or create one with the requested sample size."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = {"temperature", "humidity", "gas", "time_hours", "usage"}
    records: List[Dict[str, float]] = []

    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and required_cols.issubset(set(reader.fieldnames)):
                for row in reader:
                    records.append(
                        {
                            "temperature": float(row["temperature"]),
                            "humidity": float(row["humidity"]),
                            "gas": float(row["gas"]),
                            "time_hours": float(row["time_hours"]),
                            "usage": float(row["usage"]),
                        }
                    )

    if len(records) < num_samples:
        records = _generate_raw_records(num_samples=num_samples)
        _write_records(path, records)
    else:
        records = records[:num_samples]

    return records


def create_profile_dataset(csv_path: str, profile: str, num_samples: int = 1000, seed: int = 42) -> str:
    """Create a profile-specific dataset at the requested path."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_profile_records(num_samples=num_samples, profile=profile, seed=seed)
    _write_records(path, records)
    return str(path)


def ensure_default_datasets(data_dir: str = "data", num_samples: int = 1000) -> List[str]:
    """Ensure multiple default datasets exist in the data folder."""
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)

    specs = [
        ("raw_sensor_energy_1000.csv", "baseline", 42),
        ("raw_sensor_energy_industrial_1200.csv", "industrial", 133),
        ("raw_sensor_energy_outdoor_1000.csv", "outdoor", 271),
        ("raw_sensor_energy_critical_1500.csv", "critical", 911),
    ]

    created_or_existing: List[str] = []
    for filename, profile, seed in specs:
        path = base / filename
        if not path.exists():
            create_profile_dataset(str(path), profile=profile, num_samples=num_samples if "1500" not in filename else 1500, seed=seed)
        created_or_existing.append(str(path))
    return created_or_existing


def list_dataset_paths(data_dir: str = "data") -> List[str]:
    """List CSV dataset files under data directory."""
    return sorted(glob(str(Path(data_dir) / "*.csv")))


def load_dataset(csv_path: str, limit: int | None = None) -> List[Dict[str, float]]:
    """Load a single dataset file if schema matches expected columns."""
    path = Path(csv_path)
    if not path.exists():
        return []

    required_cols = {"temperature", "humidity", "gas", "time_hours", "usage"}
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            return []
        for row in reader:
            rows.append(
                {
                    "temperature": float(row["temperature"]),
                    "humidity": float(row["humidity"]),
                    "gas": float(row["gas"]),
                    "time_hours": float(row["time_hours"]),
                    "usage": float(row["usage"]),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_multiple_datasets(csv_paths: List[str], per_dataset_limit: int = 1000) -> List[Dict[str, float]]:
    """Load and concatenate multiple dataset files."""
    all_rows: List[Dict[str, float]] = []
    for p in csv_paths:
        all_rows.extend(load_dataset(p, limit=per_dataset_limit))
    return all_rows


def calibrate_with_raw_dataset(env_trainer, energy_trainer, records: List[Dict[str, float]], epochs: int = 5) -> Dict[str, float]:
    """Train PGNN models on raw records and return rolling validation accuracies."""
    split = int(len(records) * 0.8)
    train_set = records[:split]
    val_set = records[split:]

    for _ in range(epochs):
        for row in train_set:
            sensor = {
                "temperature": row["temperature"],
                "humidity": row["humidity"],
                "gas": row["gas"],
            }
            env_trainer.step(sensor, enable_replay=False)
            energy_trainer.step(time_hours=row["time_hours"], device_usage=row["usage"], enable_replay=False)

    env_scores = []
    energy_scores = []
    for row in val_set:
        sensor = {
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "gas": row["gas"],
        }
        env_out = env_trainer.step(sensor, enable_replay=False)
        energy_out = energy_trainer.step(time_hours=row["time_hours"], device_usage=row["usage"], enable_replay=False)
        env_scores.append(float(env_out.get("accuracy", 0.0)))
        energy_scores.append(float(energy_out.get("accuracy", 0.0)))

    env_acc = float(np.mean(env_scores)) if env_scores else 0.0
    energy_acc = float(np.mean(energy_scores)) if energy_scores else 0.0

    return {
        "samples": float(len(records)),
        "epochs": float(epochs),
        "env_accuracy": round(env_acc, 2),
        "energy_accuracy": round(energy_acc, 2),
        "meets_target": float(env_acc >= 95.0 and energy_acc >= 95.0),
    }
