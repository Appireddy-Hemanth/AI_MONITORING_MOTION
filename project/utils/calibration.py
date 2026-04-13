from __future__ import annotations

import json
from pathlib import Path


def compute_meters_per_pixel(known_distance_m: float, pixel_distance_px: float) -> float:
    if known_distance_m <= 0 or pixel_distance_px <= 0:
        return 0.0
    return float(known_distance_m / pixel_distance_px)


def save_calibration(path: str, known_distance_m: float, pixel_distance_px: float) -> float:
    meters_per_pixel = compute_meters_per_pixel(known_distance_m, pixel_distance_px)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "known_distance_m": float(known_distance_m),
        "pixel_distance_px": float(pixel_distance_px),
        "meters_per_pixel": float(meters_per_pixel),
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meters_per_pixel


def load_calibration(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "known_distance_m": 1.0,
            "pixel_distance_px": 100.0,
            "meters_per_pixel": 0.0,
        }

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        known_distance_m = float(payload.get("known_distance_m", 1.0))
        pixel_distance_px = float(payload.get("pixel_distance_px", 100.0))
        meters_per_pixel = float(payload.get("meters_per_pixel", compute_meters_per_pixel(known_distance_m, pixel_distance_px)))
        return {
            "known_distance_m": known_distance_m,
            "pixel_distance_px": pixel_distance_px,
            "meters_per_pixel": meters_per_pixel,
        }
    except Exception:
        return {
            "known_distance_m": 1.0,
            "pixel_distance_px": 100.0,
            "meters_per_pixel": 0.0,
        }
