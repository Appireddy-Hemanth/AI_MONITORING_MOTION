from __future__ import annotations

import csv
from datetime import datetime

from utils.evaluation import evaluate_logs


def _write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_logs_supports_legacy_track_confidence(tmp_path):
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    _write_rows(
        log_dir / f"cycles_{today}.csv",
        ["env_accuracy", "energy_accuracy", "detection_count", "person_only_mode"],
        [
            {"env_accuracy": 95.0, "energy_accuracy": 96.0, "detection_count": 1, "person_only_mode": 1},
            {"env_accuracy": 96.0, "energy_accuracy": 95.5, "detection_count": 0, "person_only_mode": 1},
        ],
    )

    # Legacy file: no track_confidence column.
    _write_rows(
        log_dir / f"detections_{today}.csv",
        ["class", "track_id", "tracked_seconds", "speed_px_s", "confidence"],
        [
            {"class": "person", "track_id": 1, "tracked_seconds": 2.2, "speed_px_s": 33.0, "confidence": 0.82},
            {"class": "person", "track_id": 2, "tracked_seconds": 1.2, "speed_px_s": 21.0, "confidence": 0.72},
        ],
    )

    _write_rows(
        log_dir / f"alerts_{today}.csv",
        ["code", "severity", "risk_score"],
        [{"code": "INTRUSION", "severity": "high", "risk_score": 0.9}],
    )

    result = evaluate_logs(str(log_dir))

    assert result.summary["cycles_count"] == 2
    assert result.summary["detections_count"] == 2
    assert result.summary["avg_track_confidence"] > 0.0
    assert result.summary["id_switch_proxy"] > 0.0
