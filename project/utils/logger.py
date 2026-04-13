from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


class MonitoringLogger:
    """Writes cycle, detection, and alert logs with daily rotation."""

    def __init__(self, log_dir: str = "data/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _daily_path(self, prefix: str, now: datetime) -> Path:
        return self.log_dir / f"{prefix}_{now.strftime('%Y-%m-%d')}.csv"

    @staticmethod
    def _append_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return

        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def log_cycle(
        self,
        now: datetime,
        sensor: dict,
        env_pred: dict,
        energy_pred: dict,
        step_ms: float,
        fps: float,
        yolo_ms: float,
        detection_count: int,
        alert_count: int,
        detection_enabled: bool,
        person_only_mode: bool,
        uptime_s: float = 0.0,
        dropped_frames: int = 0,
        reconnect_count: int = 0,
        inference_ms: float = 0.0,
    ) -> None:
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": round(float(sensor.get("temperature", 0.0)), 3),
            "humidity": round(float(sensor.get("humidity", 0.0)), 3),
            "gas": round(float(sensor.get("gas", 0.0)), 3),
            "env_accuracy": round(float(env_pred.get("accuracy", 0.0)), 3),
            "energy_accuracy": round(float(energy_pred.get("accuracy", 0.0)), 3),
            "pred_energy": round(float(energy_pred.get("predicted_energy", 0.0)), 5),
            "expected_energy": round(float(energy_pred.get("expected_energy", 0.0)), 5),
            "step_ms": round(float(step_ms), 3),
            "fps": round(float(fps), 3),
            "yolo_ms": round(float(yolo_ms), 3),
            "detection_count": int(detection_count),
            "alert_count": int(alert_count),
            "detection_enabled": int(bool(detection_enabled)),
            "person_only_mode": int(bool(person_only_mode)),
            "uptime_s": round(float(uptime_s), 3),
            "dropped_frames": int(dropped_frames),
            "reconnect_count": int(reconnect_count),
            "inference_ms": round(float(inference_ms), 3),
        }
        self._append_rows(
            self._daily_path("cycles", now),
            list(row.keys()),
            [row],
        )

    def log_detections(self, now: datetime, detections: list[dict]) -> None:
        rows = []
        for det in detections:
            rows.append(
                {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "track_id": int(det.get("track_id", -1)),
                    "class": str(det.get("class", "")),
                    "confidence": round(float(det.get("confidence", 0.0)), 4),
                    "speed_px_s": round(float(det.get("speed_px_s", 0.0)), 4),
                    "speed_m_s": round(float(det.get("speed_m_s", 0.0)), 4),
                    "tracked_seconds": round(float(det.get("tracked_seconds", 0.0)), 4),
                    "track_confidence": round(float(det.get("track_confidence", 0.0)), 4),
                    "bbox": str(det.get("bbox", "")),
                    "source": str(det.get("source", "")),
                }
            )

        self._append_rows(
            self._daily_path("detections", now),
            [
                "timestamp",
                "track_id",
                "class",
                "confidence",
                "speed_px_s",
                "speed_m_s",
                "tracked_seconds",
                "track_confidence",
                "bbox",
                "source",
            ],
            rows,
        )

    def log_alerts(self, now: datetime, alerts: list[dict]) -> None:
        rows = []
        for alert in alerts:
            rows.append(
                {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": str(alert.get("source", "")),
                    "severity": str(alert.get("severity", "")),
                    "code": str(alert.get("code", "")),
                    "risk_score": round(float(alert.get("risk_score", 0.0)), 4),
                    "message": str(alert.get("message", "")),
                }
            )

        self._append_rows(
            self._daily_path("alerts", now),
            ["timestamp", "source", "severity", "code", "risk_score", "message"],
            rows,
        )
