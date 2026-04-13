"""Incident snapshot persistence utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def save_incident_snapshot(frame_rgb: np.ndarray, alerts: list, output_dir: str = "data/incidents") -> str | None:
    """Save frame snapshot when critical/high alert is active."""
    if frame_rgb is None or len(alerts) == 0:
        return None

    severities = {a.get("severity", "").lower() for a in alerts}
    if not ({"critical", "high"} & severities):
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    primary_code = alerts[0].get("code", "EVENT")
    filename = f"{ts}_{primary_code}.jpg"
    path = output_path / filename

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        return None
    return str(path)


def save_incident_clip(frames_rgb: list[np.ndarray], output_dir: str = "data/incidents", fps: int = 10) -> str | None:
    """Persist a short MP4 clip from recent frames as incident evidence."""
    if not frames_rgb:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_path / f"{ts}_clip.mp4"

    first = frames_rgb[0]
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, int(fps)), (int(w), int(h)))
    if not writer.isOpened():
        return None

    try:
        for frame in frames_rgb:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if bgr.shape[0] != h or bgr.shape[1] != w:
                bgr = cv2.resize(bgr, (int(w), int(h)))
            writer.write(bgr)
    finally:
        writer.release()

    return str(path)
