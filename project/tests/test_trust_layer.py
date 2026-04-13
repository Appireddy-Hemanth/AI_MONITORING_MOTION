from __future__ import annotations

from utils.trust_layer import AlertTrustLayer


def test_trust_layer_queues_and_releases_verified_alert():
    layer = AlertTrustLayer()
    alerts = [
        {
            "code": "FIRE",
            "severity": "critical",
            "risk_score": 0.92,
            "message": "possible fire",
            "source": "vision",
            "color": "#ff1e1e",
        }
    ]

    detections = [{"class": "fire", "bbox": (10, 10, 20, 20), "confidence": 0.9}]
    trust_config = {
        "active_hours": {"start": 0, "end": 24},
        "restricted_zone": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9, "frame_width": 100, "frame_height": 100},
        "occupancy_limit": 10,
        "require_human_verification_codes": ["FIRE"],
    }

    gated, queued = layer.apply(alerts, detections, trust_config, {"FIRE": 1}, frame_path="")
    assert gated == []
    assert len(queued) == 1

    released = layer.resolve_verification(queued[0]["id"], approved=True)
    assert released is not None
    assert released["code"] == "FIRE"
