from __future__ import annotations

from collections import defaultdict
from datetime import datetime


class AlertTrustLayer:
    """Applies multi-frame confirmation, zone rules, and verification queue handling."""

    def __init__(self) -> None:
        self.code_streaks = defaultdict(int)
        self.verification_queue = []
        self.feedback_stats = defaultdict(lambda: {"true_positive": 0, "false_positive": 0})

    @staticmethod
    def _is_in_restricted_zone(detections: list[dict], zone: dict) -> bool:
        if not detections:
            return False
        zx1 = float(zone.get("x1", 0.2))
        zy1 = float(zone.get("y1", 0.2))
        zx2 = float(zone.get("x2", 0.8))
        zy2 = float(zone.get("y2", 0.9))
        for det in detections:
            if det.get("class") != "person":
                continue
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0
            fw = max(1.0, float(zone.get("frame_width", 960.0)))
            fh = max(1.0, float(zone.get("frame_height", 540.0)))
            nx = cx / fw
            ny = cy / fh
            if zx1 <= nx <= zx2 and zy1 <= ny <= zy2:
                return True
        return False

    @staticmethod
    def _is_time_allowed(now: datetime, active_hours: dict) -> bool:
        start_h = int(active_hours.get("start", 0))
        end_h = int(active_hours.get("end", 24))
        h = now.hour
        if start_h <= end_h:
            return start_h <= h < end_h
        return h >= start_h or h < end_h

    @staticmethod
    def occupancy_alert(detections: list[dict], occupancy_limit: int) -> dict | None:
        people = [d for d in detections if d.get("class") == "person"]
        if len(people) <= int(occupancy_limit):
            return None
        return {
            "timestamp": datetime.now(),
            "source": "vision",
            "message": f"Occupancy exceeded: {len(people)} people over limit {occupancy_limit}.",
            "severity": "high",
            "color": "#ff8c00",
            "code": "OCCUPANCY",
            "risk_score": 0.75,
            "escalate_after_seconds": 120.0,
        }

    def apply(
        self,
        alerts: list[dict],
        detections: list[dict],
        trust_config: dict,
        confirmation_frames_by_code: dict,
        frame_path: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        now = datetime.now()
        restricted_zone = dict(trust_config.get("restricted_zone", {}))
        active_hours = dict(trust_config.get("active_hours", {"start": 0, "end": 24}))
        occupancy_limit = int(trust_config.get("occupancy_limit", 3))
        verify_codes = set(trust_config.get("require_human_verification_codes", ["FIRE", "GAS_HIGH"]))

        gated = []
        queued = []

        occupancy = self.occupancy_alert(detections, occupancy_limit)
        if occupancy is not None:
            alerts = list(alerts) + [occupancy]

        for alert in alerts:
            code = str(alert.get("code", "UNKNOWN"))
            needed_frames = max(1, int(confirmation_frames_by_code.get(code, 1)))
            self.code_streaks[code] += 1
            if self.code_streaks[code] < needed_frames:
                continue

            if code == "INTRUSION":
                if not self._is_time_allowed(now, active_hours):
                    continue
                if not self._is_in_restricted_zone(detections, restricted_zone):
                    continue

            if code in verify_codes and str(alert.get("severity", "")).lower() in {"high", "critical"}:
                item = {
                    "id": f"verify_{now.strftime('%Y%m%d_%H%M%S_%f')}",
                    "status": "pending",
                    "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "alert": dict(alert),
                    "frame_path": frame_path or "",
                }
                self.verification_queue.append(item)
                queued.append(item)
                continue

            gated.append(alert)

        current_codes = {str(a.get("code", "UNKNOWN")) for a in alerts}
        stale = [k for k in list(self.code_streaks.keys()) if k not in current_codes]
        for k in stale:
            self.code_streaks[k] = 0

        self.verification_queue = self.verification_queue[-300:]
        return gated, queued

    def resolve_verification(self, item_id: str, approved: bool) -> dict | None:
        for item in self.verification_queue:
            if item.get("id") != item_id:
                continue
            if item.get("status") != "pending":
                return None
            item["status"] = "approved" if approved else "rejected"
            item["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            code = str(item.get("alert", {}).get("code", "UNKNOWN"))
            if approved:
                self.feedback_stats[code]["true_positive"] += 1
                return dict(item.get("alert", {}))
            self.feedback_stats[code]["false_positive"] += 1
            return None
        return None

    def auto_tune(self, policy: dict) -> dict:
        out = dict(policy)
        code_cfg = dict(out.get("alert_codes", {}))
        for code, stats in self.feedback_stats.items():
            fp = int(stats.get("false_positive", 0))
            tp = int(stats.get("true_positive", 0))
            total = fp + tp
            if total < 5 or code not in code_cfg:
                continue
            ratio = fp / total
            if ratio < 0.5:
                continue
            cfg = dict(code_cfg[code])
            cfg["min_risk"] = min(0.95, float(cfg.get("min_risk", 0.5)) + 0.03)
            cfg["cooldown_seconds"] = min(300.0, float(cfg.get("cooldown_seconds", 10.0)) + 2.0)
            code_cfg[code] = cfg
        out["alert_codes"] = code_cfg
        return out
