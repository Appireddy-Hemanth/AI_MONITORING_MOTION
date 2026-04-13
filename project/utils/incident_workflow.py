from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class IncidentManager:
    """Manages incident lifecycle with SLA escalation and evidence metadata."""

    def __init__(self, store_path: str = "data/incidents/incidents.json") -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.incidents = []
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            self.incidents = []
            return
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            self.incidents = list(payload) if isinstance(payload, list) else []
        except Exception:
            self.incidents = []

    def _save(self) -> None:
        self.store_path.write_text(json.dumps(self.incidents[-1000:], indent=2), encoding="utf-8")

    def open_or_update(
        self,
        alert: dict,
        sensor: dict,
        detections: list[dict],
        evidence_snapshot: str,
        evidence_clip: str,
        timeline_note: str,
        sla_seconds: int,
        assigned_to: str,
    ) -> dict:
        code = str(alert.get("code", "UNKNOWN"))
        now = datetime.now()

        open_incident = None
        for inc in reversed(self.incidents):
            if inc.get("code") == code and inc.get("status") in {"open", "acknowledged", "assigned"}:
                open_incident = inc
                break

        if open_incident is None:
            open_incident = {
                "id": f"inc_{now.strftime('%Y%m%d_%H%M%S_%f')}",
                "code": code,
                "status": "open",
                "severity": str(alert.get("severity", "high")),
                "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "acknowledged_at": "",
                "assigned_to": assigned_to,
                "resolved_at": "",
                "sla_seconds": int(sla_seconds),
                "escalated": False,
                "escalation_target": "supervisor",
                "root_cause": "",
                "feedback": "",
                "timeline": [],
                "evidence": {
                    "snapshot_paths": [],
                    "clip_paths": [],
                    "sensor_context": [],
                    "model_confidence": [],
                },
            }
            self.incidents.append(open_incident)

        if evidence_snapshot:
            open_incident["evidence"]["snapshot_paths"].append(evidence_snapshot)
        if evidence_clip:
            open_incident["evidence"]["clip_paths"].append(evidence_clip)
        open_incident["evidence"]["sensor_context"].append(
            {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": float(sensor.get("temperature", 0.0)),
                "humidity": float(sensor.get("humidity", 0.0)),
                "gas": float(sensor.get("gas", 0.0)),
            }
        )
        open_incident["evidence"]["model_confidence"].append(
            {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "max_detection_confidence": max([float(d.get("confidence", 0.0)) for d in detections] + [0.0]),
                "risk_score": float(alert.get("risk_score", 0.0)),
            }
        )
        open_incident["timeline"].append(
            {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "event": timeline_note,
                "status": open_incident.get("status", "open"),
            }
        )

        open_incident["evidence"]["snapshot_paths"] = open_incident["evidence"]["snapshot_paths"][-20:]
        open_incident["evidence"]["clip_paths"] = open_incident["evidence"]["clip_paths"][-10:]
        open_incident["evidence"]["sensor_context"] = open_incident["evidence"]["sensor_context"][-40:]
        open_incident["evidence"]["model_confidence"] = open_incident["evidence"]["model_confidence"][-40:]
        open_incident["timeline"] = open_incident["timeline"][-80:]

        self._save()
        return open_incident

    def acknowledge(self, incident_id: str, actor: str = "operator") -> bool:
        for inc in self.incidents:
            if inc.get("id") != incident_id:
                continue
            if inc.get("status") in {"resolved"}:
                return False
            inc["status"] = "acknowledged"
            inc["acknowledged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            inc["timeline"].append({"timestamp": inc["acknowledged_at"], "event": f"acknowledged by {actor}", "status": "acknowledged"})
            self._save()
            return True
        return False

    def assign(self, incident_id: str, assignee: str) -> bool:
        for inc in self.incidents:
            if inc.get("id") != incident_id:
                continue
            if inc.get("status") in {"resolved"}:
                return False
            inc["status"] = "assigned"
            inc["assigned_to"] = assignee
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            inc["timeline"].append({"timestamp": now, "event": f"assigned to {assignee}", "status": "assigned"})
            self._save()
            return True
        return False

    def resolve(self, incident_id: str, root_cause: str, feedback: str = "") -> bool:
        for inc in self.incidents:
            if inc.get("id") != incident_id:
                continue
            inc["status"] = "resolved"
            inc["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            inc["root_cause"] = root_cause
            inc["feedback"] = feedback
            inc["timeline"].append({"timestamp": inc["resolved_at"], "event": f"resolved with root_cause={root_cause}", "status": "resolved"})
            self._save()
            return True
        return False

    def check_sla_escalations(self) -> list[dict]:
        now = datetime.now()
        escalated = []
        for inc in self.incidents:
            if inc.get("status") in {"resolved", "acknowledged", "assigned"}:
                continue
            if inc.get("escalated", False):
                continue
            created_at = datetime.strptime(inc.get("created_at", now.strftime("%Y-%m-%d %H:%M:%S")), "%Y-%m-%d %H:%M:%S")
            elapsed = (now - created_at).total_seconds()
            if elapsed < int(inc.get("sla_seconds", 120)):
                continue
            inc["escalated"] = True
            inc["timeline"].append(
                {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": f"SLA missed; escalated to {inc.get('escalation_target', 'supervisor')}",
                    "status": inc.get("status", "open"),
                }
            )
            escalated.append(inc)
        if escalated:
            self._save()
        return escalated

    def latest(self, limit: int = 100) -> list[dict]:
        return self.incidents[-limit:]
