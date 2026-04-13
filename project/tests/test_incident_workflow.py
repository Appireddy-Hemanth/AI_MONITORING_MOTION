from __future__ import annotations

from datetime import datetime, timedelta

from utils.incident_workflow import IncidentManager


def test_incident_lifecycle_and_sla_escalation(tmp_path):
    manager = IncidentManager(str(tmp_path / "incidents.json"))

    alert = {"code": "INTRUSION", "severity": "high", "risk_score": 0.8}
    sensor = {"temperature": 30.0, "humidity": 50.0, "gas": 220.0}
    detections = [{"confidence": 0.8, "class": "person"}]

    inc = manager.open_or_update(
        alert,
        sensor=sensor,
        detections=detections,
        evidence_snapshot="",
        evidence_clip="",
        timeline_note="opened",
        sla_seconds=1,
        assigned_to="security-team",
    )

    assert manager.acknowledge(inc["id"], actor="op1")
    assert manager.assign(inc["id"], assignee="lead1")
    assert manager.resolve(inc["id"], root_cause="test", feedback="done")

    # Reopen a fresh incident to exercise escalation path.
    inc2 = manager.open_or_update(
        alert,
        sensor=sensor,
        detections=detections,
        evidence_snapshot="",
        evidence_clip="",
        timeline_note="opened2",
        sla_seconds=1,
        assigned_to="security-team",
    )
    inc2["created_at"] = (datetime.now() - timedelta(seconds=120)).strftime("%Y-%m-%d %H:%M:%S")
    inc2["status"] = "open"
    manager._save()

    escalated = manager.check_sla_escalations()
    assert len(escalated) == 1
    assert escalated[0]["id"] == inc2["id"]
    assert escalated[0]["escalated"] is True
