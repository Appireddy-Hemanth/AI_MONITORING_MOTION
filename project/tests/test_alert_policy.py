from __future__ import annotations

from utils.alert_policy import (
    approve_policy_change,
    create_policy_change_request,
    list_policy_versions,
    load_policy,
    resolve_policy_context,
    save_policy,
)


def test_policy_context_resolution_and_approval_flow(tmp_path):
    policy_path = tmp_path / "alert_policy.json"

    initial = load_policy(str(policy_path))
    initial["contexts"] = {
        "night_plant": {
            "site": "plant-a",
            "shift": "night",
            "risk_profile": "high",
            "global_overrides": {"cooldown_seconds": 4.0},
            "code_overrides": {"INTRUSION": {"min_risk": 0.85, "confirmation_frames": 3}},
        }
    }
    save_policy(str(policy_path), initial)

    candidate = dict(initial)
    candidate["global"] = dict(initial["global"])
    candidate["global"]["gas_threshold"] = 500.0

    change_id = create_policy_change_request(str(policy_path), candidate, requested_by="tester")
    ok, _ = approve_policy_change(str(policy_path), change_id, approved_by="lead")
    assert ok

    loaded = load_policy(str(policy_path))
    assert float(loaded["global"]["gas_threshold"]) == 500.0

    resolved = resolve_policy_context(loaded, site="plant-a", shift="night", risk_profile="high")
    assert float(resolved["global"]["cooldown_seconds"]) == 4.0
    assert float(resolved["alert_codes"]["INTRUSION"]["min_risk"]) == 0.85
    assert int(resolved["alert_codes"]["INTRUSION"]["confirmation_frames"]) == 3

    assert list_policy_versions(str(policy_path))
