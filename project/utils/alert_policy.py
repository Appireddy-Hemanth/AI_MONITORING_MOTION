from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_POLICY = {
    "global": {
        "cooldown_seconds": 10.0,
        "intrusion_clear_seconds": 8.0,
        "gas_threshold": 450.0,
        "temp_threshold": 42.0,
        "gas_clear_ratio": 0.90,
        "temp_clear_delta": 2.0,
        "normal_after_quiet_seconds": 20.0,
    },
    "alert_codes": {
        "INTRUSION": {
            "cooldown_seconds": 10.0,
            "min_risk": 0.62,
            "severity": "high",
            "confirmation_frames": 2,
            "escalate_after_seconds": 120.0,
        },
        "FIRE": {
            "cooldown_seconds": 6.0,
            "min_risk": 0.78,
            "severity": "critical",
            "confirmation_frames": 1,
            "escalate_after_seconds": 30.0,
        },
        "GAS_HIGH": {
            "cooldown_seconds": 8.0,
            "min_risk": 0.80,
            "severity": "critical",
            "confirmation_frames": 2,
            "escalate_after_seconds": 90.0,
        },
        "TEMP_HIGH": {
            "cooldown_seconds": 8.0,
            "min_risk": 0.66,
            "severity": "high",
            "confirmation_frames": 2,
            "escalate_after_seconds": 180.0,
        },
        "NORMAL": {
            "cooldown_seconds": 30.0,
            "min_risk": 0.0,
            "severity": "low",
            "confirmation_frames": 1,
            "escalate_after_seconds": 0.0,
        },
    },
    "contexts": {
        "default": {
            "site": "default-site",
            "shift": "all",
            "risk_profile": "normal",
            "global_overrides": {},
            "code_overrides": {},
        }
    },
}


def _to_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _to_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _normalize_code_policy(payload: dict, fallback: dict) -> dict:
    out = dict(fallback)
    out["cooldown_seconds"] = _to_float(payload.get("cooldown_seconds", out["cooldown_seconds"]), out["cooldown_seconds"])
    out["min_risk"] = _to_float(payload.get("min_risk", out["min_risk"]), out["min_risk"])
    out["confirmation_frames"] = max(1, _to_int(payload.get("confirmation_frames", out["confirmation_frames"]), out["confirmation_frames"]))
    out["escalate_after_seconds"] = _to_float(payload.get("escalate_after_seconds", out["escalate_after_seconds"]), out["escalate_after_seconds"])
    out["severity"] = str(payload.get("severity", out["severity"]))
    return out


def _normalize_global_policy(payload: dict, fallback: dict) -> dict:
    out = dict(fallback)
    for key, val in fallback.items():
        out[key] = _to_float(payload.get(key, val), val)
    return out


def _normalize_policy(payload: dict) -> dict:
    policy = {
        "global": _normalize_global_policy(payload.get("global", {}), DEFAULT_POLICY["global"]),
        "alert_codes": {},
        "contexts": payload.get("contexts", DEFAULT_POLICY["contexts"]),
    }

    raw_codes = payload.get("alert_codes", {})
    for code, fallback in DEFAULT_POLICY["alert_codes"].items():
        policy["alert_codes"][code] = _normalize_code_policy(raw_codes.get(code, {}), fallback)

    if not isinstance(policy["contexts"], dict) or not policy["contexts"]:
        policy["contexts"] = dict(DEFAULT_POLICY["contexts"])
    return policy


def load_policy(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return dict(DEFAULT_POLICY)

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if "global" not in payload and any(k in DEFAULT_POLICY["global"] for k in payload.keys()):
            payload = {
                "global": {k: payload.get(k, DEFAULT_POLICY["global"][k]) for k in DEFAULT_POLICY["global"]},
                "alert_codes": payload.get("alert_codes", {}),
                "contexts": payload.get("contexts", DEFAULT_POLICY["contexts"]),
            }
        return _normalize_policy(payload)
    except Exception:
        return dict(DEFAULT_POLICY)


def save_policy(path: str, policy: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    safe = _normalize_policy(policy)
    file_path.write_text(json.dumps(safe, indent=2), encoding="utf-8")


def apply_policy(manager, policy: dict) -> None:
    global_policy = policy.get("global", {})
    manager.cooldown_seconds = float(global_policy.get("cooldown_seconds", manager.cooldown_seconds))
    manager.intrusion_clear_seconds = float(global_policy.get("intrusion_clear_seconds", manager.intrusion_clear_seconds))
    manager.gas_threshold = float(global_policy.get("gas_threshold", manager.gas_threshold))
    manager.temp_threshold = float(global_policy.get("temp_threshold", manager.temp_threshold))
    manager.gas_clear_ratio = float(global_policy.get("gas_clear_ratio", manager.gas_clear_ratio))
    manager.temp_clear_delta = float(global_policy.get("temp_clear_delta", manager.temp_clear_delta))
    manager.normal_after_quiet_seconds = float(global_policy.get("normal_after_quiet_seconds", manager.normal_after_quiet_seconds))
    manager.code_policy = dict(policy.get("alert_codes", {}))


def resolve_policy_context(policy: dict, site: str, shift: str, risk_profile: str) -> dict:
    merged = _normalize_policy(policy)
    contexts = merged.get("contexts", {})
    for ctx in contexts.values():
        if str(ctx.get("site", "default-site")) != str(site):
            continue
        if str(ctx.get("shift", "all")) not in {"all", str(shift)}:
            continue
        if str(ctx.get("risk_profile", "normal")) not in {"all", str(risk_profile)}:
            continue

        for key, val in dict(ctx.get("global_overrides", {})).items():
            if key in merged["global"]:
                merged["global"][key] = _to_float(val, merged["global"][key])

        for code, overrides in dict(ctx.get("code_overrides", {})).items():
            if code not in merged["alert_codes"]:
                continue
            merged["alert_codes"][code] = _normalize_code_policy(overrides, merged["alert_codes"][code])
    return merged


def create_policy_change_request(policy_path: str, candidate_policy: dict, requested_by: str = "operator") -> str:
    base = Path(policy_path)
    pending_dir = base.parent / "policy_pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    change_id = datetime.now().strftime("chg_%Y%m%d_%H%M%S_%f")
    payload = {
        "change_id": change_id,
        "requested_by": requested_by,
        "requested_at": datetime.now().isoformat(),
        "policy": _normalize_policy(candidate_policy),
        "status": "pending",
    }
    (pending_dir / f"{change_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return change_id


def list_pending_policy_changes(policy_path: str) -> list[dict]:
    base = Path(policy_path)
    pending_dir = base.parent / "policy_pending"
    if not pending_dir.exists():
        return []
    rows = []
    for p in sorted(pending_dir.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def approve_policy_change(policy_path: str, change_id: str, approved_by: str = "supervisor") -> tuple[bool, str]:
    base = Path(policy_path)
    pending_file = base.parent / "policy_pending" / f"{change_id}.json"
    if not pending_file.exists():
        return False, "Change request not found"

    try:
        payload = json.loads(pending_file.read_text(encoding="utf-8"))
        candidate = _normalize_policy(payload.get("policy", {}))
    except Exception:
        return False, "Invalid change request file"

    history_dir = base.parent / "policy_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    if base.exists():
        history_name = datetime.now().strftime("policy_%Y%m%d_%H%M%S.json")
        shutil.copy2(base, history_dir / history_name)

    save_policy(str(base), candidate)
    payload["status"] = "approved"
    payload["approved_by"] = approved_by
    payload["approved_at"] = datetime.now().isoformat()
    pending_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True, "Policy change approved and applied"


def rollback_policy(policy_path: str, version_file: str) -> tuple[bool, str]:
    base = Path(policy_path)
    version_path = base.parent / "policy_history" / version_file
    if not version_path.exists():
        return False, "Policy version not found"
    try:
        payload = json.loads(version_path.read_text(encoding="utf-8"))
        save_policy(str(base), payload)
        return True, "Policy rollback completed"
    except Exception:
        return False, "Failed to rollback policy"


def list_policy_versions(policy_path: str) -> list[str]:
    base = Path(policy_path)
    history_dir = base.parent / "policy_history"
    if not history_dir.exists():
        return []
    return sorted([p.name for p in history_dir.glob("*.json")], reverse=True)
