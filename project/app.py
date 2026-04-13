"""AI-powered smart monitoring focused on PINN-based sensor and energy modeling."""

from __future__ import annotations

import csv
import io
import time
import wave
from datetime import datetime
from pathlib import Path
import statistics

import cv2
import numpy as np
import streamlit as st

from pgnn.energy_model import EnergyPGNNTrainer
from pgnn.env_model import EnvironmentPGNNTrainer
from utils.alerts import AlertManager
from utils.alert_policy import (
    apply_policy,
    approve_policy_change,
    create_policy_change_request,
    list_pending_policy_changes,
    list_policy_versions,
    load_policy,
    resolve_policy_context,
    rollback_policy,
    save_policy,
)
from utils.calibration import load_calibration, save_calibration
from utils.emailer import EmailConfig, EmailNotifier
from utils.evaluation import evaluate_logs, export_evaluation
from utils.incident_workflow import IncidentManager
from utils.incidents import save_incident_clip, save_incident_snapshot
from utils.logger import MonitoringLogger
from utils.review_bundle import create_review_bundle
from utils.scenario_log import ScenarioRecorder
from utils.trust_layer import AlertTrustLayer
from utils.raw_data_pipeline import (
    calibrate_with_raw_dataset,
    ensure_default_datasets,
    list_dataset_paths,
    load_dataset,
    load_multiple_datasets,
    load_or_create_raw_dataset,
)
from utils.sensors import SensorSimulator
from yolo.detect import VisionDetector


st.set_page_config(
    page_title="AI-Powered Smart Monitoring System",
    page_icon="AI",
    layout="wide",
)


def inject_styles() -> None:
    """Apply dashboard styling with dark theme and card-like panels."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --card: #10233c;
            --card-alt: #133153;
            --text: #ecf1ff;
            --muted: #9cb3d4;
            --line: #24476d;
            --accent: #17c0eb;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 10%, rgba(23,192,235,0.12), transparent 35%),
                radial-gradient(circle at 80% 0%, rgba(0,250,154,0.10), transparent 28%),
                linear-gradient(145deg, #050a14 0%, #07111f 48%, #08182a 100%);
            color: var(--text);
            font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
        }

        .panel {
            background: linear-gradient(160deg, var(--card), var(--card-alt));
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
        }

        .metric-big {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text);
        }

        .metric-label {
            font-size: 0.86rem;
            color: var(--muted);
        }

        .title {
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }

        .subtitle {
            color: var(--muted);
            margin-bottom: 14px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #081224 0%, #0c1d36 100%);
            border-right: 1px solid #17314f;
        }

        .alert-box {
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 5px solid;
            background: rgba(7, 15, 28, 0.72);
            color: #f4f7ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_beep_wav(duration_s: float = 0.2, freq: float = 880.0) -> bytes:
    """Generate a short WAV tone for critical alerts."""
    sample_rate = 44100
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave_data = 0.25 * np.sin(2 * np.pi * freq * t)
    int_data = np.int16(wave_data * 32767)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_data.tobytes())

    return buffer.getvalue()


def fallback_frame(message: str = "No camera/video input") -> np.ndarray:
    """Return a placeholder frame when video source is unavailable."""
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    cv2.putText(frame, message, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def release_capture() -> None:
    """Release active video capture if present."""
    if st.session_state.capture is not None:
        st.session_state.capture.release()
        st.session_state.capture = None


def _open_capture_for_current_source() -> bool:
    """Open capture using current source settings and return readiness."""
    if st.session_state.video_mode == "Upload Video" and st.session_state.video_file_path:
        st.session_state.capture = cv2.VideoCapture(st.session_state.video_file_path)
    else:
        st.session_state.capture = cv2.VideoCapture(0)

    if st.session_state.capture is None:
        return False

    st.session_state.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    st.session_state.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return bool(st.session_state.capture.isOpened())


def prune_stale_incident_paths(max_items: int = 500) -> None:
    """Drop incident image references that no longer exist on disk."""
    valid = []
    for p in list(st.session_state.incident_history)[-max_items:]:
        if p and Path(p).exists():
            valid.append(p)
    st.session_state.incident_history = valid

    latest = str(st.session_state.latest_incident_path or "")
    if latest and not Path(latest).exists():
        st.session_state.latest_incident_path = ""


def check_logs_writable(log_dir: str = "data/logs") -> bool:
    """Verify log directory can be written before monitoring starts."""
    try:
        base = Path(log_dir)
        base.mkdir(parents=True, exist_ok=True)
        probe = base / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def init_state() -> None:
    """Initialize Streamlit session state objects."""
    defaults = {
        "monitoring": False,
        "detector": None,
        "detector_error": "",
        "capture": None,
        "video_mode": "Webcam",
        "video_file_path": None,
        "fire_model_path": "",
        "vision_detection_enabled": True,
        "person_only_mode": False,
        "detect_all_objects": True,
        "vision_min_confidence": 0.15,
        "vision_profile": "Comprehensive",
        "person_threshold": 0.25,
        "object_threshold": 0.20,
        "paper_threshold": 0.15,
        "fire_threshold": 0.45,
        "calibration_known_distance_m": 1.0,
        "calibration_pixel_distance_px": 100.0,
        "calibration_meters_per_pixel": 0.0,
        "calibration_x1": 100,
        "calibration_y1": 100,
        "calibration_x2": 400,
        "calibration_y2": 100,
        "calibration_loaded": False,
        "alert_policy": {},
        "alert_policy_loaded": False,
        "policy_site": "default-site",
        "policy_shift": "day",
        "policy_risk_profile": "normal",
        "policy_requester": "operator",
        "policy_approver": "supervisor",
        "policy_pending_choice": "",
        "policy_version_choice": "",
        "monitor_logger": MonitoringLogger("data/logs"),
        "trust_layer": AlertTrustLayer(),
        "trust_config": {
            "active_hours": {"start": 0, "end": 24},
            "restricted_zone": {"x1": 0.25, "y1": 0.20, "x2": 0.75, "y2": 0.90, "frame_width": 960, "frame_height": 540},
            "occupancy_limit": 3,
            "require_human_verification_codes": ["FIRE", "GAS_HIGH"],
            "sla_seconds": 120,
            "default_assignee": "security-team",
        },
        "incident_manager": IncidentManager("data/incidents/incidents.json"),
        "recent_frames": [],
        "resolved_alerts": [],
        "incident_root_cause": "unknown",
        "incident_feedback_note": "",
        "scenario_recorder": ScenarioRecorder("data/scenarios"),
        "scenario_active": False,
        "scenario_name": "",
        "scenario_mark_text": "",
        "scenario_file": "",
        "sensor_sim": SensorSimulator(),
        "alert_manager": AlertManager(),
        "env_pgnn": EnvironmentPGNNTrainer(lambda_physics=0.45),
        "energy_pgnn": EnergyPGNNTrainer(lambda_physics=0.80),
        "latest_frame": fallback_frame(),
        "latest_detections": [],
        "latest_alerts": [],
        "latest_sensor": {"temperature": 0.0, "humidity": 0.0, "gas": 0.0},
        "latest_env_pred": {},
        "latest_energy_pred": {},
        "latest_env_accuracy": 0.0,
        "latest_energy_accuracy": 0.0,
        "calibration_report": {},
        "latest_incident_path": "",
        "latest_detection_age_s": -1.0,
        "camera_ok": False,
        "capture_fail_streak": 0,
        "dropped_frames": 0,
        "reconnect_count": 0,
        "camera_reconnect_threshold": 5,
        "last_reconnect_at": "",
        "monitoring_started_perf": 0.0,
        "latest_uptime_s": 0.0,
        "latest_inference_ms": 0.0,
        "preflight_last_checks": [],
        "incident_history": [],
        "alert_cooldown_seconds": 10,
        "email_enabled": False,
        "email_notifier": EmailNotifier(cooldown_seconds=60.0),
        "email_cooldown_seconds": 60,
        "email_smtp_host": "smtp.gmail.com",
        "email_smtp_port": 587,
        "email_sender": "",
        "email_password": "",
        "email_receiver": "",
        "email_status": "",
        "email_last_sent": "",
        "email_attach_intrusion_image": True,
        "dataset_paths": [],
        "selected_datasets": [],
        "pinn_warm_started": False,
        "latest_step_ms": 0.0,
        "latest_yolo_ms": 0.0,
        "latest_fps": 0.0,
        "critical_beep": False,
        "tick": 0,
        "vision_stride": 1,
        "history": {
            "time": [],
            "temperature": [],
            "humidity": [],
            "gas": [],
            "pred_temperature": [],
            "pred_humidity": [],
            "pred_gas": [],
            "usage": [],
            "power_kw": [],
            "pred_energy": [],
            "expected_energy": [],
            "env_accuracy": [],
            "energy_accuracy": [],
            "step_ms": [],
            "fps": [],
            "yolo_ms": [],
            "inference_ms": [],
            "uptime_s": [],
            "dropped_frames": [],
            "reconnect_count": [],
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prepare_detector(
    fire_model_path: str = "",
    min_confidence: float = 0.25,
    detect_all_objects: bool = True,
    detection_profile: str = "balanced",
    person_only_mode: bool = False,
    person_threshold: float = 0.25,
    object_threshold: float = 0.20,
    paper_threshold: float = 0.15,
    fire_threshold: float = 0.45,
) -> None:
    """Load YOLO detector once when vision detection is enabled."""
    if st.session_state.detector is None:
        try:
            chosen_fire_model = fire_model_path.strip() if fire_model_path else None
            st.session_state.detector = VisionDetector(
                "yolov8n.pt",
                fire_model_path=chosen_fire_model,
                min_confidence=min_confidence,
                detect_all_objects=detect_all_objects,
                detection_profile=detection_profile,
                person_only_mode=person_only_mode,
                person_threshold=person_threshold,
                object_threshold=object_threshold,
                paper_threshold=paper_threshold,
                fire_threshold=fire_threshold,
            )
            st.session_state.detector_error = ""
        except Exception as exc:
            st.session_state.detector = None
            st.session_state.detector_error = str(exc)


def prepare_capture(video_mode: str, uploaded_file) -> None:
    """Create appropriate OpenCV capture object for webcam or uploaded video."""
    if st.session_state.capture is not None:
        return

    if video_mode == "Upload Video" and uploaded_file is not None:
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        save_path = data_dir / "uploaded_input.mp4"
        save_path.write_bytes(uploaded_file.getbuffer())
        st.session_state.video_file_path = str(save_path)
        st.session_state.capture = cv2.VideoCapture(str(save_path))
    else:
        st.session_state.video_file_path = None
        st.session_state.capture = cv2.VideoCapture(0)

    if st.session_state.capture is not None:
        st.session_state.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        st.session_state.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)


def _attempt_capture_reconnect() -> bool:
    """Try reopening capture when repeated frame reads fail."""
    release_capture()
    opened = _open_capture_for_current_source()
    if opened:
        st.session_state.reconnect_count += 1
        st.session_state.last_reconnect_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return opened


def read_current_frame() -> tuple[np.ndarray, bool]:
    """Read frame from source, restarting uploaded video if finished."""
    cap = st.session_state.capture
    if cap is None or not cap.isOpened():
        st.session_state.capture_fail_streak += 1
        st.session_state.dropped_frames += 1
        if st.session_state.capture_fail_streak >= int(st.session_state.camera_reconnect_threshold):
            if _attempt_capture_reconnect():
                cap = st.session_state.capture
                ok, frame = cap.read()
                if ok and frame is not None:
                    st.session_state.capture_fail_streak = 0
                    return frame, True
        return fallback_frame("Camera/video source unavailable"), False

    ok, frame = cap.read()
    if ok and frame is not None:
        st.session_state.capture_fail_streak = 0
        return frame, True

    st.session_state.capture_fail_streak += 1
    st.session_state.dropped_frames += 1

    if st.session_state.video_mode == "Upload Video" and st.session_state.video_file_path:
        release_capture()
        st.session_state.capture = cv2.VideoCapture(st.session_state.video_file_path)
        st.session_state.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        st.session_state.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        ok, frame = st.session_state.capture.read()
        if ok and frame is not None:
            st.session_state.capture_fail_streak = 0
            return frame, True

    if st.session_state.capture_fail_streak >= int(st.session_state.camera_reconnect_threshold):
        if _attempt_capture_reconnect():
            cap = st.session_state.capture
            ok, frame = cap.read()
            if ok and frame is not None:
                st.session_state.capture_fail_streak = 0
                return frame, True

    return fallback_frame("No frames available"), False


def append_history(sensor: dict, env_pred: dict, energy_pred: dict, usage: float) -> None:
    """Append latest values into rolling history buffers."""
    hist = st.session_state.history
    timestamp = datetime.now().strftime("%H:%M:%S")

    hist["time"].append(timestamp)
    hist["temperature"].append(sensor["temperature"])
    hist["humidity"].append(sensor["humidity"])
    hist["gas"].append(sensor["gas"])
    hist["pred_temperature"].append(round(env_pred.get("temperature", 0.0), 2))
    hist["pred_humidity"].append(round(env_pred.get("humidity", 0.0), 2))
    hist["pred_gas"].append(round(env_pred.get("gas", 0.0), 2))
    hist["usage"].append(round(usage, 3))
    hist["power_kw"].append(round(energy_pred.get("power_kw", 0.0), 3))
    hist["pred_energy"].append(round(energy_pred.get("predicted_energy", 0.0), 3))
    hist["expected_energy"].append(round(energy_pred.get("expected_energy", 0.0), 3))
    hist["env_accuracy"].append(round(env_pred.get("accuracy", 0.0), 2))
    hist["energy_accuracy"].append(round(energy_pred.get("accuracy", 0.0), 2))
    hist["step_ms"].append(round(st.session_state.latest_step_ms, 2))
    hist["fps"].append(round(st.session_state.latest_fps, 2))
    hist["yolo_ms"].append(round(st.session_state.latest_yolo_ms, 2))
    hist["inference_ms"].append(round(st.session_state.latest_inference_ms, 2))
    hist["uptime_s"].append(round(st.session_state.latest_uptime_s, 2))
    hist["dropped_frames"].append(int(st.session_state.dropped_frames))
    hist["reconnect_count"].append(int(st.session_state.reconnect_count))

    max_len = 240
    for key in hist:
        hist[key] = hist[key][-max_len:]


def build_report_bytes() -> bytes:
    """Generate CSV report for sensor, energy, and alerts history."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["AI-Powered Smart Monitoring Report"])
    writer.writerow(["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])

    hist = st.session_state.history
    writer.writerow(
        [
            "time",
            "temperature",
            "humidity",
            "gas",
            "pred_temperature",
            "pred_humidity",
            "pred_gas",
            "usage",
            "power_kw",
            "pred_energy",
            "expected_energy",
            "env_accuracy",
            "energy_accuracy",
            "step_ms",
            "fps",
            "yolo_ms",
            "inference_ms",
            "uptime_s",
            "dropped_frames",
            "reconnect_count",
        ]
    )
    for idx in range(len(hist["time"])):
        writer.writerow(
            [
                hist["time"][idx],
                hist["temperature"][idx],
                hist["humidity"][idx],
                hist["gas"][idx],
                hist["pred_temperature"][idx],
                hist["pred_humidity"][idx],
                hist["pred_gas"][idx],
                hist["usage"][idx],
                hist["power_kw"][idx],
                hist["pred_energy"][idx],
                hist["expected_energy"][idx],
                hist["env_accuracy"][idx],
                hist["energy_accuracy"][idx],
                hist["step_ms"][idx],
                hist["fps"][idx],
                hist["yolo_ms"][idx],
                hist["inference_ms"][idx],
                hist["uptime_s"][idx],
                hist["dropped_frames"][idx],
                hist["reconnect_count"][idx],
            ]
        )

    writer.writerow([])
    writer.writerow(["timestamp", "source", "severity", "code", "risk_score", "message"])
    for alert in st.session_state.alert_manager.latest(300):
        writer.writerow(
            [
                alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                alert["source"],
                alert["severity"],
                alert.get("code", ""),
                alert.get("risk_score", 0.0),
                alert["message"],
            ]
        )

    writer.writerow([])
    writer.writerow(["incident_snapshot_path"])
    for p in st.session_state.incident_history[-200:]:
        writer.writerow([p])

    return output.getvalue().encode("utf-8")


def run_monitor_step() -> None:
    """Process one full monitoring cycle for PINN and optional vision detection."""
    step_start = time.perf_counter()
    sensor = st.session_state.sensor_sim.read()
    env_pred = st.session_state.env_pgnn.step(sensor)

    usage = min(1.0, 0.4 * (sensor["temperature"] / 60.0) + 0.6 * (sensor["gas"] / 900.0))
    time_hours = max(0.05, (st.session_state.tick + 1) * 0.05)
    energy_pred = st.session_state.energy_pgnn.step(time_hours=time_hours, device_usage=usage)

    cap = st.session_state.capture
    st.session_state.camera_ok = bool(cap is not None and cap.isOpened())
    frame, camera_live = read_current_frame()
    st.session_state.camera_ok = bool(camera_live)
    prune_stale_incident_paths(max_items=500)
    detections = []
    should_detect = (
        st.session_state.vision_detection_enabled
        and st.session_state.detector is not None
        and camera_live
        and (st.session_state.tick % max(1, st.session_state.vision_stride)) == 0
    )
    yolo_start = time.perf_counter()
    if should_detect:
        try:
            frame, detections = st.session_state.detector.detect(
                frame,
                meters_per_pixel=float(st.session_state.calibration_meters_per_pixel),
            )
        except Exception as exc:
            st.session_state.detector_error = str(exc)
            detections = []
    st.session_state.latest_yolo_ms = (time.perf_counter() - yolo_start) * 1000.0 if should_detect else 0.0
    st.session_state.latest_inference_ms = st.session_state.latest_yolo_ms

    alerts = st.session_state.alert_manager.evaluate(
        detections,
        sensor,
        env_accuracy=env_pred.get("accuracy", 100.0),
        energy_accuracy=energy_pred.get("accuracy", 100.0),
    )

    confirmation_by_code = {
        code: int(cfg.get("confirmation_frames", 1))
        for code, cfg in dict(st.session_state.alert_policy.get("alert_codes", {})).items()
    }

    gated_alerts, queued_items = st.session_state.trust_layer.apply(
        alerts,
        detections,
        st.session_state.trust_config,
        confirmation_by_code,
        frame_path=st.session_state.latest_incident_path,
    )
    alerts = gated_alerts

    # Process operator decisions from verification queue.
    if st.session_state.resolved_alerts:
        for action in st.session_state.resolved_alerts:
            approved = bool(action.get("approved", False))
            item_id = str(action.get("id", ""))
            released_alert = st.session_state.trust_layer.resolve_verification(item_id, approved=approved)
            if released_alert:
                alerts.append(released_alert)
        st.session_state.resolved_alerts = []

    tuned_policy = st.session_state.trust_layer.auto_tune(st.session_state.alert_policy)
    if tuned_policy != st.session_state.alert_policy:
        st.session_state.alert_policy = tuned_policy
        effective = resolve_policy_context(
            st.session_state.alert_policy,
            st.session_state.policy_site,
            st.session_state.policy_shift,
            st.session_state.policy_risk_profile,
        )
        apply_policy(st.session_state.alert_manager, effective)

    st.session_state.latest_sensor = sensor
    st.session_state.latest_env_pred = env_pred
    st.session_state.latest_energy_pred = energy_pred
    st.session_state.latest_env_accuracy = env_pred.get("accuracy", 0.0)
    st.session_state.latest_energy_accuracy = energy_pred.get("accuracy", 0.0)
    st.session_state.latest_detections = detections
    st.session_state.latest_alerts = alerts
    st.session_state.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.recent_frames.append(st.session_state.latest_frame.copy())
    st.session_state.recent_frames = st.session_state.recent_frames[-40:]
    st.session_state.critical_beep = any(a["severity"] == "critical" for a in alerts)

    now_dt = datetime.now()
    if detections:
        st.session_state.latest_detection_age_s = 0.0
    elif st.session_state.latest_detection_age_s >= 0:
        st.session_state.latest_detection_age_s += max(0.0, st.session_state.latest_step_ms / 1000.0)

    if st.session_state.scenario_active:
        st.session_state.scenario_recorder.log_event(
            "cycle",
            f"detections={len(detections)} alerts={len(alerts)} temp={sensor.get('temperature', 0.0):.2f} gas={sensor.get('gas', 0.0):.2f}",
        )

    incident_path = save_incident_snapshot(st.session_state.latest_frame, alerts)
    incident_clip_path = ""
    if incident_path:
        incident_clip_path = save_incident_clip(st.session_state.recent_frames) or ""
    if incident_path:
        st.session_state.latest_incident_path = incident_path
        st.session_state.incident_history.append(incident_path)
        st.session_state.incident_history = st.session_state.incident_history[-500:]

    for item in queued_items:
        st.session_state.latest_alerts.append(
            {
                "timestamp": datetime.now(),
                "source": "trust-layer",
                "message": f"Human verification required for {item.get('alert', {}).get('code', 'UNKNOWN')}",
                "severity": "high",
                "color": "#ffd166",
                "code": "VERIFY_PENDING",
                "risk_score": float(item.get("alert", {}).get("risk_score", 0.0)),
            }
        )

    for alert in alerts:
        if str(alert.get("severity", "")).lower() not in {"high", "critical"}:
            continue
        st.session_state.incident_manager.open_or_update(
            alert,
            sensor=sensor,
            detections=detections,
            evidence_snapshot=incident_path or "",
            evidence_clip=incident_clip_path,
            timeline_note=f"alert {alert.get('code', 'UNKNOWN')} raised",
            sla_seconds=int(st.session_state.trust_config.get("sla_seconds", 120)),
            assigned_to=str(st.session_state.trust_config.get("default_assignee", "security-team")),
        )

    escalated = st.session_state.incident_manager.check_sla_escalations()
    for inc in escalated:
        alerts.append(
            {
                "timestamp": datetime.now(),
                "source": "incident-workflow",
                "message": f"Incident {inc.get('id')} SLA missed. Escalated to supervisor.",
                "severity": "critical",
                "color": "#ff3b30",
                "code": "INCIDENT_ESCALATION",
                "risk_score": 0.92,
            }
        )

    st.session_state.latest_alerts = alerts

    # Real-time email notifications for high intrusion alerts.
    if st.session_state.email_enabled:
        notifier = st.session_state.email_notifier
        notifier.cooldown_seconds = float(st.session_state.email_cooldown_seconds)

        cfg = EmailConfig(
            smtp_host=st.session_state.email_smtp_host,
            smtp_port=int(st.session_state.email_smtp_port),
            sender_email=st.session_state.email_sender,
            sender_password=st.session_state.email_password,
            receiver_email=st.session_state.email_receiver,
            use_tls=True,
        )

        for alert in alerts:
            if alert.get("code") == "INTRUSION" and alert.get("severity") in {"high", "critical"}:
                if cfg.sender_email and cfg.sender_password and cfg.receiver_email:
                    attachment_path = incident_path if st.session_state.email_attach_intrusion_image else None
                    ok, msg = notifier.send_intrusion_email(
                        cfg,
                        alert,
                        sensor=sensor,
                        detection_count=len(detections),
                        image_path=attachment_path,
                    )
                    # Compatibility fallback in case an older notifier object is still loaded.
                    # Streamlit can retain old module objects until a full server restart.
                    if not ok and "unexpected keyword argument 'image_path'" in str(msg):
                        ok, msg = notifier.send_intrusion_email(
                            cfg,
                            alert,
                            sensor=sensor,
                            detection_count=len(detections),
                        )
                    if ok:
                        if attachment_path:
                            st.session_state.email_status = "Intrusion email sent with image attachment"
                        else:
                            st.session_state.email_status = "Intrusion email sent"
                        st.session_state.email_last_sent = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elif msg != "cooldown-active":
                        st.session_state.email_status = f"Email error: {msg}"

    step_ms = (time.perf_counter() - step_start) * 1000.0
    st.session_state.latest_step_ms = step_ms
    st.session_state.latest_fps = 1000.0 / step_ms if step_ms > 0 else 0.0
    if float(st.session_state.monitoring_started_perf) > 0:
        st.session_state.latest_uptime_s = max(0.0, time.perf_counter() - float(st.session_state.monitoring_started_perf))
    else:
        st.session_state.latest_uptime_s = 0.0

    # Persist structured runtime logs for evaluation and demo evidence.
    try:
        now = now_dt
        logger = st.session_state.monitor_logger
        logger.log_cycle(
            now,
            sensor,
            env_pred,
            energy_pred,
            step_ms=st.session_state.latest_step_ms,
            fps=st.session_state.latest_fps,
            yolo_ms=st.session_state.latest_yolo_ms,
            detection_count=len(detections),
            alert_count=len(alerts),
            detection_enabled=bool(st.session_state.vision_detection_enabled),
            person_only_mode=bool(st.session_state.person_only_mode),
            uptime_s=float(st.session_state.latest_uptime_s),
            dropped_frames=int(st.session_state.dropped_frames),
            reconnect_count=int(st.session_state.reconnect_count),
            inference_ms=float(st.session_state.latest_inference_ms),
        )
        logger.log_detections(now, detections)
        logger.log_alerts(now, alerts)
    except Exception:
        # Logging should never stop monitoring.
        pass

    append_history(sensor, env_pred, energy_pred, usage)
    st.session_state.tick += 1


def run_raw_data_calibration(num_samples: int = 1000, epochs: int = 5) -> None:
    """Train both PINN trainers on raw records and store calibration report."""
    records = load_or_create_raw_dataset("data/raw_sensor_energy_1000.csv", num_samples=num_samples)
    report = calibrate_with_raw_dataset(
        st.session_state.env_pgnn,
        st.session_state.energy_pgnn,
        records,
        epochs=epochs,
    )
    st.session_state.calibration_report = report


def run_selected_datasets_calibration(selected_paths: list[str], per_dataset_limit: int = 1000, epochs: int = 4) -> None:
    """Calibrate PINNs on selected datasets with a bounded workload for responsiveness."""
    if not selected_paths:
        st.session_state.calibration_report = {
            "samples": 0.0,
            "epochs": float(epochs),
            "env_accuracy": 0.0,
            "energy_accuracy": 0.0,
            "meets_target": 0.0,
        }
        return

    rows = load_multiple_datasets(selected_paths, per_dataset_limit=per_dataset_limit)
    if not rows:
        st.session_state.calibration_report = {
            "samples": 0.0,
            "epochs": float(epochs),
            "env_accuracy": 0.0,
            "energy_accuracy": 0.0,
            "meets_target": 0.0,
        }
        return

    # Keep selected-dataset calibration interactive in Streamlit by capping total rows.
    max_total_rows = 1600
    if len(rows) > max_total_rows:
        rows = rows[:max_total_rows]

    report = calibrate_with_raw_dataset(
        st.session_state.env_pgnn,
        st.session_state.energy_pgnn,
        rows,
        epochs=epochs,
    )
    st.session_state.calibration_report = report


def summarize_dataset(path: str) -> dict:
    """Compute quick summary stats for one dataset file."""
    rows = load_dataset(path, limit=None)
    if not rows:
        return {"path": path, "rows": 0}

    temps = [r["temperature"] for r in rows]
    hums = [r["humidity"] for r in rows]
    gases = [r["gas"] for r in rows]
    return {
        "path": path,
        "rows": len(rows),
        "temp_mean": round(statistics.mean(temps), 2),
        "humidity_mean": round(statistics.mean(hums), 2),
        "gas_mean": round(statistics.mean(gases), 2),
    }


def get_latest_eval_summary(eval_dir: str = "data/eval") -> dict:
    base = Path(eval_dir)
    files = sorted(base.glob("evaluation_summary_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {}

    latest = files[0]
    out = {"_file": str(latest)}
    try:
        with latest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                metric = str(row[0]).strip()
                value = str(row[1]).strip()
                if metric and metric != "metric":
                    out[metric] = value
    except Exception:
        return {}

    required_metrics = {"false_alert_rate_proxy", "id_switch_proxy", "avg_track_confidence"}
    if not required_metrics.issubset(set(out.keys())):
        # Compatibility fallback: recompute current metrics from logs when viewing old summary files.
        try:
            result = evaluate_logs("data/logs")
            for k, v in result.summary.items():
                out[k] = str(v)
            out["_file"] = f"{out.get('_file', '')} (merged with live logs)"
        except Exception:
            pass
    return out


def render_status_chips() -> None:
    cam_ok = bool(st.session_state.camera_ok)
    detector_ok = st.session_state.detector is not None if st.session_state.vision_detection_enabled else True
    age = float(st.session_state.latest_detection_age_s)

    cam_color = "#1ec66b" if cam_ok else "#e05757"
    det_color = "#1ec66b" if detector_ok else "#e0a323"
    if age < 0:
        age_text = "No detections yet"
        age_color = "#8ea0be"
    elif age <= 5:
        age_text = f"Last detection {age:.1f}s ago"
        age_color = "#1ec66b"
    else:
        age_text = f"Last detection {age:.1f}s ago"
        age_color = "#e0a323"

    st.markdown(
        (
            f"<div class='panel'><strong>Status</strong><br>"
            f"<span style='display:inline-block;padding:4px 10px;border-radius:20px;background:{cam_color};margin-right:8px;'>"
            f"camera {'ok' if cam_ok else 'fallback'}</span>"
            f"<span style='display:inline-block;padding:4px 10px;border-radius:20px;background:{det_color};margin-right:8px;'>"
            f"detector {'ready' if detector_ok else 'not ready'}</span>"
            f"<span style='display:inline-block;padding:4px 10px;border-radius:20px;background:{age_color};'>"
            f"{age_text}</span></div>"
        ),
        unsafe_allow_html=True,
    )


def render_title() -> None:
    st.markdown('<div class="title">AI-Powered Smart Monitoring System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Computer Vision + PINN forecasting + intelligent risk alerts for real-time smart monitoring.</div>',
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    """Dashboard page with live feed and key system status."""
    left, right = st.columns([1.8, 1.0], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Live Vision Feed")
        st.image(st.session_state.latest_frame, channels="RGB", width="stretch")
        st.caption(f"Active detections: {len(st.session_state.latest_detections)}")
        st.markdown("</div>", unsafe_allow_html=True)

        detections = st.session_state.latest_detections
        person_speeds_px = [float(d.get("speed_px_s", 0.0)) for d in detections if d.get("class") == "person"]
        person_speeds_m = [float(d.get("speed_m_s", 0.0)) for d in detections if d.get("class") == "person"]
        longest_person_s = max([float(d.get("tracked_seconds", 0.0)) for d in detections if d.get("class") == "person"] + [0.0])
        track_conf = [float(d.get("track_confidence", 0.0)) for d in detections]
        detector = st.session_state.detector
        lost_total = int(getattr(detector, "track_lost_total", 0)) if detector else 0
        recovered_total = int(getattr(detector, "track_recovered_total", 0)) if detector else 0

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Tracking Analytics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Tracks", len(detections))
        c2.metric("Avg Person Speed (px/s)", f"{(sum(person_speeds_px) / max(1, len(person_speeds_px))):.1f}")
        if person_speeds_m and st.session_state.calibration_meters_per_pixel > 0:
            c3.metric("Avg Person Speed (m/s)", f"{(sum(person_speeds_m) / max(1, len(person_speeds_m))):.2f}")
        else:
            c3.metric("Longest Person Track", f"{longest_person_s:.1f}s")
        c4.metric("Avg Track Confidence", f"{(sum(track_conf) / max(1, len(track_conf))):.2f}")
        st.caption(f"Track lifecycle: lost={lost_total} | recovered={recovered_total}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        sensor = st.session_state.latest_sensor
        env = st.session_state.latest_env_pred
        energy = st.session_state.latest_energy_pred

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Temperature</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{sensor["temperature"]:.2f} C</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Humidity</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{sensor["humidity"]:.2f} %</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Gas</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{sensor["gas"]:.2f} ppm</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**PINN Snapshot**")
        env_acc = st.session_state.latest_env_accuracy
        energy_acc = st.session_state.latest_energy_accuracy
        has_pinn_data = bool(env) and bool(energy)
        if has_pinn_data:
            st.write(
                {
                    "env_total_loss": round(env.get("total_loss", 0.0), 6),
                    "energy_total_loss": round(energy.get("total_loss", 0.0), 6),
                    "pred_energy_kwh": round(energy.get("predicted_energy", 0.0), 4),
                    "expected_energy_kwh": round(energy.get("expected_energy", 0.0), 4),
                    "env_accuracy_percent": round(env_acc, 2),
                    "energy_accuracy_percent": round(energy_acc, 2),
                }
            )
            if env_acc >= 95.0 and energy_acc >= 95.0:
                st.success("Accuracy target met: both PINN modules are above 95%.")
            else:
                st.warning("Accuracy target not yet reached. Keep monitoring for more training steps.")
        else:
            st.info("PINN snapshot is empty. Click Start or run PINN training to generate values.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Real-Time Alerts")
    for alert in st.session_state.latest_alerts[-5:]:
        risk = float(alert.get("risk_score", 0.0))
        st.markdown(
            f'<div class="alert-box" style="border-left-color: {alert["color"]};">'
            f'<strong>{alert["severity"].upper()}</strong> - {alert["message"]}'
            f'<br><span style="opacity:0.9;">risk score: {risk:.2f}</span>'
            f'<br><span style="opacity:0.7;">{alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.latest_incident_path:
        incident_file = Path(st.session_state.latest_incident_path)
        if not incident_file.exists():
            st.session_state.latest_incident_path = ""
        else:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Latest Incident Snapshot")
            st.image(str(incident_file), width="stretch")
            st.caption(st.session_state.latest_incident_path)
            st.markdown("</div>", unsafe_allow_html=True)


def render_review_mode() -> None:
    """Clean, presenter-friendly view with only critical review metrics."""
    render_status_chips()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Review Mode - Key Live Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detections", len(st.session_state.latest_detections))
    c2.metric("Env Accuracy", f"{st.session_state.latest_env_accuracy:.2f}%")
    c3.metric("Energy Accuracy", f"{st.session_state.latest_energy_accuracy:.2f}%")
    c4.metric("FPS", f"{st.session_state.latest_fps:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Latest Evaluation Summary")
    summary = get_latest_eval_summary("data/eval")
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("False Alert Proxy", f"{float(summary.get('false_alert_rate_proxy', 0.0)):.3f}")
        c2.metric("ID Switch Proxy", f"{float(summary.get('id_switch_proxy', 0.0)):.3f}")
        c3.metric("Track Confidence", f"{float(summary.get('avg_track_confidence', 0.0)):.3f}")
        st.caption(f"Source: {summary.get('_file', '')}")
    else:
        st.info("No evaluation summary found yet. Click Generate Evaluation Report in sidebar.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Live Feed")
    st.image(st.session_state.latest_frame, channels="RGB", width="stretch")
    st.caption(f"Mode: {'Person-only' if st.session_state.person_only_mode else 'All objects'} | Scale: {st.session_state.calibration_meters_per_pixel:.6f} m/px")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Latest 5 Alerts")
    latest = st.session_state.alert_manager.latest(5)
    if latest:
        for a in reversed(latest):
            st.write(f"[{a['severity'].upper()}] {a.get('code','')} | risk={a.get('risk_score',0.0):.2f} | {a['message']}")
    else:
        st.info("No alerts yet")
    st.markdown("</div>", unsafe_allow_html=True)


def render_environment_page() -> None:
    """Environment monitoring page with actual and PINN-predicted trends."""
    hist = st.session_state.history
    if not hist["temperature"]:
        st.info("Start monitoring to populate environment charts in real time.")
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Temperature Trend")
    st.line_chart(
        {
            "temperature": hist["temperature"],
            "pinn_prediction": hist["pred_temperature"],
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Humidity Trend")
        st.line_chart({"humidity": hist["humidity"], "pinn_prediction": hist["pred_humidity"]})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Gas Level Trend")
        st.line_chart({"gas": hist["gas"], "pinn_prediction": hist["pred_gas"]})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Environment Prediction Accuracy (%)")
    st.line_chart({"env_accuracy": hist["env_accuracy"]})
    st.markdown("</div>", unsafe_allow_html=True)


def render_energy_page() -> None:
    """Energy monitoring page with PINN predictions and usage behavior."""
    hist = st.session_state.history
    if not hist["pred_energy"]:
        st.info("Start monitoring to populate energy charts in real time.")
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Energy Prediction (kWh)")
    st.line_chart(
        {
            "predicted_energy": hist["pred_energy"],
            "expected_physics_energy": hist["expected_energy"],
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Device Usage Trend")
        st.line_chart({"usage_ratio": hist["usage"]})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Estimated Power (kW)")
        st.line_chart({"power_kw": hist["power_kw"]})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Energy Prediction Accuracy (%)")
    st.line_chart({"energy_accuracy": hist["energy_accuracy"]})
    st.markdown("</div>", unsafe_allow_html=True)


def render_alerts_page() -> None:
    """List all recent alerts with severity colors and timestamps."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Triggered Alerts")

    latest = st.session_state.alert_manager.latest(120)
    if not latest:
        st.info("No alerts generated yet.")
    else:
        for alert in reversed(latest):
            risk = float(alert.get("risk_score", 0.0))
            st.markdown(
                f'<div class="alert-box" style="border-left-color: {alert["color"]};">'
                f'<strong>{alert["severity"].upper()}</strong> - {alert["message"]}'
                f'<br><span style="opacity:0.85;">risk score: {risk:.2f} | code: {alert.get("code", "")}</span>'
                f'<br><span style="opacity:0.7;">{alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S")} | source: {alert["source"]}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_incident_center() -> None:
    """Production incident operations: verification queue and lifecycle actions."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Human Verification Queue")
    pending = [q for q in st.session_state.trust_layer.verification_queue if q.get("status") == "pending"]
    if not pending:
        st.info("No pending items")
    else:
        for item in pending[-20:]:
            alert = dict(item.get("alert", {}))
            c1, c2 = st.columns([4, 2])
            with c1:
                st.write(
                    f"{item.get('id')} | {alert.get('code', 'UNKNOWN')} | "
                    f"severity={alert.get('severity', 'n/a')} | risk={float(alert.get('risk_score', 0.0)):.2f}"
                )
            with c2:
                a1, a2 = st.columns(2)
                with a1:
                    if st.button("Approve", key=f"approve_{item.get('id')}", width="stretch"):
                        st.session_state.resolved_alerts.append({"id": item.get("id"), "approved": True})
                with a2:
                    if st.button("Reject", key=f"reject_{item.get('id')}", width="stretch"):
                        st.session_state.resolved_alerts.append({"id": item.get("id"), "approved": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Incident Lifecycle")
    incidents = st.session_state.incident_manager.latest(50)
    if not incidents:
        st.info("No incidents yet")
    else:
        open_count = sum(1 for i in incidents if i.get("status") in {"open", "acknowledged", "assigned"})
        resolved_count = sum(1 for i in incidents if i.get("status") == "resolved")
        c1, c2 = st.columns(2)
        c1.metric("Open Incidents", open_count)
        c2.metric("Resolved Incidents", resolved_count)

        incident_ids = [i.get("id", "") for i in reversed(incidents) if i.get("id")]
        selected = st.selectbox("Select incident", options=incident_ids)
        selected_incident = next((i for i in incidents if i.get("id") == selected), None)
        if selected_incident:
            st.write(
                {
                    "id": selected_incident.get("id"),
                    "code": selected_incident.get("code"),
                    "status": selected_incident.get("status"),
                    "severity": selected_incident.get("severity"),
                    "assigned_to": selected_incident.get("assigned_to"),
                    "created_at": selected_incident.get("created_at"),
                    "acknowledged_at": selected_incident.get("acknowledged_at"),
                    "resolved_at": selected_incident.get("resolved_at"),
                    "sla_seconds": selected_incident.get("sla_seconds"),
                    "escalated": selected_incident.get("escalated"),
                    "root_cause": selected_incident.get("root_cause"),
                }
            )

            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("Acknowledge", key=f"ack_{selected}", width="stretch"):
                    st.session_state.incident_manager.acknowledge(selected, actor="operator")
            with ac2:
                assignee = st.text_input("Assign to", value=selected_incident.get("assigned_to", "security-team"), key=f"assign_{selected}")
                if st.button("Assign", key=f"assign_btn_{selected}", width="stretch"):
                    st.session_state.incident_manager.assign(selected, assignee=assignee)
            with ac3:
                st.session_state.incident_root_cause = st.text_input(
                    "Root cause",
                    value=selected_incident.get("root_cause", "unknown"),
                    key=f"root_{selected}",
                )
                st.session_state.incident_feedback_note = st.text_input(
                    "Resolution note",
                    value=selected_incident.get("feedback", ""),
                    key=f"feedback_{selected}",
                )
                if st.button("Resolve", key=f"resolve_{selected}", width="stretch"):
                    st.session_state.incident_manager.resolve(
                        selected,
                        root_cause=st.session_state.incident_root_cause,
                        feedback=st.session_state.incident_feedback_note,
                    )

            evidence = dict(selected_incident.get("evidence", {}))
            snapshots = [p for p in list(evidence.get("snapshot_paths", [])) if p and Path(p).exists()]
            clips = [p for p in list(evidence.get("clip_paths", [])) if p and Path(p).exists()]
            if snapshots:
                p = Path(snapshots[-1])
                if p.exists():
                    st.image(str(p), width="stretch")
            if clips:
                st.caption(f"Latest clip: {clips[-1]}")

            st.markdown("**Timeline**")
            for event in reversed(list(selected_incident.get("timeline", []))[-15:]):
                st.write(f"{event.get('timestamp')} | {event.get('status')} | {event.get('event')}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_data_lab_page() -> None:
    """Data lab page for dataset inspection and curation."""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Available Datasets")

    dataset_paths = st.session_state.dataset_paths
    if not dataset_paths:
        st.info("No dataset files found in data folder.")
    else:
        for p in dataset_paths:
            summary = summarize_dataset(p)
            st.write(summary)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    if st.session_state.selected_datasets:
        sample_rows = load_dataset(st.session_state.selected_datasets[0], limit=10)
        st.write(sample_rows)
    else:
        st.info("Select a dataset in sidebar to preview records.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_system_health_page() -> None:
    """System runtime health metrics page."""
    hist = st.session_state.history

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Runtime Metrics")
    st.write(
        {
            "latest_step_ms": round(st.session_state.latest_step_ms, 2),
            "latest_fps": round(st.session_state.latest_fps, 2),
            "latest_yolo_ms": round(st.session_state.latest_yolo_ms, 2),
            "latest_inference_ms": round(st.session_state.latest_inference_ms, 2),
            "uptime_seconds": round(st.session_state.latest_uptime_s, 2),
            "dropped_frames": int(st.session_state.dropped_frames),
            "reconnect_count": int(st.session_state.reconnect_count),
            "capture_fail_streak": int(st.session_state.capture_fail_streak),
            "last_reconnect_at": st.session_state.last_reconnect_at,
            "monitoring": st.session_state.monitoring,
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if hist["step_ms"]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Step Latency (ms)")
        st.line_chart({"step_ms": hist["step_ms"]})
        st.markdown("</div>", unsafe_allow_html=True)

    if hist["fps"]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Throughput (FPS)")
        st.line_chart({"fps": hist["fps"]})
        st.markdown("</div>", unsafe_allow_html=True)

    if hist["yolo_ms"]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("YOLO Inference Latency (ms)")
        st.line_chart({"yolo_ms": hist["yolo_ms"]})
        st.markdown("</div>", unsafe_allow_html=True)

    if hist["dropped_frames"]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Capture Reliability")
        st.line_chart(
            {
                "dropped_frames_total": hist["dropped_frames"],
                "reconnect_count_total": hist["reconnect_count"],
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if hist["uptime_s"]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Monitoring Uptime (s)")
        st.line_chart({"uptime_s": hist["uptime_s"]})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Incident Snapshots")
    prune_stale_incident_paths(max_items=500)
    if st.session_state.incident_history:
        for p in reversed(st.session_state.incident_history[-20:]):
            st.write(p)
    else:
        st.info("No incident snapshots captured yet.")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    init_state()

    if not st.session_state.calibration_loaded:
        cal = load_calibration("data/calibration.json")
        st.session_state.calibration_known_distance_m = float(cal.get("known_distance_m", 1.0))
        st.session_state.calibration_pixel_distance_px = float(cal.get("pixel_distance_px", 100.0))
        st.session_state.calibration_meters_per_pixel = float(cal.get("meters_per_pixel", 0.0))
        st.session_state.calibration_loaded = True

    if not st.session_state.alert_policy_loaded:
        st.session_state.alert_policy = load_policy("data/alert_policy.json")
        effective = resolve_policy_context(
            st.session_state.alert_policy,
            st.session_state.policy_site,
            st.session_state.policy_shift,
            st.session_state.policy_risk_profile,
        )
        apply_policy(st.session_state.alert_manager, effective)
        st.session_state.alert_policy_loaded = True

    st.session_state.dataset_paths = ensure_default_datasets("data", num_samples=1000)
    st.session_state.dataset_paths = list_dataset_paths("data")

    with st.sidebar:
        page = st.radio(
            "Navigate",
            [
                "Review Mode",
                "Dashboard",
                "Environment Monitoring",
                "Energy Monitoring",
                "Alerts Panel",
                "Incident Center",
                "Data Lab",
                "System Health",
            ],
        )

        st.markdown("**Production Policy Engine**")
        policy = dict(st.session_state.alert_policy)
        policy_global = dict(policy.get("global", {}))
        policy_codes = dict(policy.get("alert_codes", {}))
        policy_contexts = dict(policy.get("contexts", {}))

        st.session_state.policy_site = st.text_input("Site", value=st.session_state.policy_site)
        st.session_state.policy_shift = st.selectbox("Shift", options=["day", "night", "all"], index=["day", "night", "all"].index(st.session_state.policy_shift if st.session_state.policy_shift in {"day", "night", "all"} else "day"))
        st.session_state.policy_risk_profile = st.selectbox(
            "Risk profile",
            options=["normal", "strict", "relaxed"],
            index=["normal", "strict", "relaxed"].index(st.session_state.policy_risk_profile if st.session_state.policy_risk_profile in {"normal", "strict", "relaxed"} else "normal"),
        )

        policy_global["cooldown_seconds"] = float(st.slider("Global cooldown (seconds)", 3, 180, int(policy_global.get("cooldown_seconds", 10.0)), 1))
        policy_global["intrusion_clear_seconds"] = float(st.slider("Intrusion clear (seconds)", 2, 45, int(policy_global.get("intrusion_clear_seconds", 8.0)), 1))
        policy_global["gas_threshold"] = float(st.slider("Gas threshold (ppm)", 200, 900, int(policy_global.get("gas_threshold", 450.0)), 5))
        policy_global["temp_threshold"] = float(st.slider("Temperature threshold (C)", 20, 80, int(policy_global.get("temp_threshold", 42.0)), 1))
        policy_global["gas_clear_ratio"] = float(st.slider("Gas clear ratio", 0.70, 1.00, float(policy_global.get("gas_clear_ratio", 0.90)), 0.01))
        policy_global["temp_clear_delta"] = float(st.slider("Temperature clear delta (C)", 0.5, 10.0, float(policy_global.get("temp_clear_delta", 2.0)), 0.5))
        policy_global["normal_after_quiet_seconds"] = float(st.slider("Normal-status quiet time (seconds)", 5, 180, int(policy_global.get("normal_after_quiet_seconds", 20.0)), 1))

        code_options = ["INTRUSION", "FIRE", "GAS_HIGH", "TEMP_HIGH", "NORMAL"]
        selected_code = st.selectbox("Policy code", options=code_options)
        selected_cfg = dict(policy_codes.get(selected_code, {}))
        selected_cfg["cooldown_seconds"] = float(st.slider(f"{selected_code} cooldown (seconds)", 1, 300, int(selected_cfg.get("cooldown_seconds", 10.0)), 1))
        selected_cfg["min_risk"] = float(st.slider(f"{selected_code} min risk", 0.0, 1.0, float(selected_cfg.get("min_risk", 0.5)), 0.01))
        selected_cfg["confirmation_frames"] = int(st.slider(f"{selected_code} confirmation frames", 1, 10, int(selected_cfg.get("confirmation_frames", 1)), 1))
        selected_cfg["escalate_after_seconds"] = float(st.slider(f"{selected_code} escalate after (seconds)", 0, 600, int(selected_cfg.get("escalate_after_seconds", 60.0)), 5))
        selected_cfg["severity"] = st.selectbox(
            f"{selected_code} severity",
            options=["low", "medium", "high", "critical"],
            index=["low", "medium", "high", "critical"].index(selected_cfg.get("severity", "high") if selected_cfg.get("severity", "high") in {"low", "medium", "high", "critical"} else "high"),
        )
        policy_codes[selected_code] = selected_cfg

        default_ctx = dict(policy_contexts.get("default", {}))
        default_ctx["site"] = st.session_state.policy_site
        default_ctx["shift"] = st.session_state.policy_shift
        default_ctx["risk_profile"] = st.session_state.policy_risk_profile
        default_ctx["global_overrides"] = dict(default_ctx.get("global_overrides", {}))
        default_ctx["code_overrides"] = dict(default_ctx.get("code_overrides", {}))
        policy_contexts["default"] = default_ctx

        policy = {"global": policy_global, "alert_codes": policy_codes, "contexts": policy_contexts}
        st.session_state.alert_policy = policy

        effective = resolve_policy_context(
            policy,
            st.session_state.policy_site,
            st.session_state.policy_shift,
            st.session_state.policy_risk_profile,
        )
        apply_policy(st.session_state.alert_manager, effective)

        st.session_state.policy_requester = st.text_input("Policy requester", value=st.session_state.policy_requester)
        st.session_state.policy_approver = st.text_input("Policy approver", value=st.session_state.policy_approver)

        pa1, pa2 = st.columns(2)
        with pa1:
            if st.button("Submit Policy Change", width="stretch"):
                change_id = create_policy_change_request("data/alert_policy.json", policy, requested_by=st.session_state.policy_requester)
                st.success(f"Policy change submitted: {change_id}")
        with pa2:
            if st.button("Save Policy Directly", width="stretch"):
                save_policy("data/alert_policy.json", policy)
                st.success("Policy saved.")

        pending = list_pending_policy_changes("data/alert_policy.json")
        pending_ids = [p.get("change_id", "") for p in pending if p.get("status") == "pending"]
        if pending_ids:
            st.session_state.policy_pending_choice = st.selectbox("Pending change", options=pending_ids)
            if st.button("Approve Pending Change", width="stretch"):
                ok, msg = approve_policy_change(
                    "data/alert_policy.json",
                    st.session_state.policy_pending_choice,
                    approved_by=st.session_state.policy_approver,
                )
                if ok:
                    st.success(msg)
                    st.session_state.alert_policy = load_policy("data/alert_policy.json")
                else:
                    st.warning(msg)

        versions = list_policy_versions("data/alert_policy.json")
        if versions:
            st.session_state.policy_version_choice = st.selectbox("Policy versions", options=versions)
            if st.button("Rollback Selected Version", width="stretch"):
                ok, msg = rollback_policy("data/alert_policy.json", st.session_state.policy_version_choice)
                if ok:
                    st.success(msg)
                    st.session_state.alert_policy = load_policy("data/alert_policy.json")
                else:
                    st.warning(msg)

        st.markdown("**Trust Layer**")
        trust = dict(st.session_state.trust_config)
        active = dict(trust.get("active_hours", {"start": 0, "end": 24}))
        active["start"] = int(st.slider("Active hour start", 0, 23, int(active.get("start", 0)), 1))
        active["end"] = int(st.slider("Active hour end", 1, 24, int(active.get("end", 24)), 1))
        trust["active_hours"] = active

        zone = dict(trust.get("restricted_zone", {}))
        zone["x1"] = float(st.slider("Zone x1", 0.0, 1.0, float(zone.get("x1", 0.25)), 0.01))
        zone["y1"] = float(st.slider("Zone y1", 0.0, 1.0, float(zone.get("y1", 0.20)), 0.01))
        zone["x2"] = float(st.slider("Zone x2", 0.0, 1.0, float(zone.get("x2", 0.75)), 0.01))
        zone["y2"] = float(st.slider("Zone y2", 0.0, 1.0, float(zone.get("y2", 0.90)), 0.01))
        trust["restricted_zone"] = zone

        trust["occupancy_limit"] = int(st.slider("Occupancy limit", 1, 20, int(trust.get("occupancy_limit", 3)), 1))
        trust["sla_seconds"] = int(st.slider("Incident SLA (seconds)", 30, 1800, int(trust.get("sla_seconds", 120)), 30))
        trust["default_assignee"] = st.text_input("Default assignee", value=str(trust.get("default_assignee", "security-team")))
        trust["require_human_verification_codes"] = st.multiselect(
            "Require human verification for",
            options=["FIRE", "GAS_HIGH", "INTRUSION", "TEMP_HIGH"],
            default=list(trust.get("require_human_verification_codes", ["FIRE", "GAS_HIGH"])),
        )
        st.session_state.trust_config = trust

        st.markdown("**Email Alerts (Real-Time Intrusion)**")
        st.session_state.email_enabled = st.checkbox("Enable intrusion email", value=st.session_state.email_enabled)
        st.session_state.email_smtp_host = st.text_input("SMTP Host", value=st.session_state.email_smtp_host)
        st.session_state.email_smtp_port = st.number_input(
            "SMTP Port",
            min_value=1,
            max_value=65535,
            value=int(st.session_state.email_smtp_port),
            step=1,
        )
        st.session_state.email_sender = st.text_input("Sender Email", value=st.session_state.email_sender)
        st.session_state.email_password = st.text_input(
            "Sender App Password",
            value=st.session_state.email_password,
            type="password",
        )
        st.session_state.email_receiver = st.text_input("Receiver Email", value=st.session_state.email_receiver)
        st.session_state.email_cooldown_seconds = st.slider(
            "Email cooldown (seconds)",
            min_value=10,
            max_value=300,
            value=int(st.session_state.email_cooldown_seconds),
            step=5,
            help="Minimum time between repeated intrusion emails.",
        )
        st.session_state.email_attach_intrusion_image = st.checkbox(
            "Attach captured intrusion image",
            value=st.session_state.email_attach_intrusion_image,
        )

        if st.button("Send Test Email", width="stretch"):
            cfg = EmailConfig(
                smtp_host=st.session_state.email_smtp_host,
                smtp_port=int(st.session_state.email_smtp_port),
                sender_email=st.session_state.email_sender,
                sender_password=st.session_state.email_password,
                receiver_email=st.session_state.email_receiver,
                use_tls=True,
            )
            test_alert = {
                "code": "INTRUSION",
                "severity": "high",
                "message": "Test intrusion alert from Smart Monitoring System.",
                "risk_score": 0.75,
            }
            ok, msg = st.session_state.email_notifier.send_intrusion_email(
                cfg,
                test_alert,
                sensor=st.session_state.latest_sensor,
                detection_count=len(st.session_state.latest_detections),
                image_path=st.session_state.latest_incident_path if st.session_state.email_attach_intrusion_image else None,
            )
            if not ok and "unexpected keyword argument 'image_path'" in str(msg):
                ok, msg = st.session_state.email_notifier.send_intrusion_email(
                    cfg,
                    test_alert,
                    sensor=st.session_state.latest_sensor,
                    detection_count=len(st.session_state.latest_detections),
                )
            if ok:
                st.session_state.email_status = "Test email sent"
                st.session_state.email_last_sent = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.session_state.email_status = f"Test email failed: {msg}"

        if st.session_state.email_status:
            st.caption(f"Email status: {st.session_state.email_status}")
        if st.session_state.email_last_sent:
            st.caption(f"Last email sent at: {st.session_state.email_last_sent}")

        st.caption("PINN is active for sensor/energy prediction.")

        st.session_state.vision_detection_enabled = st.checkbox(
            "Enable object detection",
            value=st.session_state.vision_detection_enabled,
        )
        st.session_state.person_only_mode = st.checkbox(
            "Person-only detection mode",
            value=st.session_state.person_only_mode,
            help="Detect and track only persons, with speed and tracked-time overlays.",
        )
        if st.session_state.person_only_mode:
            st.caption("Person-only mode is ON. Phone/paper/object labels are disabled.")
        st.session_state.vision_profile = st.selectbox(
            "Detection profile",
            options=["Comprehensive", "Balanced"],
            index=0 if st.session_state.vision_profile == "Comprehensive" else 1,
            help="Comprehensive detects all classes and living organisms with higher recall.",
        )
        st.session_state.detect_all_objects = st.checkbox(
            "Detect all objects (all YOLO classes)",
            value=st.session_state.detect_all_objects,
            disabled=bool(st.session_state.person_only_mode),
        )
        st.session_state.vision_min_confidence = st.slider(
            "Detection confidence threshold",
            min_value=0.10,
            max_value=0.80,
            value=float(st.session_state.vision_min_confidence),
            step=0.05,
            help="Lower values detect more objects, but can increase false positives.",
        )
        st.markdown("**Class-Specific Thresholds**")
        st.session_state.person_threshold = st.slider(
            "Person threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(st.session_state.person_threshold),
            step=0.01,
        )
        st.session_state.object_threshold = st.slider(
            "Object threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(st.session_state.object_threshold),
            step=0.01,
        )
        st.session_state.paper_threshold = st.slider(
            "Paper threshold",
            min_value=0.05,
            max_value=0.90,
            value=float(st.session_state.paper_threshold),
            step=0.01,
        )
        st.session_state.fire_threshold = st.slider(
            "Fire threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(st.session_state.fire_threshold),
            step=0.01,
        )
        st.session_state.vision_stride = st.slider(
            "Detection stride (higher = lighter CPU)",
            min_value=1,
            max_value=6,
            value=int(st.session_state.vision_stride),
            step=1,
            help="YOLO runs every N cycles.",
        )

        st.markdown("**Speed Calibration (m/s)**")
        st.session_state.calibration_known_distance_m = st.number_input(
            "Known distance (meters)",
            min_value=0.1,
            max_value=100.0,
            value=float(st.session_state.calibration_known_distance_m),
            step=0.1,
        )
        st.session_state.calibration_pixel_distance_px = st.number_input(
            "Measured pixel distance",
            min_value=1.0,
            max_value=5000.0,
            value=float(st.session_state.calibration_pixel_distance_px),
            step=1.0,
        )
        st.caption("Optional two-point calibration from image coordinates")
        cx1, cy1 = st.columns(2)
        with cx1:
            st.session_state.calibration_x1 = st.number_input(
                "Point1 X",
                min_value=0,
                max_value=5000,
                value=int(st.session_state.calibration_x1),
                step=1,
            )
            st.session_state.calibration_x2 = st.number_input(
                "Point2 X",
                min_value=0,
                max_value=5000,
                value=int(st.session_state.calibration_x2),
                step=1,
            )
        with cy1:
            st.session_state.calibration_y1 = st.number_input(
                "Point1 Y",
                min_value=0,
                max_value=5000,
                value=int(st.session_state.calibration_y1),
                step=1,
            )
            st.session_state.calibration_y2 = st.number_input(
                "Point2 Y",
                min_value=0,
                max_value=5000,
                value=int(st.session_state.calibration_y2),
                step=1,
            )
        if st.button("Compute Pixel Distance from Points", width="stretch"):
            dx = float(st.session_state.calibration_x2 - st.session_state.calibration_x1)
            dy = float(st.session_state.calibration_y2 - st.session_state.calibration_y1)
            dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
            st.session_state.calibration_pixel_distance_px = dist
            st.success(f"Computed pixel distance: {dist:.2f}")
        if st.button("Save Calibration", width="stretch"):
            st.session_state.calibration_meters_per_pixel = save_calibration(
                "data/calibration.json",
                st.session_state.calibration_known_distance_m,
                st.session_state.calibration_pixel_distance_px,
            )
            st.success("Calibration saved.")
        st.caption(f"Current scale: {st.session_state.calibration_meters_per_pixel:.6f} m/px")

        fire_model_path = st.text_input(
            "Optional Fire Model Path (.pt)",
            value=st.session_state.fire_model_path,
            help="Optional custom fire/smoke model.",
        )
        st.session_state.fire_model_path = fire_model_path

        video_mode = st.selectbox("Video Source", ["Webcam", "Upload Video"], index=0)
        uploaded_file = None
        if video_mode == "Upload Video":
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        st.session_state.video_mode = video_mode

        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("Start", width="stretch"):
                checks = []
                release_capture()
                try:
                    prepare_capture(video_mode, uploaded_file)
                except Exception as exc:
                    checks.append(("video_source", False, f"Video source issue: {exc}"))

                cap = st.session_state.capture
                camera_ready = bool(cap is not None and cap.isOpened())
                checks.append(("camera_opened", camera_ready, "Camera or video stream ready" if camera_ready else "Camera/video source is unavailable"))

                if st.session_state.vision_detection_enabled:
                    st.session_state.detector = None
                    prepare_detector(
                        st.session_state.fire_model_path,
                        min_confidence=float(st.session_state.vision_min_confidence),
                        detect_all_objects=bool(st.session_state.detect_all_objects),
                        detection_profile=st.session_state.vision_profile.lower(),
                        person_only_mode=bool(st.session_state.person_only_mode),
                        person_threshold=float(st.session_state.person_threshold),
                        object_threshold=float(st.session_state.object_threshold),
                        paper_threshold=float(st.session_state.paper_threshold),
                        fire_threshold=float(st.session_state.fire_threshold),
                    )
                    detector_ready = st.session_state.detector is not None
                    checks.append(
                        (
                            "detector_loaded",
                            detector_ready,
                            "Detector initialized" if detector_ready else f"Detector unavailable: {st.session_state.detector_error or 'unknown error'}",
                        )
                    )
                else:
                    checks.append(("detector_loaded", True, "Detector disabled by user"))

                policy_ready = bool(st.session_state.alert_policy_loaded and st.session_state.alert_policy)
                checks.append(("policy_loaded", policy_ready, "Alert policy loaded" if policy_ready else "Alert policy missing"))

                logs_ready = check_logs_writable("data/logs")
                checks.append(("logs_writable", logs_ready, "Log folder writable" if logs_ready else "Cannot write to data/logs"))

                st.session_state.preflight_last_checks = [
                    {"name": name, "ok": ok, "detail": detail} for name, ok, detail in checks
                ]

                if not all(ok for _, ok, _ in checks):
                    st.session_state.monitoring = False
                    st.session_state.camera_ok = camera_ready
                    st.error("Preflight failed. Resolve checklist items before starting monitoring.")
                    for _, ok, detail in checks:
                        if not ok:
                            st.warning(detail)
                else:
                    st.session_state.capture_fail_streak = 0
                    st.session_state.monitoring_started_perf = time.perf_counter()
                    st.session_state.latest_uptime_s = 0.0

                    if not st.session_state.pinn_warm_started:
                        warm_paths = (
                            st.session_state.selected_datasets
                            if st.session_state.selected_datasets
                            else st.session_state.dataset_paths[:1]
                        )
                        if warm_paths:
                            with st.spinner("Warm-starting PINN model..."):
                                run_selected_datasets_calibration(
                                    warm_paths,
                                    per_dataset_limit=400,
                                    epochs=2,
                                )
                            st.session_state.pinn_warm_started = True
                    st.session_state.monitoring = True

        with stop_col:
            if st.button("Stop", width="stretch"):
                st.session_state.monitoring = False
                st.session_state.monitoring_started_perf = 0.0
                release_capture()

        if st.session_state.preflight_last_checks:
            with st.expander("Preflight Checklist", expanded=False):
                for item in st.session_state.preflight_last_checks:
                    status = "[OK]" if bool(item.get("ok")) else "[FAIL]"
                    st.write(f"{status} {item.get('name')}: {item.get('detail')}")

        if st.button("Reset History", width="stretch"):
            st.session_state.history = {k: [] for k in st.session_state.history}
            st.session_state.alert_manager.alert_history = []
            st.session_state.incident_history = []
            st.session_state.latest_incident_path = ""

        st.markdown("**Dataset Controls**")
        selected = st.multiselect(
            "Select datasets for training",
            options=st.session_state.dataset_paths,
            default=st.session_state.selected_datasets if st.session_state.selected_datasets else st.session_state.dataset_paths[:1],
        )
        st.session_state.selected_datasets = selected

        uploaded_dataset = st.file_uploader("Upload dataset CSV", type=["csv"], key="dataset_upload")
        if uploaded_dataset is not None:
            target = Path("data") / uploaded_dataset.name
            target.write_bytes(uploaded_dataset.getbuffer())
            st.session_state.dataset_paths = list_dataset_paths("data")
            st.success(f"Uploaded dataset: {target}")

        if st.button("Train PINN with Selected Datasets", width="stretch"):
            was_monitoring = st.session_state.monitoring
            st.session_state.monitoring = False
            with st.spinner("Training PINN using selected datasets..."):
                run_selected_datasets_calibration(
                    st.session_state.selected_datasets,
                    per_dataset_limit=600,
                    epochs=2,
                )
            if was_monitoring:
                st.session_state.monitoring = True

        st.session_state.play_sound = st.checkbox("Enable sound alert", value=True)

        if st.button("Train with 1000 Raw Samples", width="stretch"):
            was_monitoring = st.session_state.monitoring
            st.session_state.monitoring = False
            with st.spinner("Calibrating PINN models with 1000 raw samples..."):
                run_raw_data_calibration(num_samples=1000, epochs=4)
            if was_monitoring:
                st.session_state.monitoring = True

        report = st.session_state.calibration_report
        if report:
            st.markdown("**Raw Data Calibration Report**")
            st.write(
                {
                    "samples": int(report.get("samples", 0)),
                    "epochs": int(report.get("epochs", 0)),
                    "env_accuracy_percent": report.get("env_accuracy", 0.0),
                    "energy_accuracy_percent": report.get("energy_accuracy", 0.0),
                }
            )
            if report.get("meets_target", 0.0) >= 1.0:
                st.success("Calibration complete: both models are above 95%.")
            else:
                st.warning("Calibration done, but one of the models is below 95%. Run calibration again.")

        if st.session_state.detector_error:
            st.warning("Object detection unavailable: " + st.session_state.detector_error)

        st.download_button(
            "Download Monitoring Report",
            data=build_report_bytes(),
            file_name=f"smart_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )

        if st.button("Generate Evaluation Report", width="stretch"):
            result = evaluate_logs("data/logs")
            csv_path, txt_path = export_evaluation(result, "data/eval")
            st.success(f"Evaluation generated: {csv_path}")
            st.caption(f"Text report: {txt_path}")

        st.markdown("**Scenario Recorder**")
        st.session_state.scenario_name = st.text_input("Scenario name", value=st.session_state.scenario_name)
        st.session_state.scenario_mark_text = st.text_input("Mark event text", value=st.session_state.scenario_mark_text)
        s1, s2 = st.columns(2)
        with s1:
            if st.button("Start Scenario", width="stretch"):
                file_path = st.session_state.scenario_recorder.start(st.session_state.scenario_name)
                st.session_state.scenario_active = True
                st.session_state.scenario_file = file_path
                st.success(f"Scenario recording: {file_path}")
            if st.button("Mark Event", width="stretch"):
                st.session_state.scenario_recorder.log_event("marker", st.session_state.scenario_mark_text or "manual marker")
        with s2:
            if st.button("Stop Scenario", width="stretch"):
                file_path = st.session_state.scenario_recorder.stop()
                st.session_state.scenario_active = False
                st.session_state.scenario_file = file_path
                st.info(f"Scenario saved: {file_path}")

        if st.button("Export Review Bundle", width="stretch"):
            out_dir, summary = create_review_bundle("data")
            st.success(f"Review bundle exported: {out_dir}")
            st.caption(f"Files: logs={len(summary.get('logs', []))}, eval={len(summary.get('evaluation', []))}, incidents={len(summary.get('incidents', []))}")

        status = "RUNNING" if st.session_state.monitoring else "STOPPED"
        st.markdown(f"System Status: **{status}**")

    if st.session_state.monitoring:
        run_monitor_step()

    render_title()

    if page == "Review Mode":
        render_review_mode()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Environment Monitoring":
        render_environment_page()
    elif page == "Energy Monitoring":
        render_energy_page()
    elif page == "Data Lab":
        render_data_lab_page()
    elif page == "Incident Center":
        render_incident_center()
    elif page == "System Health":
        render_system_health_page()
    else:
        render_alerts_page()

    if st.session_state.monitoring and st.session_state.critical_beep and st.session_state.play_sound:
        st.audio(generate_beep_wav(), format="audio/wav")

    if st.session_state.monitoring:
        # Trigger smooth pseudo-streaming updates.
        time.sleep(0.08)
        st.rerun()


if __name__ == "__main__":
    main()
