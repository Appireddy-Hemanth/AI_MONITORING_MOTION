from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


class EvaluationResult:
    def __init__(
        self,
        summary: dict,
        per_class_speed: dict,
        detection_counts: dict,
        alert_counts: dict,
        alert_precision_proxy: dict,
    ) -> None:
        self.summary = summary
        self.per_class_speed = per_class_speed
        self.detection_counts = detection_counts
        self.alert_counts = alert_counts
        self.alert_precision_proxy = alert_precision_proxy


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def evaluate_logs(log_dir: str = "data/logs") -> EvaluationResult:
    base = Path(log_dir)
    today = datetime.now().strftime("%Y-%m-%d")

    cycles_rows = _read_csv_rows(base / f"cycles_{today}.csv")
    detections_rows = _read_csv_rows(base / f"detections_{today}.csv")
    alerts_rows = _read_csv_rows(base / f"alerts_{today}.csv")

    env_acc = [float(r.get("env_accuracy", 0.0) or 0.0) for r in cycles_rows]
    energy_acc = [float(r.get("energy_accuracy", 0.0) or 0.0) for r in cycles_rows]

    detection_count_by_class = Counter(r.get("class", "unknown") for r in detections_rows)

    track_max_duration = defaultdict(float)
    speed_by_class = defaultdict(list)
    for r in detections_rows:
        cls = r.get("class", "unknown")
        tid = r.get("track_id", "-1")
        tracked_s = float(r.get("tracked_seconds", 0.0) or 0.0)
        speed_px_s = float(r.get("speed_px_s", 0.0) or 0.0)
        key = f"{cls}:{tid}"
        if tracked_s > track_max_duration[key]:
            track_max_duration[key] = tracked_s
        if speed_px_s > 0:
            speed_by_class[cls].append(speed_px_s)

    avg_track_duration = 0.0
    if track_max_duration:
        avg_track_duration = sum(track_max_duration.values()) / len(track_max_duration)

    avg_speed_per_class = {}
    for cls, speeds in speed_by_class.items():
        if speeds:
            avg_speed_per_class[cls] = sum(speeds) / len(speeds)

    track_conf_values = []
    for r in detections_rows:
        raw_track_conf = r.get("track_confidence", "")
        if str(raw_track_conf).strip() == "":
            # Backward compatibility: older detection logs may not include track_confidence.
            raw_track_conf = r.get("confidence", 0.0)
        track_conf_values.append(float(raw_track_conf or 0.0))
    avg_track_confidence = (sum(track_conf_values) / len(track_conf_values)) if track_conf_values else 0.0

    id_switch_proxy = 0.0
    person_rows = [r for r in detections_rows if r.get("class") == "person"]
    if person_rows:
        unique_person_tracks = len({r.get("track_id", "-1") for r in person_rows if r.get("track_id", "-1") != "-1"})
        id_switch_proxy = unique_person_tracks / max(1, len(person_rows))

    person_mode_cycles = [r for r in cycles_rows if str(r.get("person_only_mode", "0")) == "1"]
    no_detection_ratio_person_mode = 0.0
    if person_mode_cycles:
        no_det = sum(1 for r in person_mode_cycles if int(float(r.get("detection_count", 0) or 0)) == 0)
        no_detection_ratio_person_mode = no_det / len(person_mode_cycles)

    alert_count_by_code = Counter(r.get("code", "UNKNOWN") for r in alerts_rows)
    alert_risk_sum = defaultdict(float)
    for r in alerts_rows:
        code = r.get("code", "UNKNOWN")
        alert_risk_sum[code] += float(r.get("risk_score", 0.0) or 0.0)
    alert_precision_proxy = {
        code: (alert_risk_sum[code] / max(1, count)) for code, count in alert_count_by_code.items()
    }

    low_risk_high_severity = 0
    for r in alerts_rows:
        risk = float(r.get("risk_score", 0.0) or 0.0)
        severity = str(r.get("severity", "")).lower()
        if severity in {"high", "critical"} and risk < 0.55:
            low_risk_high_severity += 1
    false_alert_rate_proxy = 0.0
    if alerts_rows:
        false_alert_rate_proxy = low_risk_high_severity / len(alerts_rows)

    summary = {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cycles_count": len(cycles_rows),
        "detections_count": len(detections_rows),
        "alerts_count": len(alerts_rows),
        "env_accuracy_mean": (sum(env_acc) / len(env_acc)) if env_acc else 0.0,
        "energy_accuracy_mean": (sum(energy_acc) / len(energy_acc)) if energy_acc else 0.0,
        "avg_track_duration_s": avg_track_duration,
        "avg_track_confidence": avg_track_confidence,
        "id_switch_proxy": id_switch_proxy,
        "no_detection_ratio_person_mode": no_detection_ratio_person_mode,
        "false_alert_rate_proxy": false_alert_rate_proxy,
    }

    return EvaluationResult(
        summary=summary,
        per_class_speed=avg_speed_per_class,
        detection_counts=dict(detection_count_by_class),
        alert_counts=dict(alert_count_by_code),
        alert_precision_proxy=alert_precision_proxy,
    )


def export_evaluation(result: EvaluationResult, out_dir: str = "data/eval") -> tuple[str, str]:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = base / f"evaluation_summary_{stamp}.csv"
    txt_path = base / f"evaluation_report_{stamp}.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in result.summary.items():
            writer.writerow([k, v])

        writer.writerow([])
        writer.writerow(["detection_count_by_class", "count"])
        for k, v in sorted(result.detection_counts.items()):
            writer.writerow([k, v])

        writer.writerow([])
        writer.writerow(["avg_speed_px_s_by_class", "speed"])
        for k, v in sorted(result.per_class_speed.items()):
            writer.writerow([k, v])

        writer.writerow([])
        writer.writerow(["alert_count_by_code", "count"])
        for k, v in sorted(result.alert_counts.items()):
            writer.writerow([k, v])

        writer.writerow([])
        writer.writerow(["alert_precision_proxy_by_code", "mean_risk_score"])
        for k, v in sorted(result.alert_precision_proxy.items()):
            writer.writerow([k, v])

    lines = [
        "Smart Monitoring Evaluation Report",
        "================================",
        f"Generated: {result.summary.get('evaluated_at')}",
        "",
        "Core Metrics",
        "------------",
    ]
    for k, v in result.summary.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Detection Count by Class")
    lines.append("------------------------")
    if result.detection_counts:
        for k, v in sorted(result.detection_counts.items()):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No detections logged")

    lines.append("")
    lines.append("Average Speed by Class (px/s)")
    lines.append("------------------------------")
    if result.per_class_speed:
        for k, v in sorted(result.per_class_speed.items()):
            lines.append(f"- {k}: {v:.3f}")
    else:
        lines.append("- No speed samples logged")

    lines.append("")
    lines.append("Alert Count by Code")
    lines.append("-------------------")
    if result.alert_counts:
        for k, v in sorted(result.alert_counts.items()):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No alerts logged")

    lines.append("")
    lines.append("Alert Precision Proxy by Code (mean risk)")
    lines.append("------------------------------------------")
    if result.alert_precision_proxy:
        for k, v in sorted(result.alert_precision_proxy.items()):
            lines.append(f"- {k}: {v:.3f}")
    else:
        lines.append("- No alerts logged")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return str(csv_path), str(txt_path)
