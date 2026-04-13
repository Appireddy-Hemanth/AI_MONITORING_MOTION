from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _json_response(payload: dict, start_response, status: str = "200 OK"):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
        ]
        start_response(status, headers)
        return [body]


def _html_response(html: str, start_response, status: str = "200 OK"):
        body = html.encode("utf-8")
        headers = [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
        ]
        start_response(status, headers)
        return [body]


def _find_first_existing(paths: list[str]) -> Path | None:
        for p in paths:
                path = Path(p)
                if path.exists():
                        return path
        return None


def _latest_file(base: Path, pattern: str) -> Path | None:
        files = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None


def _read_csv_rows(path: Path) -> list[dict]:
        if not path.exists():
                return []
        with path.open("r", newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))


def _build_summary() -> dict:
        eval_dir = _find_first_existing(["project/data/eval", "data/eval"])
        logs_dir = _find_first_existing(["project/data/logs", "data/logs"])

        summary = {
                "status": "online",
                "evaluated_at": "",
                "env_accuracy_mean": 0.0,
                "energy_accuracy_mean": 0.0,
                "alerts_count": 0,
                "detections_count": 0,
                "cycles_count": 0,
                "false_alert_rate_proxy": 0.0,
                "id_switch_proxy": 0.0,
                "avg_track_confidence": 0.0,
                "latest_fps": 0.0,
                "latest_yolo_ms": 0.0,
                "camera_ok": False,
                "detector_ready": False,
        }

        if eval_dir:
                latest_summary = _latest_file(eval_dir, "evaluation_summary_*.csv")
                if latest_summary is not None:
                        with latest_summary.open("r", newline="", encoding="utf-8") as f:
                                reader = csv.reader(f)
                                for row in reader:
                                        if len(row) < 2:
                                                continue
                                        key = str(row[0]).strip()
                                        val = str(row[1]).strip()
                                        if not key or key == "metric":
                                                continue
                                        if key in {"alerts_count", "detections_count", "cycles_count"}:
                                                try:
                                                        summary[key] = int(float(val))
                                                except Exception:
                                                        pass
                                        elif key in {"env_accuracy_mean", "energy_accuracy_mean"}:
                                                try:
                                                        summary[key] = float(val)
                                                except Exception:
                                                        pass
                                        elif key in {"false_alert_rate_proxy", "id_switch_proxy", "avg_track_confidence"}:
                                                try:
                                                        summary[key] = float(val)
                                                except Exception:
                                                        pass
                                        elif key == "evaluated_at":
                                                summary[key] = val

        if logs_dir:
                today = datetime.now().strftime("%Y-%m-%d")
                alerts_file = logs_dir / f"alerts_{today}.csv"
                detections_file = logs_dir / f"detections_{today}.csv"
                cycles_file = logs_dir / f"cycles_{today}.csv"

                if alerts_file.exists():
                        summary["alerts_count"] = len(_read_csv_rows(alerts_file))
                if detections_file.exists():
                        summary["detections_count"] = len(_read_csv_rows(detections_file))
                if cycles_file.exists():
                        cycle_rows = _read_csv_rows(cycles_file)
                        summary["cycles_count"] = len(cycle_rows)
                        if cycle_rows:
                                latest = cycle_rows[-1]
                                try:
                                        summary["latest_fps"] = float(latest.get("fps", 0.0) or 0.0)
                                except Exception:
                                        pass
                                try:
                                        summary["latest_yolo_ms"] = float(latest.get("yolo_ms", 0.0) or 0.0)
                                except Exception:
                                        pass
                                summary["detector_ready"] = bool(int(float(latest.get("detection_enabled", 0) or 0)) == 1)

                # If there are detections today, treat camera as active.
                summary["camera_ok"] = summary["detections_count"] > 0

        return summary


def _latest_alerts(limit: int = 12) -> list[dict]:
        logs_dir = _find_first_existing(["project/data/logs", "data/logs"])
        if not logs_dir:
                return []
        today = datetime.now().strftime("%Y-%m-%d")
        rows = _read_csv_rows(logs_dir / f"alerts_{today}.csv")
        return rows[-limit:]


def _dashboard_html() -> str:
                                return """<!doctype html>
<html lang=\"en\">
<head>
                <meta charset=\"utf-8\" />
                <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
                <title>AI-Powered Smart Monitoring System</title>
                <style>
                                :root {
                                                --bg: #060f1c;
                                                --sidebar: #07182f;
                                                --panel: #0f2743;
                                                --line: #1e4b75;
                                                --text: #ebf2ff;
                                                --muted: #9eb7d9;
                                                --chip-red: #d95c62;
                                                --chip-amber: #da9c2e;
                                                --chip-blue: #7f98bd;
                                }
                                body {
                                                margin: 0;
                                                font-family: Segoe UI, Tahoma, sans-serif;
                                                color: var(--text);
                                                background: radial-gradient(circle at 25% -5%, #0d3158 0%, var(--bg) 45%);
                                }
                                .layout {
                                                display: grid;
                                                grid-template-columns: 260px 1fr;
                                                min-height: 100vh;
                                }
                                .sidebar {
                                                background: linear-gradient(180deg, #061124 0%, var(--sidebar) 100%);
                                                border-right: 1px solid #17395e;
                                                padding: 18px 16px;
                                }
                                .s-title {
                                                font-size: 0.85rem;
                                                color: var(--muted);
                                                margin-bottom: 8px;
                                }
                                .s-heading {
                                                font-size: 1.02rem;
                                                font-weight: 700;
                                                margin: 14px 0 8px;
                                }
                                .ctrl-group {
                                                margin-bottom: 10px;
                                }
                                .ctrl-group label {
                                                display: block;
                                                font-size: 0.86rem;
                                                margin-bottom: 4px;
                                                color: #d9e7ff;
                                }
                                .ctrl-group input,
                                .ctrl-group select {
                                                width: 100%;
                                                box-sizing: border-box;
                                                border-radius: 8px;
                                                border: 1px solid #20496f;
                                                background: #0b1e35;
                                                color: #e8f1ff;
                                                padding: 8px;
                                                font-size: 0.9rem;
                                }
                                .range-wrap {
                                                margin: 0 0 12px;
                                }
                                .btn {
                                                width: 100%;
                                                border: 1px solid #425471;
                                                border-radius: 10px;
                                                background: #2a2f42;
                                                color: #f0f5ff;
                                                padding: 10px 12px;
                                                font-size: 0.95rem;
                                                cursor: pointer;
                                                margin-bottom: 10px;
                                }
                                .btn-row {
                                                display: grid;
                                                grid-template-columns: 1fr 1fr;
                                                gap: 10px;
                                                margin-bottom: 10px;
                                }
                                .action-msg {
                                                margin-top: 6px;
                                                margin-bottom: 8px;
                                                color: #b9ccee;
                                                font-size: 0.84rem;
                                                border: 1px solid #1e4a74;
                                                border-radius: 8px;
                                                padding: 8px;
                                                background: rgba(8, 23, 41, 0.45);
                                }
                                .range-wrap label {
                                                display: block;
                                                font-size: 0.86rem;
                                                margin-bottom: 4px;
                                                color: #d9e7ff;
                                }
                                .range-val {
                                                font-size: 0.8rem;
                                                color: #ff7c7c;
                                                margin-bottom: 2px;
                                }
                                input[type=range] {
                                                width: 100%;
                                }
                                .nav {
                                                list-style: none;
                                                padding: 0;
                                                margin: 0 0 16px;
                                }
                                .nav li { margin-bottom: 4px; }
                                .nav-btn {
                                                width: 100%;
                                                text-align: left;
                                                border: 1px solid transparent;
                                                border-radius: 8px;
                                                color: #cfe0ff;
                                                font-size: 0.92rem;
                                                background: rgba(10, 24, 42, 0.35);
                                                padding: 7px 10px;
                                                cursor: pointer;
                                }
                                .nav-btn.active {
                                                border-color: var(--line);
                                                background: rgba(23, 58, 98, 0.55);
                                }
                                .main {
                                                padding: 28px 22px 40px;
                                }
                                .title {
                                                font-size: 2.1rem;
                                                font-weight: 700;
                                                margin: 0;
                                }
                                .sub {
                                                color: var(--muted);
                                                margin: 6px 0 16px;
                                }
                                .panel {
                                                background: linear-gradient(165deg, var(--panel), #173a62);
                                                border: 1px solid var(--line);
                                                border-radius: 14px;
                                                padding: 14px;
                                                margin-bottom: 12px;
                                }
                                .chips {
                                                display: flex;
                                                gap: 8px;
                                                flex-wrap: wrap;
                                }
                                .chip {
                                                display: inline-block;
                                                padding: 6px 10px;
                                                border-radius: 999px;
                                                font-size: 0.85rem;
                                                font-weight: 600;
                                }
                                .chip.red { background: var(--chip-red); }
                                .chip.amber { background: var(--chip-amber); }
                                .chip.blue { background: var(--chip-blue); }
                                .grid {
                                                display: grid;
                                                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                                                gap: 12px;
                                }
                                .card {
                                                background: rgba(7, 19, 35, 0.35);
                                                border: 1px solid var(--line);
                                                border-radius: 12px;
                                                padding: 14px;
                                }
                                .label { color: var(--muted); font-size: 0.9rem; }
                                .value { font-size: 1.4rem; font-weight: 700; margin-top: 6px; }
                                .section-title {
                                                margin-top: 4px;
                                                margin-bottom: 10px;
                                                font-size: 1.8rem;
                                                font-weight: 600;
                                }
                                table {
                                                width: 100%;
                                                border-collapse: collapse;
                                                background: rgba(7, 19, 35, 0.7);
                                                border: 1px solid var(--line);
                                                border-radius: 10px;
                                                overflow: hidden;
                                }
                                th, td {
                                                border-bottom: 1px solid var(--line);
                                                text-align: left;
                                                padding: 10px;
                                                font-size: 0.92rem;
                                }
                                th { color: var(--muted); }
                                .pill {
                                                display: inline-block;
                                                border: 1px solid var(--line);
                                                padding: 4px 8px;
                                                border-radius: 999px;
                                                font-size: 0.85rem;
                                                color: var(--muted);
                                                margin-top: 8px;
                                }
                                .page-section { display: none; }
                                .page-section.active { display: block; }
                                .live-feed {
                                                height: 250px;
                                                border-radius: 10px;
                                                border: 1px solid #173f62;
                                                background: linear-gradient(180deg, #0a1627 0%, #060b14 100%);
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                color: #95abc8;
                                                font-size: 0.95rem;
                                }
                                @media (max-width: 900px) {
                                                .layout { grid-template-columns: 1fr; }
                                                .sidebar { border-right: none; border-bottom: 1px solid #17395e; }
                                }
                </style>
</head>
<body>
                <div class=\"layout\">
                        <aside class=\"sidebar\">
                                <div class=\"s-title\">Navigate</div>
                                <ul class=\"nav\">
                                        <li><button class=\"nav-btn active\" data-page=\"review\">Review Mode</button></li>
                                        <li><button class=\"nav-btn\" data-page=\"dashboard\">Dashboard</button></li>
                                        <li><button class=\"nav-btn\" data-page=\"environment\">Environment Monitoring</button></li>
                                        <li><button class=\"nav-btn\" data-page=\"energy\">Energy Monitoring</button></li>
                                        <li><button class=\"nav-btn\" data-page=\"alerts\">Alerts Panel</button></li>
                                        <li><button class=\"nav-btn\" data-page=\"incident\">Incident Center</button></li>
                                        <li><button class=\"nav-btn\" data-page=\"data\">Data Lab</button></li>
                                        <li><button class=\"nav-btn\" data-page=\"health\">System Health</button></li>
                                </ul>
                                <div class=\"s-heading\">Production Policy Engine</div>
                                <div class=\"ctrl-group\">
                                        <label for=\"siteInput\">Site</label>
                                        <input id=\"siteInput\" type=\"text\" value=\"default-site\" />
                                </div>
                                <div class=\"ctrl-group\">
                                        <label for=\"shiftInput\">Shift</label>
                                        <select id=\"shiftInput\">
                                                <option selected>day</option>
                                                <option>night</option>
                                                <option>all</option>
                                        </select>
                                </div>
                                <div class=\"ctrl-group\">
                                        <label for=\"riskInput\">Risk profile</label>
                                        <select id=\"riskInput\">
                                                <option selected>normal</option>
                                                <option>high</option>
                                                <option>all</option>
                                        </select>
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"cooldownRange\">Global cooldown (seconds)</label>
                                        <div id=\"cooldownVal\" class=\"range-val\">12</div>
                                        <input id=\"cooldownRange\" type=\"range\" min=\"3\" max=\"180\" value=\"12\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"intrusionRange\">Intrusion clear (seconds)</label>
                                        <div id=\"intrusionVal\" class=\"range-val\">7</div>
                                        <input id=\"intrusionRange\" type=\"range\" min=\"2\" max=\"45\" value=\"7\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"gasRange\">Gas threshold (ppm)</label>
                                        <div id=\"gasVal\" class=\"range-val\">460</div>
                                        <input id=\"gasRange\" type=\"range\" min=\"200\" max=\"900\" value=\"460\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"tempRange\">Temperature threshold (C)</label>
                                        <div id=\"tempVal\" class=\"range-val\">43</div>
                                        <input id=\"tempRange\" type=\"range\" min=\"20\" max=\"80\" value=\"43\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"gasClearRange\">Gas clear ratio</label>
                                        <div id=\"gasClearVal\" class=\"range-val\">0.90</div>
                                        <input id=\"gasClearRange\" type=\"range\" min=\"0.70\" max=\"1.00\" step=\"0.01\" value=\"0.90\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"tempDeltaRange\">Temperature clear delta (C)</label>
                                        <div id=\"tempDeltaVal\" class=\"range-val\">2.00</div>
                                        <input id=\"tempDeltaRange\" type=\"range\" min=\"0.50\" max=\"10.00\" step=\"0.10\" value=\"2.00\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"quietRange\">Normal-status quiet time (seconds)</label>
                                        <div id=\"quietVal\" class=\"range-val\">20</div>
                                        <input id=\"quietRange\" type=\"range\" min=\"5\" max=\"180\" value=\"20\" />
                                </div>
                                <div class=\"ctrl-group\">
                                        <label for=\"policyCode\">Policy code</label>
                                        <select id=\"policyCode\">
                                                <option selected>INTRUSION</option>
                                                <option>FIRE</option>
                                                <option>GAS_HIGH</option>
                                                <option>TEMP_HIGH</option>
                                                <option>NORMAL</option>
                                        </select>
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"codeCooldown\">INTRUSION cooldown (seconds)</label>
                                        <div id=\"codeCooldownVal\" class=\"range-val\">10</div>
                                        <input id=\"codeCooldown\" type=\"range\" min=\"1\" max=\"300\" value=\"10\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"codeRisk\">INTRUSION min risk</label>
                                        <div id=\"codeRiskVal\" class=\"range-val\">0.62</div>
                                        <input id=\"codeRisk\" type=\"range\" min=\"0.00\" max=\"1.00\" step=\"0.01\" value=\"0.62\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"codeConfirm\">INTRUSION confirmation frames</label>
                                        <div id=\"codeConfirmVal\" class=\"range-val\">2</div>
                                        <input id=\"codeConfirm\" type=\"range\" min=\"1\" max=\"10\" value=\"2\" />
                                </div>
                                <div class=\"range-wrap\">
                                        <label for=\"codeEscalate\">INTRUSION escalate after (seconds)</label>
                                        <div id=\"codeEscalateVal\" class=\"range-val\">120</div>
                                        <input id=\"codeEscalate\" type=\"range\" min=\"0\" max=\"600\" value=\"120\" />
                                </div>
                                <div class=\"ctrl-group\">
                                        <label for=\"severityInput\">INTRUSION severity</label>
                                        <select id=\"severityInput\"><option selected>high</option><option>critical</option><option>low</option></select>
                                </div>
                                <div class=\"ctrl-group\">
                                        <label for=\"requesterInput\">Policy requester</label>
                                        <input id=\"requesterInput\" type=\"text\" value=\"operator\" />
                                </div>
                                <div class=\"ctrl-group\">
                                        <label for=\"approverInput\">Policy approver</label>
                                        <input id=\"approverInput\" type=\"text\" value=\"supervisor\" />
                                </div>
                                <div class=\"btn-row\">
                                        <button class=\"btn\" type=\"button\">Submit Policy Change</button>
                                        <button class=\"btn\" type=\"button\">Save Policy Directly</button>
                                </div>
                                <div class=\"ctrl-group\">
                                        <label for=\"versionsInput\">Policy versions</label>
                                        <select id=\"versionsInput\"><option selected>policy_20260413_210449.json</option></select>
                                </div>
                                <button class=\"btn\" type=\"button\">Rollback Selected Version</button>

                                <div class=\"s-heading\">Trust Layer</div>
                                <div class=\"range-wrap\"><label for=\"activeStart\">Active hour start</label><div id=\"activeStartVal\" class=\"range-val\">0</div><input id=\"activeStart\" type=\"range\" min=\"0\" max=\"23\" value=\"0\" /></div>
                                <div class=\"range-wrap\"><label for=\"activeEnd\">Active hour end</label><div id=\"activeEndVal\" class=\"range-val\">24</div><input id=\"activeEnd\" type=\"range\" min=\"1\" max=\"24\" value=\"24\" /></div>
                                <div class=\"range-wrap\"><label for=\"zoneX1\">Zone x1</label><div id=\"zoneX1Val\" class=\"range-val\">0.25</div><input id=\"zoneX1\" type=\"range\" min=\"0.00\" max=\"1.00\" step=\"0.01\" value=\"0.25\" /></div>
                                <div class=\"range-wrap\"><label for=\"zoneY1\">Zone y1</label><div id=\"zoneY1Val\" class=\"range-val\">0.20</div><input id=\"zoneY1\" type=\"range\" min=\"0.00\" max=\"1.00\" step=\"0.01\" value=\"0.20\" /></div>
                                <div class=\"range-wrap\"><label for=\"zoneX2\">Zone x2</label><div id=\"zoneX2Val\" class=\"range-val\">0.75</div><input id=\"zoneX2\" type=\"range\" min=\"0.00\" max=\"1.00\" step=\"0.01\" value=\"0.75\" /></div>
                                <div class=\"range-wrap\"><label for=\"zoneY2\">Zone y2</label><div id=\"zoneY2Val\" class=\"range-val\">0.90</div><input id=\"zoneY2\" type=\"range\" min=\"0.00\" max=\"1.00\" step=\"0.01\" value=\"0.90\" /></div>
                                <div class=\"range-wrap\"><label for=\"occupancy\">Occupancy limit</label><div id=\"occupancyVal\" class=\"range-val\">3</div><input id=\"occupancy\" type=\"range\" min=\"1\" max=\"20\" value=\"3\" /></div>
                                <div class=\"range-wrap\"><label for=\"sla\">Incident SLA (seconds)</label><div id=\"slaVal\" class=\"range-val\">120</div><input id=\"sla\" type=\"range\" min=\"30\" max=\"1800\" value=\"120\" /></div>
                                <div class=\"ctrl-group\"><label for=\"assignee\">Default assignee</label><input id=\"assignee\" type=\"text\" value=\"security-team\" /></div>
                                <div class=\"ctrl-group\">
                                        <label>Require human verification for</label>
                                        <div><input type=\"checkbox\" checked /> FIRE</div>
                                        <div><input type=\"checkbox\" checked /> GAS_HIGH</div>
                                </div>

                                <div class=\"s-heading\">Email Alerts (Real-Time Intrusion)</div>
                                <div class=\"ctrl-group\"><label><input type=\"checkbox\" /> Enable intrusion email</label></div>
                                <div class=\"ctrl-group\"><label for=\"smtpHost\">SMTP Host</label><input id=\"smtpHost\" type=\"text\" value=\"smtp.gmail.com\" /></div>
                                <div class=\"ctrl-group\"><label for=\"smtpPort\">SMTP Port</label><input id=\"smtpPort\" type=\"number\" value=\"587\" /></div>
                                <div class=\"ctrl-group\"><label for=\"senderEmail\">Sender Email</label><input id=\"senderEmail\" type=\"text\" value=\"\" /></div>
                                <div class=\"ctrl-group\"><label for=\"senderPass\">Sender App Password</label><input id=\"senderPass\" type=\"password\" value=\"\" /></div>
                                <div class=\"ctrl-group\"><label for=\"receiverEmail\">Receiver Email</label><input id=\"receiverEmail\" type=\"text\" value=\"\" /></div>
                                <div class=\"range-wrap\"><label for=\"emailCooldown\">Email cooldown (seconds)</label><div id=\"emailCooldownVal\" class=\"range-val\">60</div><input id=\"emailCooldown\" type=\"range\" min=\"10\" max=\"300\" value=\"60\" /></div>
                                <div class=\"ctrl-group\"><label><input type=\"checkbox\" checked /> Attach captured intrusion image</label></div>
                                <button class=\"btn\" type=\"button\">Send Test Email</button>

                                <div class=\"s-title\">PINN is active for sensor/energy prediction.</div>
                                <div class=\"ctrl-group\"><label><input type=\"checkbox\" checked /> Enable object detection</label></div>
                                <div class=\"ctrl-group\"><label><input type=\"checkbox\" /> Person-only detection mode</label></div>
                                <div class=\"ctrl-group\"><label for=\"detProfile\">Detection profile</label><select id=\"detProfile\"><option selected>Comprehensive</option><option>Balanced</option><option>Lightweight</option></select></div>
                                <div class=\"ctrl-group\"><label><input type=\"checkbox\" checked /> Detect all objects (all YOLO classes)</label></div>
                                <div class=\"range-wrap\"><label for=\"detConf\">Detection confidence threshold</label><div id=\"detConfVal\" class=\"range-val\">0.15</div><input id=\"detConf\" type=\"range\" min=\"0.10\" max=\"0.80\" step=\"0.01\" value=\"0.15\" /></div>
                                <div class=\"s-title\">Class-Specific Thresholds</div>
                                <div class=\"range-wrap\"><label for=\"personThr\">Person threshold</label><div id=\"personThrVal\" class=\"range-val\">0.25</div><input id=\"personThr\" type=\"range\" min=\"0.10\" max=\"0.90\" step=\"0.01\" value=\"0.25\" /></div>
                                <div class=\"range-wrap\"><label for=\"objectThr\">Object threshold</label><div id=\"objectThrVal\" class=\"range-val\">0.20</div><input id=\"objectThr\" type=\"range\" min=\"0.10\" max=\"0.90\" step=\"0.01\" value=\"0.20\" /></div>
                                <div class=\"range-wrap\"><label for=\"paperThr\">Paper threshold</label><div id=\"paperThrVal\" class=\"range-val\">0.15</div><input id=\"paperThr\" type=\"range\" min=\"0.05\" max=\"0.90\" step=\"0.01\" value=\"0.15\" /></div>
                                <div class=\"range-wrap\"><label for=\"fireThr\">Fire threshold</label><div id=\"fireThrVal\" class=\"range-val\">0.45</div><input id=\"fireThr\" type=\"range\" min=\"0.10\" max=\"0.95\" step=\"0.01\" value=\"0.45\" /></div>
                                <div class=\"range-wrap\"><label for=\"stride\">Detection stride (higher = lighter CPU)</label><div id=\"strideVal\" class=\"range-val\">1</div><input id=\"stride\" type=\"range\" min=\"1\" max=\"6\" value=\"1\" /></div>
                                <div class=\"s-title\">Speed Calibration (m/s)</div>
                                <div class=\"ctrl-group\"><label for=\"knownDist\">Known distance (meters)</label><input id=\"knownDist\" type=\"number\" step=\"0.1\" value=\"2.00\" /></div>
                                <div class=\"ctrl-group\"><label for=\"pixelDist\">Measured pixel distance</label><input id=\"pixelDist\" type=\"number\" step=\"1\" value=\"220.00\" /></div>
                                <div class=\"s-title\">Optional two-point calibration from image coordinates</div>
                                <div class=\"ctrl-group\"><label>Point1 X</label><input type=\"number\" value=\"100\" /></div>
                                <div class=\"ctrl-group\"><label>Point2 X</label><input type=\"number\" value=\"400\" /></div>
                                <div class=\"ctrl-group\"><label>Point1 Y</label><input type=\"number\" value=\"100\" /></div>
                                <div class=\"ctrl-group\"><label>Point2 Y</label><input type=\"number\" value=\"100\" /></div>
                                <div class=\"s-title\">Current scale: 0.009091 m/px</div>
                                <div class=\"ctrl-group\"><label for=\"fireModel\">Optional Fire Model Path (.pt)</label><input id=\"fireModel\" type=\"text\" value=\"\" /></div>
                                <div class=\"ctrl-group\"><label for=\"videoSource\">Video Source</label><select id=\"videoSource\"><option selected>Webcam</option><option>Upload Video</option></select></div>
                                <div class=\"btn-row\">
                                        <button class=\"btn\" type=\"button\">Start</button>
                                        <button class=\"btn\" type=\"button\">Stop</button>
                                </div>
                                <button class=\"btn\" type=\"button\">Reset History</button>
                                <div class=\"s-title\">Dataset Controls</div>
                                <div class=\"ctrl-group\"><label for=\"datasets\">Select datasets for training</label><select id=\"datasets\"><option selected>data\\raw_sensor_energy_1000.csv</option></select></div>
                                <div class=\"ctrl-group\"><label for=\"datasetUpload\">Upload dataset CSV</label><input id=\"datasetUpload\" type=\"file\" /></div>
                                <button class=\"btn\" type=\"button\">Train PINN with Selected Datasets</button>
                                <div class=\"ctrl-group\"><label><input type=\"checkbox\" checked /> Enable sound alert</label></div>
                                <button class=\"btn\" type=\"button\">Train with 1000 Raw Samples</button>
                                <button class=\"btn\" type=\"button\">Download Monitoring Report</button>
                                <button class=\"btn\" type=\"button\">Generate Evaluation Report</button>
                                <div class=\"s-title\">Scenario Recorder</div>
                                <div class=\"ctrl-group\"><label for=\"scenarioName\">Scenario name</label><input id=\"scenarioName\" type=\"text\" value=\"\" /></div>
                                <div class=\"ctrl-group\"><label for=\"markText\">Mark event text</label><input id=\"markText\" type=\"text\" value=\"\" /></div>
                                <div class=\"btn-row\">
                                        <button class=\"btn\" type=\"button\">Start Scenario</button>
                                        <button class=\"btn\" type=\"button\">Stop Scenario</button>
                                </div>
                                <button class=\"btn\" type=\"button\">Mark Event</button>
                                <button class=\"btn\" type=\"button\">Export Review Bundle</button>
                                <div class=\"s-title\" id=\"systemStatus\">System Status: STOPPED</div>
                                <div class=\"action-msg\" id=\"actionMsg\">Ready.</div>
                                <div class=\"s-title\">Deployment</div>
                                <div style=\"color:#cfe0ff;font-size:0.9rem;\">Vercel dashboard mirror of your monitoring data.</div>
                        </aside>

                        <main class=\"main\">
                                <h1 class=\"title\">AI-Powered Smart Monitoring System</h1>
                                <div class=\"sub\">Computer Vision + PINN forecasting + intelligent risk alerts for real-time smart monitoring.</div>

                                <section class=\"page-section active\" id=\"page-review\">
                                        <div class=\"panel\">
                                                <div style=\"font-weight:700;margin-bottom:8px;\">Status</div>
                                                <div class=\"chips\">
                                                        <span id=\"chipCamera\" class=\"chip red\">camera fallback</span>
                                                        <span id=\"chipDetector\" class=\"chip amber\">detector not ready</span>
                                                        <span id=\"chipDetection\" class=\"chip blue\">No detections yet</span>
                                                </div>
                                        </div>

                                        <div class=\"panel\">
                                                <div class=\"section-title\">Review Mode - Key Live Metrics</div>
                                                <div class=\"grid\">
                                                        <div class=\"card\"><div class=\"label\">Detections</div><div id=\"detections\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Env Accuracy</div><div id=\"env\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Energy Accuracy</div><div id=\"energy\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">FPS</div><div id=\"fps\" class=\"value\">-</div></div>
                                                </div>
                                        </div>

                                        <div class=\"panel\">
                                                <div class=\"section-title\">Latest Evaluation Summary</div>
                                                <div class=\"grid\">
                                                        <div class=\"card\"><div class=\"label\">False Alert Proxy</div><div id=\"falseAlert\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">ID Switch Proxy</div><div id=\"idSwitch\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Track Confidence</div><div id=\"trackConf\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Alerts (today)</div><div id=\"alerts\" class=\"value\">-</div></div>
                                                </div>
                                                <div class=\"pill\" id=\"stamp\">Evaluated at: loading...</div>
                                        </div>

                                        <div class=\"panel\">
                                                <div class=\"section-title\">Live Feed</div>
                                                <div class=\"live-feed\">Live camera feed is available in local Streamlit runtime.</div>
                                        </div>
                                </section>

                                <section class=\"page-section\" id=\"page-dashboard\">
                                        <div class=\"panel\">
                                                <div class=\"section-title\">Dashboard</div>
                                                <div class=\"grid\">
                                                        <div class=\"card\"><div class=\"label\">Monitoring Cycles</div><div id=\"cyclesDash\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Alerts (today)</div><div id=\"alertsDash\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Detections (today)</div><div id=\"detectionsDash\" class=\"value\">-</div></div>
                                                </div>
                                        </div>
                                </section>

                                <section class=\"page-section\" id=\"page-environment\">
                                        <div class=\"panel\">
                                                <div class=\"section-title\">Environment Monitoring</div>
                                                <div class=\"grid\">
                                                        <div class=\"card\"><div class=\"label\">Env Accuracy</div><div id=\"envEnv\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">False Alert Proxy</div><div id=\"falseEnv\" class=\"value\">-</div></div>
                                                </div>
                                        </div>
                                </section>

                                <section class=\"page-section\" id=\"page-energy\">
                                        <div class=\"panel\">
                                                <div class=\"section-title\">Energy Monitoring</div>
                                                <div class=\"grid\">
                                                        <div class=\"card\"><div class=\"label\">Energy Accuracy</div><div id=\"energyEnergy\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Latest YOLO Latency (ms)</div><div id=\"yoloEnergy\" class=\"value\">-</div></div>
                                                </div>
                                        </div>
                                </section>

                                <section class=\"page-section\" id=\"page-alerts\">
                                        <div class=\"panel\">
                                                <div class=\"section-title\">Alerts Panel</div>
                                                <table>
                                                        <thead>
                                                                <tr><th>Time</th><th>Code</th><th>Severity</th><th>Risk</th><th>Message</th></tr>
                                                        </thead>
                                                        <tbody id=\"alertsBody\"><tr><td colspan=\"5\">Loading...</td></tr></tbody>
                                                </table>
                                        </div>
                                </section>

                                <section class=\"page-section\" id=\"page-incident\">
                                        <div class=\"panel\">
                                                <div class=\"section-title\">Incident Center</div>
                                                <div class=\"card\">Incident lifecycle actions remain in local Streamlit runtime. This hosted view shows latest alert telemetry.</div>
                                        </div>
                                </section>

                                <section class=\"page-section\" id=\"page-data\">
                                        <div class=\"panel\">
                                                <div class=\"section-title\">Data Lab</div>
                                                <div class=\"grid\">
                                                        <div class=\"card\"><div class=\"label\">Cycles Logged</div><div id=\"cyclesData\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Evaluated At</div><div id=\"stampData\" class=\"value\" style=\"font-size:1rem;\">-</div></div>
                                                </div>
                                        </div>
                                </section>

                                <section class=\"page-section\" id=\"page-health\">
                                        <div class=\"panel\">
                                                <div class=\"section-title\">System Health</div>
                                                <div class=\"grid\">
                                                        <div class=\"card\"><div class=\"label\">FPS</div><div id=\"fpsHealth\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">YOLO Latency (ms)</div><div id=\"yoloHealth\" class=\"value\">-</div></div>
                                                        <div class=\"card\"><div class=\"label\">Detector</div><div id=\"detectorHealth\" class=\"value\">-</div></div>
                                                </div>
                                        </div>
                                </section>
                        </main>
                </div>

                <script>
                                function setPage(page) {
                                                document.querySelectorAll('.page-section').forEach(el => el.classList.remove('active'));
                                                document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
                                                const section = document.getElementById('page-' + page);
                                                if (section) {
                                                                section.classList.add('active');
                                                }
                                                const btn = document.querySelector('.nav-btn[data-page="' + page + '"]');
                                                if (btn) {
                                                                btn.classList.add('active');
                                                }
                                }

                                async function loadSummary() {
                                                const res = await fetch('/api/summary', { cache: 'no-store' });
                                                const s = await res.json();
                                                document.getElementById('env').textContent = Number(s.env_accuracy_mean || 0).toFixed(2) + '%';
                                                document.getElementById('energy').textContent = Number(s.energy_accuracy_mean || 0).toFixed(2) + '%';
                                                document.getElementById('alerts').textContent = String(s.alerts_count || 0);
                                                document.getElementById('detections').textContent = String(s.detections_count || 0);
                                                document.getElementById('fps').textContent = Number(s.latest_fps || 0).toFixed(2);
                                                document.getElementById('falseAlert').textContent = Number(s.false_alert_rate_proxy || 0).toFixed(3);
                                                document.getElementById('idSwitch').textContent = Number(s.id_switch_proxy || 0).toFixed(3);
                                                document.getElementById('trackConf').textContent = Number(s.avg_track_confidence || 0).toFixed(3);
                                                document.getElementById('stamp').textContent = 'Evaluated at: ' + (s.evaluated_at || 'not available');

                                                document.getElementById('cyclesDash').textContent = String(s.cycles_count || 0);
                                                document.getElementById('alertsDash').textContent = String(s.alerts_count || 0);
                                                document.getElementById('detectionsDash').textContent = String(s.detections_count || 0);
                                                document.getElementById('envEnv').textContent = Number(s.env_accuracy_mean || 0).toFixed(2) + '%';
                                                document.getElementById('falseEnv').textContent = Number(s.false_alert_rate_proxy || 0).toFixed(3);
                                                document.getElementById('energyEnergy').textContent = Number(s.energy_accuracy_mean || 0).toFixed(2) + '%';
                                                document.getElementById('yoloEnergy').textContent = Number(s.latest_yolo_ms || 0).toFixed(2);
                                                document.getElementById('cyclesData').textContent = String(s.cycles_count || 0);
                                                document.getElementById('stampData').textContent = String(s.evaluated_at || 'not available');
                                                document.getElementById('fpsHealth').textContent = Number(s.latest_fps || 0).toFixed(2);
                                                document.getElementById('yoloHealth').textContent = Number(s.latest_yolo_ms || 0).toFixed(2);
                                                document.getElementById('detectorHealth').textContent = s.detector_ready ? 'ready' : 'not ready';

                                                const chipCamera = document.getElementById('chipCamera');
                                                const chipDetector = document.getElementById('chipDetector');
                                                const chipDetection = document.getElementById('chipDetection');
                                                chipCamera.textContent = s.camera_ok ? 'camera ok' : 'camera fallback';
                                                chipDetector.textContent = s.detector_ready ? 'detector ready' : 'detector not ready';
                                                chipDetection.textContent = Number(s.detections_count || 0) > 0 ? 'detections active' : 'No detections yet';
                                }

                                async function loadAlerts() {
                                                const res = await fetch('/api/alerts', { cache: 'no-store' });
                                                const rows = await res.json();
                                                const body = document.getElementById('alertsBody');
                                                if (!Array.isArray(rows) || rows.length === 0) {
                                                                body.innerHTML = '<tr><td colspan="5">No alerts found for today.</td></tr>';
                                                                return;
                                                }
                                                body.innerHTML = rows.reverse().map(r => {
                                                                return '<tr>' +
                                                                                '<td>' + (r.timestamp || '') + '</td>' +
                                                                                '<td>' + (r.code || '') + '</td>' +
                                                                                '<td>' + (r.severity || '') + '</td>' +
                                                                                '<td>' + (r.risk_score || '') + '</td>' +
                                                                                '<td>' + (r.message || '') + '</td>' +
                                                                                '</tr>';
                                                }).join('');
                                }

                                async function boot() {
                                                function showAction(msg) {
                                                                const n = document.getElementById('actionMsg');
                                                                if (n) {
                                                                                n.textContent = msg;
                                                                }
                                                }

                                                function setSystemRunning(isRunning) {
                                                                const s = document.getElementById('systemStatus');
                                                                if (s) {
                                                                                s.textContent = isRunning ? 'System Status: RUNNING' : 'System Status: STOPPED';
                                                                }
                                                }

                                                function downloadTextFile(filename, content, mimeType) {
                                                                const blob = new Blob([content], { type: mimeType || 'text/plain' });
                                                                const url = URL.createObjectURL(blob);
                                                                const a = document.createElement('a');
                                                                a.href = url;
                                                                a.download = filename;
                                                                document.body.appendChild(a);
                                                                a.click();
                                                                a.remove();
                                                                URL.revokeObjectURL(url);
                                                }

                                                const rangePairs = [
                                                                ['cooldownRange', 'cooldownVal'],
                                                                ['intrusionRange', 'intrusionVal'],
                                                                ['gasRange', 'gasVal'],
                                                                ['tempRange', 'tempVal'],
                                                                ['gasClearRange', 'gasClearVal'],
                                                                ['tempDeltaRange', 'tempDeltaVal'],
                                                                ['quietRange', 'quietVal'],
                                                                ['codeCooldown', 'codeCooldownVal'],
                                                                ['codeRisk', 'codeRiskVal'],
                                                                ['codeConfirm', 'codeConfirmVal'],
                                                                ['codeEscalate', 'codeEscalateVal'],
                                                                ['activeStart', 'activeStartVal'],
                                                                ['activeEnd', 'activeEndVal'],
                                                                ['zoneX1', 'zoneX1Val'],
                                                                ['zoneY1', 'zoneY1Val'],
                                                                ['zoneX2', 'zoneX2Val'],
                                                                ['zoneY2', 'zoneY2Val'],
                                                                ['occupancy', 'occupancyVal'],
                                                                ['sla', 'slaVal'],
                                                                ['emailCooldown', 'emailCooldownVal'],
                                                                ['detConf', 'detConfVal'],
                                                                ['personThr', 'personThrVal'],
                                                                ['objectThr', 'objectThrVal'],
                                                                ['paperThr', 'paperThrVal'],
                                                                ['fireThr', 'fireThrVal'],
                                                                ['stride', 'strideVal'],
                                                ];
                                                rangePairs.forEach(([rid, vid]) => {
                                                                const r = document.getElementById(rid);
                                                                const v = document.getElementById(vid);
                                                                if (r && v) {
                                                                                const sync = () => { v.textContent = r.value; };
                                                                                r.addEventListener('input', sync);
                                                                                sync();
                                                                }
                                                });
                                                document.querySelectorAll('.nav-btn').forEach(btn => {
                                                                btn.addEventListener('click', () => setPage(btn.dataset.page));
                                                });

                                                document.querySelectorAll('button.btn').forEach(btn => {
                                                                btn.addEventListener('click', async () => {
                                                                                const label = (btn.textContent || '').trim();
                                                                                if (label === 'Start') {
                                                                                                setSystemRunning(true);
                                                                                                showAction('Monitoring started.');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Stop') {
                                                                                                setSystemRunning(false);
                                                                                                showAction('Monitoring stopped.');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Reset History') {
                                                                                                const body = document.getElementById('alertsBody');
                                                                                                if (body) {
                                                                                                                body.innerHTML = '<tr><td colspan="5">History reset in dashboard view.</td></tr>';
                                                                                                }
                                                                                                showAction('Dashboard history reset.');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Send Test Email') {
                                                                                                const sender = (document.getElementById('senderEmail') || {}).value || '';
                                                                                                const receiver = (document.getElementById('receiverEmail') || {}).value || '';
                                                                                                if (!sender || !receiver) {
                                                                                                                showAction('Test email failed: sender/receiver email required.');
                                                                                                } else {
                                                                                                                showAction('Test email request simulated successfully.');
                                                                                                }
                                                                                                return;
                                                                                }
                                                                                if (label === 'Download Monitoring Report') {
                                                                                                const summaryRes = await fetch('/api/summary', { cache: 'no-store' });
                                                                                                const summary = await summaryRes.json();
                                                                                                const alertsRes = await fetch('/api/alerts', { cache: 'no-store' });
                                                                                                const alerts = await alertsRes.json();
                                                                                                let csv = 'metric,value\\n';
                                                                                                Object.keys(summary).forEach(k => {
                                                                                                                csv += `${k},${String(summary[k]).replace(/,/g, ' ')}\\n`;
                                                                                                });
                                                                                                csv += '\\n';
                                                                                                csv += 'timestamp,code,severity,risk_score,message\\n';
                                                                                                (alerts || []).forEach(a => {
                                                                                                                csv += `${a.timestamp || ''},${a.code || ''},${a.severity || ''},${a.risk_score || ''},${String(a.message || '').replace(/,/g, ' ')}\\n`;
                                                                                                });
                                                                                                downloadTextFile('monitoring_report.csv', csv, 'text/csv');
                                                                                                showAction('Monitoring report downloaded.');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Export Review Bundle') {
                                                                                                const payload = {
                                                                                                                exported_at: new Date().toISOString(),
                                                                                                                site: (document.getElementById('siteInput') || {}).value || 'default-site',
                                                                                                                shift: (document.getElementById('shiftInput') || {}).value || 'day',
                                                                                                                risk_profile: (document.getElementById('riskInput') || {}).value || 'normal',
                                                                                                };
                                                                                                downloadTextFile('review_bundle.json', JSON.stringify(payload, null, 2), 'application/json');
                                                                                                showAction('Review bundle exported.');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Generate Evaluation Report') {
                                                                                                await Promise.all([loadSummary(), loadAlerts()]);
                                                                                                showAction('Evaluation summary refreshed.');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Train PINN with Selected Datasets' || label === 'Train with 1000 Raw Samples') {
                                                                                                showAction(label + ' triggered (dashboard simulation).');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Start Scenario' || label === 'Stop Scenario' || label === 'Mark Event') {
                                                                                                showAction(label + ' recorded.');
                                                                                                return;
                                                                                }
                                                                                if (label === 'Submit Policy Change' || label === 'Save Policy Directly' || label === 'Rollback Selected Version') {
                                                                                                showAction(label + ' action captured.');
                                                                                                return;
                                                                                }
                                                                                showAction(label + ' clicked.');
                                                                });
                                                });
                                                setPage('review');
                                                await Promise.all([loadSummary(), loadAlerts()]);
                                                showAction('Dashboard loaded successfully.');
                                }
                                boot();
                </script>
</body>
</html>
"""


def app(environ, start_response):
        """WSGI entrypoint for Vercel with lightweight dashboard and APIs."""
        path = environ.get("PATH_INFO", "/")

        if path == "/health":
                return _json_response(
                        {
                                "status": "ok",
                                "service": "ai-smart-monitoring-vercel",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        start_response,
                )

        if path == "/api/summary":
                return _json_response(_build_summary(), start_response)

        if path == "/api/alerts":
                return _json_response(_latest_alerts(limit=12), start_response)

        if path == "/":
                return _html_response(_dashboard_html(), start_response)

        return _json_response({"error": "Not found", "path": path}, start_response, status="404 Not Found")
