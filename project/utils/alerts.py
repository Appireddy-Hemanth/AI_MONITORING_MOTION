from datetime import datetime


class AlertManager:
    """Generates and stores system alerts with severity labels and colors."""

    def __init__(
        self,
        gas_threshold: float = 450.0,
        temp_threshold: float = 42.0,
        cooldown_seconds: float = 10.0,
    ) -> None:
        self.gas_threshold = gas_threshold
        self.temp_threshold = temp_threshold
        self.alert_history = []
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_times = {}
        self.last_anomaly_time = None
        self.normal_after_quiet_seconds = 20.0
        self.intrusion_active = False
        self.last_person_seen_time = None
        self.intrusion_clear_seconds = 8.0
        self.gas_event_active = False
        self.temp_event_active = False
        self.gas_clear_ratio = 0.90
        self.temp_clear_delta = 2.0
        self.code_policy = {}

    @staticmethod
    def _secs_since(ts: datetime | None) -> float:
        if ts is None:
            return 10**9
        return (datetime.now() - ts).total_seconds()

    def _can_emit(self, alert_code: str) -> bool:
        code_cfg = dict(self.code_policy.get(alert_code, {}))
        cooldown = float(code_cfg.get("cooldown_seconds", self.cooldown_seconds))
        return self._secs_since(self.last_alert_times.get(alert_code)) >= cooldown

    def _mark_emitted(self, alert_code: str) -> None:
        self.last_alert_times[alert_code] = datetime.now()

    @staticmethod
    def _new_alert(
        source: str,
        message: str,
        severity: str,
        color: str,
        code: str,
        risk_score: float,
        escalate_after_seconds: float = 0.0,
    ) -> dict:
        return {
            "timestamp": datetime.now(),
            "source": source,
            "message": message,
            "severity": severity,
            "color": color,
            "code": code,
            "risk_score": round(risk_score, 3),
            "escalate_after_seconds": float(escalate_after_seconds),
        }

    def _code_cfg(self, code: str) -> dict:
        cfg = dict(self.code_policy.get(code, {}))
        return {
            "min_risk": float(cfg.get("min_risk", 0.0)),
            "severity": str(cfg.get("severity", "low")),
            "escalate_after_seconds": float(cfg.get("escalate_after_seconds", 0.0)),
        }

    @staticmethod
    def _fused_risk(sensor_data: dict, detections: list, env_accuracy: float, energy_accuracy: float) -> float:
        gas_ratio = min(1.0, sensor_data["gas"] / 700.0)
        temp_ratio = min(1.0, sensor_data["temperature"] / 55.0)
        person_conf = max([d.get("confidence", 0.0) for d in detections if d.get("class") == "person"] + [0.0])
        fire_conf = max([d.get("confidence", 0.0) for d in detections if d.get("class") == "fire"] + [0.0])

        # Lower PGNN confidence (accuracy) increases uncertainty contribution.
        pgnn_uncertainty = 1.0 - max(0.0, min(1.0, (env_accuracy + energy_accuracy) / 200.0))

        risk = (
            0.30 * fire_conf
            + 0.20 * person_conf
            + 0.22 * gas_ratio
            + 0.18 * temp_ratio
            + 0.10 * pgnn_uncertainty
        )
        return max(0.0, min(1.0, risk))

    def evaluate(self, detections: list, sensor_data: dict, env_accuracy: float = 100.0, energy_accuracy: float = 100.0) -> list:
        alerts = []

        classes = [d["class"] for d in detections]
        fire_confidences = [d.get("confidence", 0.0) for d in detections if d.get("class") == "fire"]
        fused_risk = self._fused_risk(sensor_data, detections, env_accuracy, energy_accuracy)
        anomaly_present = False

        person_present = "person" in classes
        intrusion_cfg = self._code_cfg("INTRUSION")
        if person_present:
            self.last_person_seen_time = datetime.now()

        if self.intrusion_active and self._secs_since(self.last_person_seen_time) >= self.intrusion_clear_seconds:
            self.intrusion_active = False

        if person_present and (not self.intrusion_active) and self._can_emit("INTRUSION"):
            anomaly_present = True
            alerts.append(
                self._new_alert(
                    source="vision",
                    message="Intrusion warning: person detected in monitored zone.",
                    severity=intrusion_cfg["severity"],
                    color="#ff4b4b",
                    code="INTRUSION",
                    risk_score=max(fused_risk, intrusion_cfg["min_risk"]),
                    escalate_after_seconds=intrusion_cfg["escalate_after_seconds"],
                )
            )
            self.intrusion_active = True
            self._mark_emitted("INTRUSION")
        elif person_present:
            anomaly_present = True

        fire_cfg = self._code_cfg("FIRE")
        if fire_confidences and max(fire_confidences) >= 0.60 and self._can_emit("FIRE"):
            anomaly_present = True
            alerts.append(
                self._new_alert(
                    source="vision",
                    message="Critical warning: possible fire detected.",
                    severity=fire_cfg["severity"],
                    color="#ff1e1e",
                    code="FIRE",
                    risk_score=max(fused_risk, fire_cfg["min_risk"]),
                    escalate_after_seconds=fire_cfg["escalate_after_seconds"],
                )
            )
            self._mark_emitted("FIRE")
        elif fire_confidences and max(fire_confidences) >= 0.60:
            anomaly_present = True

        gas_cfg = self._code_cfg("GAS_HIGH")
        gas_value = float(sensor_data["gas"])
        gas_high = gas_value >= self.gas_threshold
        gas_clear_threshold = self.gas_threshold * self.gas_clear_ratio
        if self.gas_event_active and gas_value < gas_clear_threshold:
            self.gas_event_active = False

        if gas_high and (not self.gas_event_active) and self._can_emit("GAS_HIGH"):
            anomaly_present = True
            alerts.append(
                self._new_alert(
                    source="sensor",
                    message=f"Gas level high: {gas_value:.2f} ppm exceeds threshold.",
                    severity=gas_cfg["severity"],
                    color="#ff6b00",
                    code="GAS_HIGH",
                    risk_score=max(fused_risk, gas_cfg["min_risk"]),
                    escalate_after_seconds=gas_cfg["escalate_after_seconds"],
                )
            )
            self.gas_event_active = True
            self._mark_emitted("GAS_HIGH")
        elif gas_high or self.gas_event_active:
            anomaly_present = True

        temp_cfg = self._code_cfg("TEMP_HIGH")
        temp_value = float(sensor_data["temperature"])
        temp_high = temp_value >= self.temp_threshold
        temp_clear_threshold = self.temp_threshold - self.temp_clear_delta
        if self.temp_event_active and temp_value < temp_clear_threshold:
            self.temp_event_active = False

        if temp_high and (not self.temp_event_active) and self._can_emit("TEMP_HIGH"):
            anomaly_present = True
            alerts.append(
                self._new_alert(
                    source="sensor",
                    message=f"Temperature abnormal: {temp_value:.2f} C.",
                    severity=temp_cfg["severity"],
                    color="#ffa500",
                    code="TEMP_HIGH",
                    risk_score=max(fused_risk, temp_cfg["min_risk"]),
                    escalate_after_seconds=temp_cfg["escalate_after_seconds"],
                )
            )
            self.temp_event_active = True
            self._mark_emitted("TEMP_HIGH")
        elif temp_high or self.temp_event_active:
            anomaly_present = True

        if anomaly_present:
            self.last_anomaly_time = datetime.now()

        quiet_long_enough = self._secs_since(self.last_anomaly_time) >= self.normal_after_quiet_seconds
        safe_for_normal = (not anomaly_present) and fused_risk < 0.45 and quiet_long_enough

        normal_cfg = self._code_cfg("NORMAL")
        if not alerts and safe_for_normal and self._can_emit("NORMAL"):
            alerts.append(
                self._new_alert(
                    source="system",
                    message="System normal: no critical anomalies detected.",
                    severity=normal_cfg["severity"],
                    color="#2ecc71",
                    code="NORMAL",
                    risk_score=max(fused_risk, normal_cfg["min_risk"]),
                    escalate_after_seconds=normal_cfg["escalate_after_seconds"],
                )
            )
            self._mark_emitted("NORMAL")

        self.alert_history.extend(alerts)
        self.alert_history = self.alert_history[-1500:]
        return alerts

    def latest(self, limit: int = 100) -> list:
        return self.alert_history[-limit:]
