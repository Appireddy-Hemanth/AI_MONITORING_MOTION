"""Email notification utilities for real-time critical alerts."""

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    sender_email: str
    sender_password: str
    receiver_email: str
    use_tls: bool = True


class EmailNotifier:
    """Sends alert emails with cooldown to prevent notification spam."""

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_sent_at = {}

    @staticmethod
    def _seconds_since(ts: datetime | None) -> float:
        if ts is None:
            return 10**9
        return (datetime.now() - ts).total_seconds()

    def can_send(self, key: str) -> bool:
        return self._seconds_since(self.last_sent_at.get(key)) >= self.cooldown_seconds

    def _mark_sent(self, key: str) -> None:
        self.last_sent_at[key] = datetime.now()

    def send_intrusion_email(
        self,
        config: EmailConfig,
        alert: dict,
        sensor: dict,
        detection_count: int,
        image_path: str | None = None,
    ) -> tuple[bool, str]:
        """Send one intrusion alert email if cooldown allows."""
        notification_key = f"{alert.get('code', 'ALERT')}:{alert.get('severity', 'unknown')}"
        if not self.can_send(notification_key):
            return False, "cooldown-active"

        subject = f"[SMART MONITOR] {alert.get('severity', '').upper()} Intrusion Alert"
        body = (
            "Real-time intrusion alert detected.\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Alert: {alert.get('message', '')}\n"
            f"Risk Score: {alert.get('risk_score', 0.0)}\n"
            f"Detections in frame: {detection_count}\n\n"
            "Sensor Snapshot:\n"
            f"Temperature: {sensor.get('temperature', 0.0)} C\n"
            f"Humidity: {sensor.get('humidity', 0.0)} %\n"
            f"Gas: {sensor.get('gas', 0.0)} ppm\n"
        )
        if image_path:
            body += f"\nAttached intrusion image: {Path(image_path).name}\n"

        message = MIMEMultipart()
        message["From"] = config.sender_email
        message["To"] = config.receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        if image_path:
            path = Path(image_path)
            if path.exists() and path.is_file():
                with path.open("rb") as image_file:
                    attachment = MIMEApplication(image_file.read(), _subtype="jpeg")
                attachment.add_header("Content-Disposition", "attachment", filename=path.name)
                message.attach(attachment)

        try:
            if config.use_tls:
                server = smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=15)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=15)

            with server:
                server.login(config.sender_email, config.sender_password)
                server.sendmail(config.sender_email, config.receiver_email, message.as_string())

            self._mark_sent(notification_key)
            return True, "sent"
        except Exception as exc:
            return False, str(exc)
