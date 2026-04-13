from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class ScenarioRecorder:
    def __init__(self, out_dir: str = "data/scenarios") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.active = False
        self.scenario_name = ""
        self.file_path = Path()

    def start(self, scenario_name: str) -> str:
        cleaned = "_".join(scenario_name.strip().split()) or "unnamed_scenario"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.out_dir / f"{cleaned}_{stamp}.csv"
        self.scenario_name = cleaned
        self.active = True

        with self.file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "details"])

        self.log_event("scenario_start", f"name={self.scenario_name}")
        return str(self.file_path)

    def stop(self) -> str:
        if self.active:
            self.log_event("scenario_stop", "manual stop")
        self.active = False
        return str(self.file_path) if self.file_path else ""

    def log_event(self, event: str, details: str = "") -> None:
        if not self.active or not self.file_path:
            return
        with self.file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, details])
