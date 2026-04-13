from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def _copy_latest(pattern: str, src_dir: Path, dst_dir: Path, count: int = 1) -> list[str]:
    matches = sorted(src_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    copied = []
    for path in matches[:count]:
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)
            copied.append(path.name)
    return copied


def create_review_bundle(base_dir: str = "data") -> tuple[str, dict]:
    base = Path(base_dir)
    bundle_root = base / "review_bundle"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = bundle_root / f"bundle_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, list[str] | str] = {}

    logs_dir = base / "logs"
    eval_dir = base / "eval"
    incidents_dir = base / "incidents"

    if logs_dir.exists():
        summary["logs"] = _copy_latest("*.csv", logs_dir, out_dir, count=6)
    else:
        summary["logs"] = []

    if eval_dir.exists():
        copied_eval = []
        copied_eval.extend(_copy_latest("evaluation_summary_*.csv", eval_dir, out_dir, count=2))
        copied_eval.extend(_copy_latest("evaluation_report_*.txt", eval_dir, out_dir, count=2))
        summary["evaluation"] = copied_eval
    else:
        summary["evaluation"] = []

    copied_cfg = []
    for cfg_name in ["calibration.json", "alert_policy.json"]:
        src = base / cfg_name
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
            copied_cfg.append(src.name)
    summary["config"] = copied_cfg

    if incidents_dir.exists():
        summary["incidents"] = _copy_latest("*.jpg", incidents_dir, out_dir, count=10)
    else:
        summary["incidents"] = []

    manifest = out_dir / "manifest.txt"
    lines = [
        "Smart Monitoring Review Bundle",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for key, values in summary.items():
        lines.append(f"{key}:")
        if values:
            for v in values:
                lines.append(f"- {v}")
        else:
            lines.append("- none")
        lines.append("")
    manifest.write_text("\n".join(lines), encoding="utf-8")

    return str(out_dir), summary
