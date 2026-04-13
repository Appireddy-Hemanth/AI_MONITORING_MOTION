# Smart Monitoring Project - 7 Day Implementation Plan

## Goal
Upgrade the current system into a deployable, evaluation-ready project with:
- reliable person tracking analytics (speed + tracked time quality)
- real-world speed calibration
- configurable alert policy
- persistent logs for detections/events
- measurable performance reports for viva/demo

## Current Baseline (Already Done)
- Streamlit dashboard with camera + sensor simulation
- YOLO-based object/person detection
- person-only mode and comprehensive mode
- per-track speed (px/s) + tracked duration
- PINN-style environment and energy prediction
- intrusion and sensor alert deduplication

---

## Day 1 - Logging Foundation (Must Have)

### Files to create
- `utils/logger.py`
- `data/logs/` (directory)

### Files to update
- `app.py`

### Tasks
1. Create a structured event logger (CSV/JSONL).
2. Log these records per cycle:
   - timestamp
   - tracking id, class, confidence
   - speed_px_s, tracked_seconds
   - sensor values
   - PINN accuracies
   - risk score and alert code (if any)
3. Add daily log file rotation by date.

### Acceptance criteria
- New files appear under `data/logs/` while app runs.
- Each record has consistent columns and timestamp.

---

## Day 2 - Real-World Speed Calibration

### Files to create
- `utils/calibration.py`

### Files to update
- `app.py`
- `yolo/detect.py`

### Tasks
1. Add calibration mode in sidebar:
   - user enters known real distance (meters)
   - user marks corresponding pixel distance (2 points or manual input)
2. Compute scale:
   - `meters_per_pixel = known_distance_m / pixel_distance`
3. Convert speed:
   - `speed_m_s = speed_px_s * meters_per_pixel`
4. Show both units in overlay: px/s and m/s.

### Acceptance criteria
- Speed in m/s shown when calibration exists.
- Falls back to px/s if calibration missing.

---

## Day 3 - Alert Policy Engine

### Files to create
- `utils/alert_policy.py`

### Files to update
- `utils/alerts.py`
- `app.py`

### Tasks
1. Move alert behavior into policy object:
   - per-code cooldown
   - event clear duration
   - severity threshold
2. Add UI controls for policy values.
3. Save/load policy from JSON file:
   - `data/alert_policy.json`

### Acceptance criteria
- User can change policy at runtime from sidebar.
- Changes persist across app restarts.

---

## Day 4 - Tracking Quality Improvements

### Files to update
- `yolo/detect.py`

### Tasks
1. Add track confidence score based on:
   - detection confidence trend
   - consistency of class label
   - centroid continuity
2. Add max-gap reattach logic (short occlusion handling).
3. Ignore unstable newborn tracks until minimum age (example: 0.5s).

### Acceptance criteria
- Fewer ID switches.
- Speed output less jittery for same person.

---

## Day 5 - Evaluation Pipeline

### Files to create
- `utils/evaluation.py`
- `scripts/run_eval.py`
- `data/eval/` (directory)

### Files to update
- `requirements.txt` (if needed)

### Tasks
1. Compute key metrics from logs:
   - detection_count_by_class
   - avg_track_duration
   - avg_speed_per_class
   - alert_count_by_code
   - false_alert_rate proxy
   - PINN env/energy mean accuracy
2. Export summary CSV and readable text report.

### Acceptance criteria
- Running eval script creates summary report file.
- Metrics are presentation-ready.

---

## Day 6 - Dashboard Enhancements

### Files to update
- `app.py`

### Tasks
1. Add “Model Confidence & Health” section:
   - env accuracy trend
   - energy accuracy trend
   - detection confidence trend
2. Add “Tracking Analytics” section:
   - active track count
   - avg speed (person)
   - longest tracked person
3. Add quick filter toggles:
   - person only
   - living organisms only
   - all objects

### Acceptance criteria
- Presenter can explain performance visually without external tools.

---

## Day 7 - Packaging + Demo Readiness

### Files to create
- `README_DEMO.md`
- `docs/VIVA_QA.md`
- `docs/FORMULA_MAP.md`

### Files to update
- `README.md` (if present later)

### Tasks
1. Prepare 3 demo scenarios:
   - intrusion/person tracking
   - object/living-organism detection
   - sensor anomaly + alert behavior
2. Add runbook with exact steps and expected outputs.
3. Add troubleshooting section for camera and model thresholds.

### Acceptance criteria
- Someone else can run and demo project from documentation only.

---

## Priority If Time Is Limited
Implement in this order:
1. Day 1 (logging)
2. Day 2 (speed calibration)
3. Day 3 (alert policy)
4. Day 5 (evaluation)

This gives the strongest technical and viva impact quickly.

---

## Exact Formula Scope (for documentation)
- Speed conversion: `speed_m_s = speed_px_s * meters_per_pixel`
- Calibration scale: `meters_per_pixel = known_distance_m / pixel_distance_px`
- Energy physics core: `E = P * t`
- Risk fusion: weighted combination of vision confidence, sensor severity, and model uncertainty

Keep each formula documented under its own module in `docs/FORMULA_MAP.md`.

---

## Quick Implementation Checklist
- [ ] Structured logs saved per frame/event
- [ ] Calibration UI + m/s conversion
- [ ] Policy-based alert behavior with persistence
- [ ] Stable track IDs and confidence
- [ ] Evaluation script and summary report
- [ ] Dashboard analytics for demo
- [ ] Viva-ready docs package
