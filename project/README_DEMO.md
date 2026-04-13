# Smart Monitoring Demo Runbook

## Quick Start
1. Activate environment and run app:
   - python -m streamlit run app.py
2. Open local URL shown in terminal.
3. In sidebar configure policy, trust, and video source.

## Review Mode Flow
1. Go to Review Mode page.
2. Click Start to begin monitoring.
3. Show key metrics tiles.
4. Show latest alerts list.
5. Click Generate Evaluation Report.
6. Click Export Review Bundle.

## Evidence Files
- Logs: data/logs
- Evaluation: data/eval
- Bundle: data/review_bundle
- Scenario traces: data/scenarios

## Common Troubleshooting
- If camera is blank:
  - close other apps using webcam
  - click Stop then Start
- If detections are low:
  - lower confidence threshold
  - reduce per-class thresholds in Vision settings
- If alerts are too frequent:
  - increase cooldown in Alert Policy panel

## Cloud Deployment Notes
- Vercel serves a Python serverless entrypoint and will not host the full Streamlit realtime UI.
- To host the complete app UI, deploy to Render (recommended) using `render.yaml` at repo root.

### Render Deployment
1. Push this repository to GitHub.
2. In Render, choose "New" -> "Blueprint" and select the repository.
3. Render will auto-detect `render.yaml` and create service `ai-smart-monitoring`.
4. Wait for build and open the generated Render URL.
