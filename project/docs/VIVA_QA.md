# Viva Q and A

## Q1. What is the core domain of this project?
A. Physics-informed AI for smart monitoring and surveillance.

## Q2. Why combine vision and sensors?
A. Vision gives scene context while sensors provide physical environment state, improving reliability.

## Q3. What does PINN contribute here?
A. It constrains predictions with physical consistency instead of pure black-box fitting.

## Q4. How is speed computed?
A. First in px/s from track displacement over time, then converted to m/s using calibration scale.

## Q5. How do you avoid repeated alert spam?
A. Event-based deduplication with clear conditions and hysteresis.

## Q6. How do you evaluate quality?
A. Runtime logs plus generated evaluation report with accuracy, detection, tracking, and alert metrics.
