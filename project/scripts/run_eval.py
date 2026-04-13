from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluation import evaluate_logs, export_evaluation


def main() -> None:
    result = evaluate_logs("data/logs")
    csv_path, txt_path = export_evaluation(result, "data/eval")
    print("Evaluation completed")
    print(f"CSV summary: {csv_path}")
    print(f"Text report: {txt_path}")


if __name__ == "__main__":
    main()
