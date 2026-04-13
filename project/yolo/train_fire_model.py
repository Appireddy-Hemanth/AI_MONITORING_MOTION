"""Train a dedicated YOLO fire model and export best weights for app usage.

Usage example:
python yolo/train_fire_model.py --data data/fire_dataset/data.yaml --epochs 60 --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO fire detector")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml for fire dataset")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", type=str, default="runs/fire_train", help="Ultralytics project dir")
    parser.add_argument("--name", type=str, default="yolov8_fire", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO("yolov8n.pt")
    train_result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True,
    )

    best_path = Path(train_result.save_dir) / "weights" / "best.pt"
    print(f"Training done. Best model: {best_path}")
    print("Use this in app sidebar 'Optional Fire Model Path (.pt)'.")


if __name__ == "__main__":
    main()
