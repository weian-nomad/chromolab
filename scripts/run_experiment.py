#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one chromosome training experiment")
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--task", default="segment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_best_metrics(results_csv: Path) -> tuple[str | None, float | None, dict[str, float]]:
    if not results_csv.exists():
        return None, None, {}

    rows = []
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return None, None, {}

    preferred_columns = [
        "metrics/mAP50-95(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "fitness",
    ]
    numeric_columns: dict[str, float] = {}
    for key, value in rows[-1].items():
        try:
            numeric_columns[key] = float(value)
        except (TypeError, ValueError):
            continue

    for column in preferred_columns:
        if column in numeric_columns:
            return column, numeric_columns[column], numeric_columns
    if numeric_columns:
        first_key = sorted(numeric_columns)[0]
        return first_key, numeric_columns[first_key], numeric_columns
    return None, None, {}


def save_preview(best_weights: Path, dataset_root: Path, output_dir: Path) -> list[str]:
    model = YOLO(str(best_weights))
    source_dir = dataset_root / "images" / "val"
    if not source_dir.exists() or not any(source_dir.iterdir()):
        source_dir = dataset_root / "images" / "train"
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model.predict(
        source=str(source_dir),
        save=True,
        save_txt=False,
        project=str(preview_dir),
        name="pred",
        exist_ok=True,
        verbose=False,
        conf=0.25,
        max_det=200,
    )
    generated_dir = preview_dir / "pred"
    if not generated_dir.exists():
        return []
    return sorted(path.name for path in generated_dir.iterdir() if path.is_file())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = model.train(
        data=args.dataset_yaml,
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=args.device,
        project=str(training_dir),
        name="run",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    metric_name, metric_value, all_metrics = read_best_metrics(run_dir / "results.csv")
    preview_files = save_preview(best_weights, Path(args.dataset_yaml).resolve().parent, output_dir)

    summary = {
        "model_label": args.model_label,
        "weights": args.weights,
        "task": args.task,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "run_dir": str(run_dir),
        "best_weights": str(best_weights),
        "results_csv": str(run_dir / "results.csv"),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metrics": all_metrics,
        "preview_dir": str(output_dir / "preview" / "pred"),
        "preview_files": preview_files,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
