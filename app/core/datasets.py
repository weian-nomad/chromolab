from __future__ import annotations

import csv
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.config import DATASETS_FILE, DATASETS_ROOT, DEFAULT_CLASS_NAMES
from app.core.storage import STORE

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_datasets() -> list[dict[str, Any]]:
    return STORE.load(DATASETS_FILE, [])


def _save_datasets(datasets: list[dict[str, Any]]) -> None:
    STORE.save(DATASETS_FILE, datasets)


def list_datasets() -> list[dict[str, Any]]:
    return sorted(_load_datasets(), key=lambda item: item["created_at"], reverse=True)


def get_dataset(dataset_id: str) -> dict[str, Any]:
    for item in _load_datasets():
        if item["id"] == dataset_id:
            return item
    raise HTTPException(status_code=404, detail="Dataset not found")


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "dataset"


def _detect_classes(extract_root: Path) -> list[str]:
    obj_names = next(extract_root.rglob("obj.names"), None)
    if obj_names and obj_names.exists():
        return [line.strip() for line in obj_names.read_text(encoding="utf-8").splitlines() if line.strip()]

    classes_txt = next(extract_root.rglob("classes.txt"), None)
    if classes_txt and classes_txt.exists():
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]

    data_yaml = next(extract_root.rglob("data.yaml"), None)
    if data_yaml and data_yaml.exists():
        text = data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines()
        names: list[str] = []
        names_section = False
        for line in text:
            stripped = line.strip()
            if stripped.startswith("names:"):
                names_section = True
                suffix = stripped.split(":", 1)[1].strip()
                if suffix.startswith("[") and suffix.endswith("]"):
                    values = suffix.strip("[]")
                    return [item.strip().strip("'\"") for item in values.split(",") if item.strip()]
                continue
            if names_section:
                if stripped.startswith("- "):
                    names.append(stripped[2:].strip().strip("'\""))
                elif stripped and not line.startswith(" "):
                    break
        if names:
            return names

    return DEFAULT_CLASS_NAMES[:]


def _collect_files(extract_root: Path) -> tuple[list[Path], dict[str, Path]]:
    image_files: list[Path] = []
    labels_by_stem: dict[str, Path] = {}
    for path in extract_root.rglob("*"):
        if path.is_dir():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image_files.append(path)
        elif path.suffix.lower() == ".txt":
            labels_by_stem[path.stem] = path
    if not image_files:
        raise HTTPException(status_code=400, detail="Zip file does not contain any images")
    return sorted(image_files), labels_by_stem


def _split_name(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    return "val" if "val" in parts or "valid" in parts else "train"


def _normalize_label_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    lines = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    return lines


def _infer_task(label_lines: list[str]) -> str:
    for line in label_lines:
        if len(line.split()) > 5:
            return "segment"
    return "detect"


def _write_dataset_yaml(dataset_root: Path, classes: list[str]) -> Path:
    yaml_path = dataset_root / "dataset.yaml"
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  - {name}" for name in classes)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _compute_stats(dataset_root: Path, classes: list[str]) -> dict[str, Any]:
    split_counts: dict[str, int] = {"train": 0, "val": 0}
    labeled_images = 0
    unlabeled_images = 0
    task = "segment"
    class_counter: Counter[str] = Counter()
    images: list[dict[str, Any]] = []

    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        for image_path in sorted(image_dir.iterdir() if image_dir.exists() else []):
            if not image_path.is_file():
                continue
            split_counts[split] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            label_lines = _normalize_label_lines(label_path)
            is_labeled = bool(label_lines)
            if is_labeled:
                labeled_images += 1
                task = _infer_task(label_lines)
            else:
                unlabeled_images += 1
            for line in label_lines:
                try:
                    class_index = int(line.split()[0])
                except (IndexError, ValueError):
                    continue
                if 0 <= class_index < len(classes):
                    class_counter[classes[class_index]] += 1
            images.append(
                {
                    "key": f"{split}/{image_path.name}",
                    "split": split,
                    "filename": image_path.name,
                    "labeled": is_labeled,
                }
            )

    return {
        "split_counts": split_counts,
        "image_count": split_counts["train"] + split_counts["val"],
        "labeled_image_count": labeled_images,
        "unlabeled_image_count": unlabeled_images,
        "class_distribution": dict(class_counter),
        "task": task,
        "images": images,
    }


def generate_recommendations(dataset: dict[str, Any], completed_jobs: list[dict[str, Any]]) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    image_count = dataset["image_count"]
    unlabeled = dataset["unlabeled_image_count"]
    labeled = dataset["labeled_image_count"]
    task = dataset["task"]

    if unlabeled:
        recommendations.append(
            {
                "severity": "warning",
                "title": "仍有未標註影像",
                "detail": f"目前 {unlabeled} 張影像尚未標註，建議先交給 annotator 完成再啟動正式優化。",
            }
        )
    if labeled < 10:
        recommendations.append(
            {
                "severity": "warning",
                "title": "標註量偏少",
                "detail": f"目前只有 {labeled} 張已標註影像，適合先做 1-5 epoch smoke run，確認流程正常。",
            }
        )
    if image_count >= 10:
        recommendations.append(
            {
                "severity": "info",
                "title": "可啟動模型比較",
                "detail": "建議先比較 YOLOv8n-seg 與 YOLO11n-seg，並把 mAP50-95(M) 當作主要指標。",
            }
        )
    if task == "segment":
        recommendations.append(
            {
                "severity": "success",
                "title": "適合持續優化",
                "detail": "資料集已採用 YOLO segmentation 格式，可直接交給 optimizer 自動排隊比模型與參數。",
            }
        )
    if completed_jobs:
        best_job = max(completed_jobs, key=lambda item: item.get("metric_value") or -1)
        if best_job.get("metric_name"):
            recommendations.append(
                {
                    "severity": "info",
                    "title": "目前最佳模型",
                    "detail": f"{best_job['model_label']} 目前最佳，{best_job['metric_name']} = {best_job['metric_value']}",
                }
            )
    return recommendations


def import_dataset_from_zip(dataset_name: str, upload_name: str, payload: bytes, uploaded_by: str) -> dict[str, Any]:
    dataset_id = f"{_slug(dataset_name or Path(upload_name).stem)}-{int(datetime.now().timestamp())}"
    dataset_root = DATASETS_ROOT / dataset_id
    if dataset_root.exists():
        raise HTTPException(status_code=409, detail="Dataset id collision")

    with tempfile.TemporaryDirectory(prefix="chromosome-upload-") as temp_dir:
        archive_path = Path(temp_dir) / upload_name
        archive_path.write_bytes(payload)
        extract_root = Path(temp_dir) / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_root)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Only zip uploads are supported") from exc

        classes = _detect_classes(extract_root)
        image_files, labels_by_stem = _collect_files(extract_root)
        dataset_root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        copied_images: list[tuple[Path, Path, str]] = []
        for image_path in image_files:
            split = _split_name(image_path)
            target_path = dataset_root / "images" / split / image_path.name
            if target_path.exists():
                target_path = dataset_root / "images" / split / f"{image_path.stem}-{len(copied_images)}{image_path.suffix.lower()}"
            shutil.copy2(image_path, target_path)
            copied_images.append((image_path, target_path, split))

            label_path = labels_by_stem.get(image_path.stem)
            if label_path and label_path.exists():
                shutil.copy2(label_path, dataset_root / "labels" / split / f"{target_path.stem}.txt")

        train_images = sorted((dataset_root / "images" / "train").iterdir())
        val_images = sorted((dataset_root / "images" / "val").iterdir())
        if not val_images and len(train_images) >= 2:
            sample_size = max(1, int(len(train_images) * 0.2))
            moved = set(random.sample(train_images, sample_size))
            for image_path in moved:
                label_path = dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
                image_target = dataset_root / "images" / "val" / image_path.name
                image_path.replace(image_target)
                if label_path.exists():
                    label_path.replace(dataset_root / "labels" / "val" / label_path.name)
        elif not val_images and len(train_images) == 1:
            image_path = train_images[0]
            image_target = dataset_root / "images" / "val" / image_path.name
            shutil.copy2(image_path, image_target)
            label_path = dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, dataset_root / "labels" / "val" / label_path.name)

    dataset_yaml = _write_dataset_yaml(dataset_root, classes)
    stats = _compute_stats(dataset_root, classes)
    record = {
        "id": dataset_id,
        "name": dataset_name or Path(upload_name).stem,
        "source_filename": upload_name,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "uploaded_by": uploaded_by,
        "revision": 1,
        "classes": classes,
        "dataset_root": str(dataset_root),
        "dataset_yaml": str(dataset_yaml),
        **stats,
        "recommendations": [],
    }

    datasets = _load_datasets()
    datasets.append(record)
    _save_datasets(datasets)
    return record


def refresh_dataset(dataset_id: str, completed_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = _load_datasets()
    for index, dataset in enumerate(datasets):
        if dataset["id"] != dataset_id:
            continue
        dataset_root = Path(dataset["dataset_root"])
        stats = _compute_stats(dataset_root, dataset["classes"])
        datasets[index] = {
            **dataset,
            **stats,
            "updated_at": dataset.get("updated_at") or dataset["created_at"],
            "revision": dataset.get("revision", 1),
            "recommendations": generate_recommendations(dataset | stats, completed_jobs),
        }
        _save_datasets(datasets)
        return datasets[index]
    raise HTTPException(status_code=404, detail="Dataset not found")


def _image_path(dataset: dict[str, Any], image_key: str) -> Path:
    image_key = image_key.strip("/")
    split, filename = image_key.split("/", 1)
    return Path(dataset["dataset_root"]) / "images" / split / filename


def _label_path(dataset: dict[str, Any], image_key: str) -> Path:
    image_key = image_key.strip("/")
    split, filename = image_key.split("/", 1)
    return Path(dataset["dataset_root"]) / "labels" / split / f"{Path(filename).stem}.txt"


def list_dataset_images(dataset_id: str, *, labeled: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    dataset = get_dataset(dataset_id)
    images = dataset["images"]
    if split:
        images = [item for item in images if item["split"] == split]
    if labeled == "labeled":
        images = [item for item in images if item["labeled"]]
    elif labeled == "unlabeled":
        images = [item for item in images if not item["labeled"]]
    return images


def read_annotation(dataset_id: str, image_key: str) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    image_path = _image_path(dataset, image_key)
    label_path = _label_path(dataset, image_key)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    polygons = []
    for line in _normalize_label_lines(label_path):
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(parts[0])
            coords = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        points = []
        for idx in range(0, len(coords), 2):
            if idx + 1 >= len(coords):
                break
            points.append({"x": coords[idx], "y": coords[idx + 1]})
        polygons.append({"class_id": class_id, "points": points})

    return {
        "image_key": image_key,
        "filename": image_path.name,
        "split": image_key.split("/", 1)[0],
        "polygons": polygons,
        "classes": dataset["classes"],
    }


def write_annotation(dataset_id: str, image_key: str, polygons: list[dict[str, Any]], completed_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    label_path = _label_path(dataset, image_key)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for polygon in polygons:
        class_id = int(polygon["class_id"])
        points = polygon.get("points") or []
        if len(points) < 3:
            continue
        coords = []
        for point in points:
            coords.append(f"{float(point['x']):.6f}")
            coords.append(f"{float(point['y']):.6f}")
        lines.append(f"{class_id} {' '.join(coords)}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    datasets = _load_datasets()
    for index, item in enumerate(datasets):
        if item["id"] != dataset_id:
            continue
        datasets[index] = {
            **item,
            "updated_at": now_iso(),
            "revision": int(item.get("revision", 1)) + 1,
        }
        _save_datasets(datasets)
        break
    refreshed = refresh_dataset(dataset_id, completed_jobs)
    return read_annotation(dataset_id, image_key) | {"dataset": refreshed}


def dataset_summary(dataset_id: str) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    summary = {key: value for key, value in dataset.items() if key != "images"}
    return summary


def export_dataset_index_csv(dataset_id: str) -> Path:
    dataset = get_dataset(dataset_id)
    export_path = Path(dataset["dataset_root"]) / "dataset_index.csv"
    with export_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "split", "filename", "labeled"])
        writer.writeheader()
        writer.writerows(dataset["images"])
    return export_path
