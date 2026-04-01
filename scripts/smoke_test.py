#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests

BASE_URL = "http://127.0.0.1:8090"
LOGIN = {"username": "admin", "password": "admin1234"}


def build_test_zip(repo_root: Path) -> bytes:
    image_dir = repo_root / "seg_data" / "images" / "train"
    label_dir = repo_root / "seg_data" / "labels" / "train"
    image_paths = sorted(image_dir.iterdir())[:2]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("obj.names", "\n".join([
            "Centromere",
            "Chromosome",
            "Dicentric",
            "Tricentric",
            "Quntracentric",
            "Acentric",
            "Ring",
            "Pair",
            "Abnormal_Zone",
            "Acentrics",
        ]))
        for image_path in image_paths:
            archive.write(image_path, f"images/train/{image_path.name}")
            archive.write(label_dir / f"{image_path.stem}.txt", f"labels/train/{image_path.stem}.txt")
    return buffer.getvalue()


def wait_for_jobs(token: str, job_ids: list[str], timeout: int = 1200) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    started = time.time()
    while time.time() - started < timeout:
        response = requests.get(f"{BASE_URL}/api/training/jobs", headers=headers, timeout=30)
        response.raise_for_status()
        jobs = response.json()
        current = [job for job in jobs if job["id"] in job_ids]
        if current and all(job["status"] in {"completed", "failed"} for job in current):
            return current
        time.sleep(5)
    raise TimeoutError("Jobs did not finish in time")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    login = requests.post(f"{BASE_URL}/api/auth/login", json=LOGIN, timeout=30)
    login.raise_for_status()
    token = login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    payload = build_test_zip(repo_root)
    files = {"file": ("smoke-dataset.zip", payload, "application/zip")}
    data = {"name": "smoke-dataset"}
    upload = requests.post(f"{BASE_URL}/api/datasets/upload", headers=headers, files=files, data=data, timeout=120)
    upload.raise_for_status()
    dataset = upload.json()
    dataset_id = dataset["id"]
    print("uploaded", dataset_id, dataset["image_count"])

    images = requests.get(f"{BASE_URL}/api/datasets/{dataset_id}/images", headers=headers, timeout=30)
    images.raise_for_status()
    image_key = images.json()[0]["key"]
    annotation = requests.get(
        f"{BASE_URL}/api/datasets/{dataset_id}/images/{quote(image_key, safe='')}/annotation",
        headers=headers,
        timeout=30,
    )
    annotation.raise_for_status()
    polygons = annotation.json()["polygons"]
    save = requests.put(
        f"{BASE_URL}/api/datasets/{dataset_id}/images/{quote(image_key, safe='')}/annotation",
        headers=headers,
        json={"polygons": polygons},
        timeout=30,
    )
    save.raise_for_status()
    print("annotation_saved", image_key)

    train = requests.post(
        f"{BASE_URL}/api/training/jobs",
        headers=headers,
        json={
            "dataset_id": dataset_id,
            "model_labels": ["YOLOv8n Segmentation", "YOLO11n Segmentation"],
            "epochs": 1,
            "imgsz": 640,
            "batch": 1,
            "device": "0",
        },
        timeout=30,
    )
    train.raise_for_status()
    jobs = train.json()["jobs"]
    job_ids = [job["id"] for job in jobs]
    print("queued_jobs", job_ids)

    final_jobs = wait_for_jobs(token, job_ids)
    print("final_jobs", json.dumps(final_jobs, ensure_ascii=False))

    compare = requests.get(
        f"{BASE_URL}/api/models/compare",
        headers=headers,
        params={"dataset_id": dataset_id},
        timeout=30,
    )
    compare.raise_for_status()
    print("compare_count", len(compare.json()["completed_jobs"]))

    optimizer = requests.post(
        f"{BASE_URL}/api/optimizer/start",
        headers=headers,
        json={
            "dataset_id": dataset_id,
            "max_trials": 1,
            "epochs": 1,
            "imgsz_options": [640],
            "batch": 1,
            "device": "0",
        },
        timeout=30,
    )
    optimizer.raise_for_status()
    print("optimizer_started", optimizer.json()["optimizer_id"])


if __name__ == "__main__":
    main()
