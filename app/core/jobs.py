from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from app.config import CHROMOSOME_ROOT, EXPERIMENTS_ROOT, JOBS_FILE, LOG_ROOT, MODEL_CANDIDATES, PORTAL_ROOT
from app.core.storage import STORE

JOB_LOCK = threading.RLock()
JOB_THREAD_STARTED = False
CURRENT_PROCESS: dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jobs() -> list[dict[str, Any]]:
    return STORE.load(JOBS_FILE, [])


def _save_jobs(jobs: list[dict[str, Any]]) -> None:
    STORE.save(JOBS_FILE, jobs)


def list_jobs() -> list[dict[str, Any]]:
    return sorted(_load_jobs(), key=lambda item: item["created_at"], reverse=True)


def get_job(job_id: str) -> dict[str, Any]:
    for job in _load_jobs():
        if job["id"] == job_id:
            return job
    raise HTTPException(status_code=404, detail="Job not found")


def job_log_tail(job_id: str, limit: int = 200) -> str:
    job = get_job(job_id)
    log_path = Path(job["log_path"])
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-limit:])


def completed_jobs_for_dataset(dataset_id: str, dataset_revision: int | None = None) -> list[dict[str, Any]]:
    return [
        job
        for job in _load_jobs()
        if job["dataset_id"] == dataset_id
        and job["status"] == "completed"
        and (dataset_revision is None or job.get("dataset_revision") == dataset_revision)
    ]


def create_job(
    *,
    dataset_id: str,
    dataset_yaml: str,
    requested_by: str,
    model_label: str,
    weights: str,
    task: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    dataset_revision: int | None = None,
    source: str = "manual",
    optimizer_id: str | None = None,
) -> dict[str, Any]:
    with JOB_LOCK:
        job_id = f"job-{uuid4().hex[:12]}"
        output_dir = EXPERIMENTS_ROOT / job_id
        log_path = LOG_ROOT / f"{job_id}.log"
        record = {
            "id": job_id,
            "dataset_id": dataset_id,
            "dataset_yaml": dataset_yaml,
            "requested_by": requested_by,
            "model_label": model_label,
            "weights": weights,
            "task": task,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "dataset_revision": dataset_revision,
            "source": source,
            "optimizer_id": optimizer_id,
            "status": "queued",
            "created_at": now_iso(),
            "started_at": None,
            "completed_at": None,
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "metric_name": None,
            "metric_value": None,
            "preview_files": [],
            "error": None,
        }
        jobs = _load_jobs()
        jobs.append(record)
        _save_jobs(jobs)
        return record


def create_comparison_jobs(
    *,
    dataset_id: str,
    dataset_yaml: str,
    requested_by: str,
    model_labels: list[str],
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    dataset_revision: int | None = None,
    source: str = "manual",
    optimizer_id: str | None = None,
) -> list[dict[str, Any]]:
    created = []
    candidate_map = {item["label"]: item for item in MODEL_CANDIDATES}
    for label in model_labels:
        candidate = candidate_map.get(label)
        if not candidate:
            raise HTTPException(status_code=400, detail=f"Unsupported model label: {label}")
        created.append(
            create_job(
                dataset_id=dataset_id,
                dataset_yaml=dataset_yaml,
                requested_by=requested_by,
                model_label=label,
                weights=str(CHROMOSOME_ROOT / candidate["weights"]),
                task=candidate["task"],
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                dataset_revision=dataset_revision,
                source=source,
                optimizer_id=optimizer_id,
            )
        )
    return created


def _update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    jobs = _load_jobs()
    for index, job in enumerate(jobs):
        if job["id"] != job_id:
            continue
        jobs[index] = {**job, **changes}
        _save_jobs(jobs)
        return jobs[index]
    raise HTTPException(status_code=404, detail="Job not found")


def _start_next_job() -> None:
    queued = next((job for job in _load_jobs() if job["status"] == "queued"), None)
    if not queued:
        return

    output_dir = Path(queued["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(queued["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(queued["log_path"], "a", encoding="utf-8")
    command = [
        str(PORTAL_ROOT / ".venv" / "bin" / "python"),
        str(PORTAL_ROOT / "scripts" / "run_experiment.py"),
        "--dataset-yaml",
        queued["dataset_yaml"],
        "--weights",
        queued["weights"],
        "--model-label",
        queued["model_label"],
        "--task",
        queued["task"],
        "--epochs",
        str(queued["epochs"]),
        "--imgsz",
        str(queued["imgsz"]),
        "--batch",
        str(queued["batch"]),
        "--device",
        str(queued["device"]),
        "--output-dir",
        queued["output_dir"],
    ]
    process = subprocess.Popen(
        command,
        cwd=str(CHROMOSOME_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    CURRENT_PROCESS.clear()
    CURRENT_PROCESS.update(
        {
            "job_id": queued["id"],
            "process": process,
            "log_handle": log_handle,
        }
    )
    _update_job(queued["id"], status="running", started_at=now_iso())


def _poll_running_job() -> None:
    if not CURRENT_PROCESS:
        return
    process: subprocess.Popen[str] = CURRENT_PROCESS["process"]
    return_code = process.poll()
    if return_code is None:
        return

    CURRENT_PROCESS.pop("process", None)
    log_handle = CURRENT_PROCESS.pop("log_handle")
    job_id = CURRENT_PROCESS.pop("job_id")
    log_handle.close()

    summary_path = Path(get_job(job_id)["summary_path"])
    if return_code == 0 and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _update_job(
            job_id,
            status="completed",
            completed_at=now_iso(),
            metric_name=summary.get("metric_name"),
            metric_value=summary.get("metric_value"),
            preview_files=summary.get("preview_files", []),
        )
    else:
        _update_job(
            job_id,
            status="failed",
            completed_at=now_iso(),
            error=f"Experiment process exited with code {return_code}",
        )


def _job_loop() -> None:
    while True:
        try:
            with JOB_LOCK:
                _poll_running_job()
                if not CURRENT_PROCESS:
                    _start_next_job()
        except Exception:
            pass
        time.sleep(3)


def ensure_job_worker_started() -> None:
    global JOB_THREAD_STARTED
    with JOB_LOCK:
        if JOB_THREAD_STARTED:
            return
        thread = threading.Thread(target=_job_loop, daemon=True, name="chromosome-job-worker")
        thread.start()
        JOB_THREAD_STARTED = True
