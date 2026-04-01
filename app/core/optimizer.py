from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.config import CHROMOSOME_ROOT, MODEL_CANDIDATES, OPTIMIZER_FILE
from app.core.datasets import get_dataset
from app.core.jobs import create_job, list_jobs
from app.core.storage import STORE

OPTIMIZER_LOCK = threading.RLock()
OPTIMIZER_THREAD_STARTED = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> dict[str, Any]:
    return STORE.load(
        OPTIMIZER_FILE,
        {
            "enabled": False,
            "optimizer_id": None,
            "dataset_id": None,
            "dataset_yaml": None,
            "dataset_revision": 1,
            "requested_by": None,
            "max_trials": 0,
            "trials_started": 0,
            "total_trials_started": 0,
            "imgsz_options": [640],
            "epochs": 10,
            "batch": 4,
            "device": "0",
            "search_space": [],
            "created_at": None,
            "last_suggestion": None,
        },
    )


def _save_state(state: dict[str, Any]) -> None:
    STORE.save(OPTIMIZER_FILE, state)


def optimizer_status() -> dict[str, Any]:
    state = _load_state()
    jobs = [
        job
        for job in list_jobs()
        if state["optimizer_id"] and job.get("optimizer_id") == state["optimizer_id"]
    ]
    return {**state, "jobs": jobs}


def start_optimizer(
    *,
    dataset_id: str,
    dataset_yaml: str,
    requested_by: str,
    dataset_revision: int,
    max_trials: int,
    epochs: int,
    imgsz_options: list[int],
    batch: int,
    device: str,
) -> dict[str, Any]:
    with OPTIMIZER_LOCK:
        optimizer_id = f"opt-{int(time.time())}"
        search_space = []
        for candidate in MODEL_CANDIDATES:
            if candidate["task"] != "segment":
                continue
            for imgsz in imgsz_options:
                search_space.append(
                    {
                        "model_label": candidate["label"],
                        "weights": candidate["weights"],
                        "task": candidate["task"],
                        "imgsz": imgsz,
                    }
                )

        state = {
            "enabled": True,
            "optimizer_id": optimizer_id,
            "dataset_id": dataset_id,
            "dataset_yaml": dataset_yaml,
            "dataset_revision": dataset_revision,
            "requested_by": requested_by,
            "max_trials": max_trials,
            "trials_started": 0,
            "total_trials_started": 0,
            "imgsz_options": imgsz_options,
            "epochs": epochs,
            "batch": batch,
            "device": device,
            "search_space": search_space,
            "created_at": now_iso(),
            "last_suggestion": f"Optimizer started on revision {dataset_revision}",
        }
        _save_state(state)
        return state


def stop_optimizer() -> dict[str, Any]:
    with OPTIMIZER_LOCK:
        state = _load_state()
        state["enabled"] = False
        state["last_suggestion"] = "Optimizer stopped"
        _save_state(state)
        return state


def _queue_next_trial() -> bool:
    state = _load_state()
    if not state["enabled"]:
        return False
    dataset = get_dataset(state["dataset_id"])
    current_revision = int(dataset.get("revision", 1))
    state["dataset_yaml"] = dataset["dataset_yaml"]
    if current_revision != int(state.get("dataset_revision", 1)):
        state["dataset_revision"] = current_revision
        state["trials_started"] = 0
        state["last_suggestion"] = f"Detected dataset revision {current_revision}; restarting search"
        _save_state(state)

    existing = [
        job
        for job in list_jobs()
        if job.get("optimizer_id") == state["optimizer_id"] and job.get("dataset_revision") == current_revision
    ]
    active = [job for job in existing if job["status"] in {"queued", "running"}]
    if active:
        return False

    tried = {(job["model_label"], job["imgsz"]) for job in existing}
    state["trials_started"] = len(tried)
    if state["trials_started"] >= state["max_trials"]:
        state["last_suggestion"] = f"Revision {current_revision} reached max_trials; waiting for new annotations"
        _save_state(state)
        return False

    next_candidate = None
    for candidate in state["search_space"]:
        key = (candidate["model_label"], candidate["imgsz"])
        if key not in tried:
            next_candidate = candidate
            break
    if not next_candidate:
        state["last_suggestion"] = f"Revision {current_revision} search space exhausted; waiting for new annotations"
        _save_state(state)
        return False

    create_job(
        dataset_id=state["dataset_id"],
        dataset_yaml=state["dataset_yaml"],
        requested_by=state["requested_by"],
        model_label=next_candidate["model_label"],
        weights=str(CHROMOSOME_ROOT / next_candidate["weights"]),
        task=next_candidate["task"],
        epochs=state["epochs"],
        imgsz=next_candidate["imgsz"],
        batch=state["batch"],
        device=state["device"],
        dataset_revision=current_revision,
        source="optimizer",
        optimizer_id=state["optimizer_id"],
    )
    state["trials_started"] += 1
    state["total_trials_started"] = int(state.get("total_trials_started", 0)) + 1
    state["last_suggestion"] = f"Queued revision {current_revision}: {next_candidate['model_label']} @ {next_candidate['imgsz']}"
    _save_state(state)
    return True


def _optimizer_loop() -> None:
    while True:
        try:
            with OPTIMIZER_LOCK:
                _queue_next_trial()
        except Exception:
            pass
        time.sleep(5)


def ensure_optimizer_started() -> None:
    global OPTIMIZER_THREAD_STARTED
    with OPTIMIZER_LOCK:
        if OPTIMIZER_THREAD_STARTED:
            return
        thread = threading.Thread(target=_optimizer_loop, daemon=True, name="chromosome-optimizer")
        thread.start()
        OPTIMIZER_THREAD_STARTED = True
