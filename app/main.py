from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import CHROMOSOME_ROOT, DEFAULT_CLASS_NAMES, PORTAL_ROOT, PORTAL_TITLE, ensure_directories
from app.core import auth
from app.core.datasets import (
    dataset_summary,
    export_dataset_index_csv,
    generate_recommendations,
    get_dataset,
    import_dataset_from_zip,
    list_dataset_images,
    list_datasets,
    read_annotation,
    refresh_dataset,
    write_annotation,
)
from app.core.jobs import (
    completed_jobs_for_dataset,
    create_comparison_jobs,
    ensure_job_worker_started,
    get_job,
    job_log_tail,
    list_jobs,
)
from app.core.optimizer import ensure_optimizer_started, optimizer_status, start_optimizer, stop_optimizer

ensure_directories()
auth.ensure_default_users()
ensure_job_worker_started()
ensure_optimizer_started()

app = FastAPI(title=PORTAL_TITLE)
app.mount("/static", StaticFiles(directory=str(PORTAL_ROOT / "app" / "static")), name="static")
LEGACY_GALLERY_ROOT = CHROMOSOME_ROOT / "runs" / "dose_inference_full"
if LEGACY_GALLERY_ROOT.exists():
    app.mount("/legacy-gallery", StaticFiles(directory=str(LEGACY_GALLERY_ROOT), html=True), name="legacy-gallery")


def asset_user(token: str | None = None, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if token:
        return auth.session_from_token(token)
    return auth.current_user(authorization)


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    display_name: str
    role: str
    password: str = Field(min_length=8)


class TrainingRequest(BaseModel):
    dataset_id: str
    model_labels: list[str]
    epochs: int = Field(default=10, ge=1, le=500)
    imgsz: int = Field(default=640, ge=320, le=2048)
    batch: int = Field(default=4, ge=1, le=64)
    device: str = Field(default="0")


class OptimizerRequest(BaseModel):
    dataset_id: str
    max_trials: int = Field(default=4, ge=1, le=20)
    epochs: int = Field(default=10, ge=1, le=500)
    imgsz_options: list[int] = Field(default_factory=lambda: [640, 768])
    batch: int = Field(default=4, ge=1, le=64)
    device: str = Field(default="0")


class AnnotationPolygon(BaseModel):
    class_id: int
    points: list[dict[str, float]]


class AnnotationWriteRequest(BaseModel):
    polygons: list[AnnotationPolygon]


@app.get("/")
async def intro() -> FileResponse:
    return FileResponse(PORTAL_ROOT / "app" / "static" / "intro.html")


@app.get("/portal")
@app.get("/portal/")
async def index() -> FileResponse:
    return FileResponse(PORTAL_ROOT / "app" / "static" / "index.html")


@app.post("/api/auth/login")
async def login(payload: LoginRequest) -> dict[str, Any]:
    user = auth.authenticate(payload.username, payload.password)
    token = auth.create_session(user)
    return {"token": token, "user": user}


@app.post("/api/auth/logout")
async def logout(user: dict[str, Any] = Depends(auth.current_user), authorization: str = Depends(auth.extract_token)) -> dict[str, str]:
    auth.revoke_session(authorization)
    return {"message": f"Logged out {user['username']}"}


@app.get("/api/auth/me")
async def me(user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    return user


@app.get("/api/overview")
async def overview(user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    datasets = list_datasets()
    jobs = list_jobs()
    return {
        "user": user,
        "dataset_count": len(datasets),
        "completed_jobs": len([job for job in jobs if job["status"] == "completed"]),
        "queued_jobs": len([job for job in jobs if job["status"] in {"queued", "running"}]),
        "default_classes": DEFAULT_CLASS_NAMES,
        "chromosome_root": str(CHROMOSOME_ROOT),
    }


@app.get("/api/users")
async def users(user: dict[str, Any] = Depends(auth.require_roles("admin"))) -> list[dict[str, Any]]:
    return auth.list_users()


@app.post("/api/users")
async def create_user(payload: CreateUserRequest, user: dict[str, Any] = Depends(auth.require_roles("admin"))) -> dict[str, Any]:
    return auth.create_user(
        username=payload.username,
        display_name=payload.display_name,
        role=payload.role.strip(),
        password=payload.password,
    )


@app.get("/api/datasets")
async def datasets(user: dict[str, Any] = Depends(auth.current_user)) -> list[dict[str, Any]]:
    items = list_datasets()
    refreshed = []
    for item in items:
        refreshed.append(refresh_dataset(item["id"], completed_jobs_for_dataset(item["id"], item.get("revision", 1))))
    return refreshed


@app.post("/api/datasets/upload")
async def upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
    user: dict[str, Any] = Depends(auth.require_roles("admin")),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    payload = await file.read()
    dataset = import_dataset_from_zip(name, file.filename, payload, user["username"])
    dataset = refresh_dataset(dataset["id"], completed_jobs_for_dataset(dataset["id"], dataset.get("revision", 1)))
    return dataset


@app.get("/api/datasets/{dataset_id}")
async def dataset_detail(dataset_id: str, user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    return refresh_dataset(dataset_id, completed_jobs_for_dataset(dataset_id, dataset.get("revision", 1)))


@app.get("/api/datasets/{dataset_id}/summary")
async def dataset_stats(dataset_id: str, user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    return dataset_summary(dataset_id)


@app.get("/api/datasets/{dataset_id}/export")
async def dataset_export(dataset_id: str, user: dict[str, Any] = Depends(auth.current_user)) -> FileResponse:
    return FileResponse(export_dataset_index_csv(dataset_id))


@app.get("/api/datasets/{dataset_id}/images")
async def dataset_images(
    dataset_id: str,
    labeled: str | None = None,
    split: str | None = None,
    user: dict[str, Any] = Depends(auth.current_user),
) -> list[dict[str, Any]]:
    return list_dataset_images(dataset_id, labeled=labeled, split=split)


@app.get("/api/datasets/{dataset_id}/images/{image_key:path}/file")
async def dataset_image_file(dataset_id: str, image_key: str, user: dict[str, Any] = Depends(asset_user)) -> FileResponse:
    dataset = get_dataset(dataset_id)
    split, filename = image_key.strip("/").split("/", 1)
    image_path = Path(dataset["dataset_root"]) / "images" / split / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.get("/api/datasets/{dataset_id}/images/{image_key:path}/annotation")
async def annotation(dataset_id: str, image_key: str, user: dict[str, Any] = Depends(auth.require_roles("admin", "annotator"))) -> dict[str, Any]:
    return read_annotation(dataset_id, image_key)


@app.put("/api/datasets/{dataset_id}/images/{image_key:path}/annotation")
async def save_annotation(
    dataset_id: str,
    image_key: str,
    payload: AnnotationWriteRequest,
    user: dict[str, Any] = Depends(auth.require_roles("admin", "annotator")),
) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    return write_annotation(
        dataset_id,
        image_key,
        [polygon.model_dump() for polygon in payload.polygons],
        completed_jobs_for_dataset(dataset_id, int(dataset.get("revision", 1)) + 1),
    )


@app.post("/api/training/jobs")
async def training_jobs(payload: TrainingRequest, user: dict[str, Any] = Depends(auth.require_roles("admin"))) -> dict[str, Any]:
    if not payload.model_labels:
        raise HTTPException(status_code=400, detail="At least one model must be selected")
    current_dataset = get_dataset(payload.dataset_id)
    dataset = refresh_dataset(payload.dataset_id, completed_jobs_for_dataset(payload.dataset_id, current_dataset.get("revision", 1)))
    if dataset["labeled_image_count"] == 0:
        raise HTTPException(status_code=400, detail="Dataset has no labeled images")
    jobs = create_comparison_jobs(
        dataset_id=payload.dataset_id,
        dataset_yaml=dataset["dataset_yaml"],
        requested_by=user["username"],
        model_labels=payload.model_labels,
        epochs=payload.epochs,
        imgsz=payload.imgsz,
        batch=payload.batch,
        device=payload.device,
        dataset_revision=dataset.get("revision", 1),
    )
    return {"jobs": jobs}


@app.get("/api/training/jobs")
async def training_job_list(
    dataset_id: str | None = None,
    user: dict[str, Any] = Depends(auth.current_user),
) -> list[dict[str, Any]]:
    jobs = list_jobs()
    if dataset_id:
        jobs = [job for job in jobs if job["dataset_id"] == dataset_id]
    return jobs


@app.get("/api/training/jobs/{job_id}")
async def training_job_detail(job_id: str, user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    return get_job(job_id)


@app.get("/api/training/jobs/{job_id}/log")
async def training_job_log(job_id: str, user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, str]:
    return {"log": job_log_tail(job_id)}


@app.get("/api/training/jobs/{job_id}/preview/{filename}")
async def training_preview_file(job_id: str, filename: str, user: dict[str, Any] = Depends(asset_user)) -> FileResponse:
    job = get_job(job_id)
    preview_path = Path(job["output_dir"]) / "preview" / "pred" / filename
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview file not found")
    return FileResponse(preview_path)


@app.get("/api/models/compare")
async def model_compare(
    dataset_id: str,
    user: dict[str, Any] = Depends(auth.current_user),
) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    completed = completed_jobs_for_dataset(dataset_id, dataset.get("revision", 1))
    ranked = sorted(completed, key=lambda item: item.get("metric_value") or -1, reverse=True)
    return {
        "dataset": dataset_summary(dataset_id),
        "recommendations": generate_recommendations(dataset, completed),
        "completed_jobs": ranked,
    }


@app.post("/api/optimizer/start")
async def optimizer_start(payload: OptimizerRequest, user: dict[str, Any] = Depends(auth.require_roles("admin"))) -> dict[str, Any]:
    current_dataset = get_dataset(payload.dataset_id)
    dataset = refresh_dataset(payload.dataset_id, completed_jobs_for_dataset(payload.dataset_id, current_dataset.get("revision", 1)))
    if dataset["labeled_image_count"] == 0:
        raise HTTPException(status_code=400, detail="Dataset has no labeled images")
    return start_optimizer(
        dataset_id=payload.dataset_id,
        dataset_yaml=dataset["dataset_yaml"],
        dataset_revision=dataset.get("revision", 1),
        requested_by=user["username"],
        max_trials=payload.max_trials,
        epochs=payload.epochs,
        imgsz_options=payload.imgsz_options,
        batch=payload.batch,
        device=payload.device,
    )


@app.post("/api/optimizer/stop")
async def optimizer_stop(user: dict[str, Any] = Depends(auth.require_roles("admin"))) -> dict[str, Any]:
    return stop_optimizer()


@app.get("/api/optimizer/status")
async def optimizer_state(user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    return optimizer_status()
