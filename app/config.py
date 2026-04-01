from __future__ import annotations

from pathlib import Path


PORTAL_ROOT = Path(__file__).resolve().parent.parent
CHROMOSOME_ROOT = PORTAL_ROOT.parent
CONFIG_ROOT = PORTAL_ROOT / "config"
DATA_ROOT = PORTAL_ROOT / "data"
MANAGED_ROOT = DATA_ROOT / "managed"
DATASETS_ROOT = MANAGED_ROOT / "datasets"
EXPERIMENTS_ROOT = MANAGED_ROOT / "experiments"
STATE_ROOT = DATA_ROOT / "state"
UPLOAD_ROOT = DATA_ROOT / "uploads"
LOG_ROOT = DATA_ROOT / "logs"

USERS_FILE = STATE_ROOT / "users.json"
SESSIONS_FILE = STATE_ROOT / "sessions.json"
DATASETS_FILE = STATE_ROOT / "datasets.json"
JOBS_FILE = STATE_ROOT / "jobs.json"
OPTIMIZER_FILE = STATE_ROOT / "optimizer.json"

DEFAULT_USERS_FILE = CONFIG_ROOT / "default_users.json"
PORTAL_TITLE = "ChromoLab Ops Portal"
DEFAULT_CLASS_NAMES = [
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
]

MODEL_CANDIDATES = [
    {
        "label": "YOLOv8n Segmentation",
        "weights": "yolov8n-seg.pt",
        "task": "segment",
    },
    {
        "label": "YOLO11n Segmentation",
        "weights": "yolo11n-seg.pt",
        "task": "segment",
    },
]


def ensure_directories() -> None:
    for path in [
        DATA_ROOT,
        MANAGED_ROOT,
        DATASETS_ROOT,
        EXPERIMENTS_ROOT,
        STATE_ROOT,
        UPLOAD_ROOT,
        LOG_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)
