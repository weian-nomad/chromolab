# ChromoLab Ops Portal

A FastAPI-based admin portal for chromosome dataset operations.

## What It Does

- Role-based access for `admin`, `annotator`, and `viewer`
- Upload YOLO-format dataset ZIP files
- Review dataset statistics and training recommendations
- Annotate segmentation polygons online
- Queue model comparison jobs
- Run continuous optimization across model/image-size combinations
- Review training previews and metrics
- Re-open the legacy inference gallery from the same deployment

## Stack

- FastAPI
- Uvicorn
- Ultralytics YOLO
- Vanilla HTML/CSS/JS frontend

## Project Layout

- `app/`: API, auth, dataset management, job worker, optimizer, static UI
- `config/`: seeded user config
- `scripts/`: training runner, deployment scripts, smoke test, tunnel scripts
- `requirements.txt`: Python dependencies

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

## Default Users

- `admin / admin1234`
- `annotator / annotator1234`
- `viewer / viewer1234`

Change these before production use.
