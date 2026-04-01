# ChromoLab Ops Portal

ChromoLab is an AI operations portal for research teams.

It uses chromosome imagery as the working example, but the real goal is broader: keep annotation, training control, validation, model comparison, revision tracking, and the next research question inside one system that can keep evolving over time.

中文說明請見 [README.zh-TW.md](README.zh-TW.md).

## What This Project Does

- Separates responsibilities across `admin`, `annotator`, and `viewer`
- Accepts YOLO-format dataset ZIP uploads and inspects dataset readiness
- Lets annotators edit segmentation polygons directly in the browser
- Lets admins queue model comparisons, tune training settings, and track revisions
- Runs continuous optimization as new annotations arrive
- Stores metrics, previews, and logs so results can be reviewed instead of guessed
- Surfaces research-direction prompts based on coverage gaps, class balance, revisions, and completed jobs

## Product Positioning

This is not meant to be a one-off demo or a single-model screen.

ChromoLab is closer to an AI research control plane:

1. New image batches come in and are organized into managed datasets
2. Annotators push the dataset forward through new revisions
3. Admins control comparisons, optimization loops, and training jobs
4. The platform records metrics, previews, logs, and dataset state over time
5. The team uses those artifacts to decide what to label next, what to test next, and what research direction should follow

## Core Capabilities

- Role-based operations portal
- YOLO ZIP dataset upload
- Browser-based annotation
- Model comparison scheduling
- Continuous optimizer
- Training preview and log review
- Legacy inference gallery mounted in the same deployment
- Research-direction guidance cards

## Stack

- FastAPI
- Uvicorn
- Ultralytics YOLO
- Vanilla HTML / CSS / JS

## Project Layout

- `app/`: API, auth, dataset management, job worker, optimizer, and static UI
- `config/`: seeded config and default users
- `scripts/`: training runner, smoke test, deployment, and tunnel lifecycle scripts
- `requirements.txt`: Python dependencies

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

After startup:

- `/` serves the English introduction page
- `/zh-TW` serves the Traditional Chinese introduction page
- `/portal` serves the admin portal

## Default Accounts

- `admin / admin1234`
- `annotator / annotator1234`
- `viewer / viewer1234`

Change these before production use.
