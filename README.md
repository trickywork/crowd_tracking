# Crowd Tracking

Crowd Tracking is a portfolio demo for person tracking in short video clips. It accepts common video formats, detects people with a YOLO model, tracks identities with DeepSORT, and produces a side-by-side output video: annotated frames on the left and a trajectory mini-map on the right.

## Features

- Upload `.mp4`, `.mov`, `.avi`, `.mkv`, or `.webm` clips.
- Detect people frame-by-frame with a configurable YOLO model.
- Track consistent person IDs with DeepSORT.
- Export a tracked MP4, final trajectory map, CSV, JSON, and summary.
- Run locally or deploy to Google Cloud Run.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8090
```

Open:

```text
http://localhost:8090
```

Optional synthetic video:

```bash
python scripts/create_demo_video.py
```

## API

```text
GET  /api/health
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/tracked_output.mp4
GET  /api/jobs/{job_id}/files/trajectory_map.png
GET  /api/jobs/{job_id}/files/trajectories.csv
GET  /api/jobs/{job_id}/files/trajectories.json
```

## Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --project caramel-vim-441513-e1
```

The Cloud Run service is configured for low-cost portfolio use: zero minimum instances, one maximum instance, and CPU-only inference.

