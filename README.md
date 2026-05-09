# Crowd Tracking

[![CI](https://github.com/trickywork/crowd_tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/trickywork/crowd_tracking/actions/workflows/ci.yml)

Crowd Tracking is a FastAPI web app for tracking people in short video clips. It accepts common video formats, detects people with YOLO, tracks identities with DeepSORT, and produces visual outputs that make a person's movement easier to inspect frame by frame.

## Live Demo

- Cloud Run service: `crowd-tracking`
- Cloud Run URL: `https://crowd-tracking-gb7rmueyna-uc.a.run.app`
- Planned portfolio URL: `https://crowdtracking.junliu.dev`
- Google Cloud project: `caramel-vim-441513-e1`
- Region: `us-central1`

The custom domain mapping has been created. Cloudflare still needs the DNS record below before the Google-managed certificate can finish provisioning:

```text
Type: CNAME
Name: crowdtracking
Target: ghs.googlehosted.com
```

## Tech Stack

- Python
- FastAPI
- Uvicorn
- Ultralytics YOLO
- DeepSORT realtime
- OpenCV
- NumPy
- HTML/CSS/JavaScript frontend served by FastAPI
- Docker, Artifact Registry, Google Cloud Build, Google Cloud Run
- API testing via local Postman workspace

## Project Structure

```text
crowd_tracking/
  app/
    main.py
    tracker.py
    static/
      index.html
      app.js
      styles.css
  docs/
    configuration.md
  scripts/
    create_demo_video.py
  data/
  requirements.txt
  Dockerfile
  cloudbuild.yaml
```

## Features

- Upload `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm` clips.
- Detect people frame by frame with a configurable YOLO model.
- Track stable identities with DeepSORT.
- Render an annotated output video.
- Render a trajectory mini-map next to the video.
- Export tracking results as MP4, PNG, CSV, JSON, and summary metadata.
- Run locally or on Cloud Run with CPU-only inference.

## Local Development

Create and activate a virtual environment:

```bash
cd crowd_tracking
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the app:

```bash
uvicorn app.main:app --reload --port 8090
```

Open:

```text
http://localhost:8090
```

Optional synthetic test video:

```bash
python scripts/create_demo_video.py
```

Expected result:

- The web page accepts a short video upload.
- Processing creates a job.
- The result page shows an annotated video and a trajectory map.
- CSV/JSON files can be downloaded for analysis.

## Environment Variables

Common local defaults:

```env
PORT=8090
MODEL_NAME=yolov8n.pt
TRACKING_STORAGE_DIR=data/jobs
MAX_UPLOAD_MB=200
```

Cloud Run can use the same defaults. The YOLO model is downloaded/cached during container use unless already present.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Web UI. |
| `GET` | `/api/health` | Health check. |
| `GET` | `/api/jobs` | List processed jobs. |
| `POST` | `/api/jobs` | Upload and process a video. |
| `GET` | `/api/jobs/{job_id}` | Job metadata and output links. |
| `GET` | `/api/jobs/{job_id}/files/tracked_output.mp4` | Annotated video. |
| `GET` | `/api/jobs/{job_id}/files/trajectory_map.png` | Trajectory map. |
| `GET` | `/api/jobs/{job_id}/files/trajectories.csv` | Tracking CSV. |
| `GET` | `/api/jobs/{job_id}/files/trajectories.json` | Tracking JSON. |

## Postman

Use the local Postman workspace collection:

```text
Crowd Tracking - Local API Tests
```

Suggested variables:

```text
baseUrl=http://localhost:8090
videoPath=/absolute/path/to/video.mp4
```

For Cloud Run:

```text
baseUrl=https://crowd-tracking-gb7rmueyna-uc.a.run.app
```

The exported backup copy is kept in a private local archive outside this public repo.

## Tests And Smoke Checks

Compile Python files:

```bash
python -m py_compile app/main.py app/tracker.py
```

Run a local API smoke check after starting Uvicorn:

```bash
curl http://localhost:8090/api/health
```

Generate a small demo video and upload it through the UI or Postman.

## Cloud Deployment

Manual deployment:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --project caramel-vim-441513-e1
```

Artifact Registry image path:

```text
us-central1-docker.pkg.dev/caramel-vim-441513-e1/portfolio-apps/crowd-tracking
```

Cloud Run cost controls:

- `min-instances=0`
- `max-instances=1`
- CPU-only inference
- no database
- no persistent disk

For heavier or longer videos, cost and runtime can increase because object detection runs per frame. Keep portfolio demo videos short.

## Expected Portfolio Behavior

A visitor should be able to upload a short video, wait for processing, and inspect both the annotated tracked video and the movement map. The output should clearly show detected person boxes, track IDs, and a small path visualization beside the frames.

## Additional Notes

Runtime and deployment setup notes are in:

```text
docs/configuration.md
```
