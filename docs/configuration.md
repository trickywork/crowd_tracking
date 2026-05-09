# Configuration

## Runtime

| Name | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8080` | Web server port. |
| `JOBS_DIR` | `app/jobs` locally, `/tmp/crowd-tracking-jobs` in Docker | Upload and artifact directory. |
| `YOLO_MODEL` | `yolo11n.pt` | Default person detector. |

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8090
```

Open `http://localhost:8090`.

## Cloud Run

The Cloud Run profile uses CPU-only inference and keeps `min-instances=0` and `max-instances=1` for cost control. The first request may be slow because model weights are loaded into memory.

Recommended service settings:

```text
memory: 4Gi
cpu: 2
timeout: 900s
concurrency: 1
min instances: 0
max instances: 1
```

