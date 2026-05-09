from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.tracker import DEFAULT_MODEL, SUPPORTED_VIDEO_EXTENSIONS, TrackingOptions, process_video


APP_ROOT = Path(__file__).resolve().parent
JOBS_DIR = Path(os.getenv("JOBS_DIR", APP_ROOT / "jobs"))
STATIC_DIR = APP_ROOT / "static"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Crowd Tracking", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

JOBS: dict[str, dict[str, Any]] = {}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "crowd-tracking"}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    jobs = sorted(JOBS.values(), key=lambda item: item["created_at"], reverse=True)
    return {"jobs": jobs[:20]}


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL),
    confidence: float = Form(0.35),
    max_frames: int = Form(0),
) -> dict[str, Any]:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported format. Supported: {supported}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{suffix}"
    with input_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    job = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": video.filename,
        "progress": "waiting for worker",
    }
    JOBS[job_id] = job

    options = TrackingOptions(model_name=model_name, confidence=confidence, max_frames=max_frames)
    background_tasks.add_task(_run_job, job_id, input_path, job_dir, options)
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = _job_or_404(job_id)
    if job["status"] == "done":
        job = {**job, "links": _result_links(job_id)}
    return job


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_job_file(job_id: str, filename: str) -> FileResponse:
    _job_or_404(job_id)
    allowed = {"tracked_output.mp4", "trajectory_map.png", "trajectories.json", "trajectories.csv", "summary.json"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = JOBS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not ready")
    return FileResponse(path)


def _run_job(job_id: str, input_path: Path, job_dir: Path, options: TrackingOptions) -> None:
    JOBS[job_id].update({"status": "running", "progress": "processing video"})
    try:
        summary = process_video(input_path, job_dir, options)
        JOBS[job_id].update(
            {
                "status": "done",
                "progress": "finished",
                "summary": summary,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        JOBS[job_id].update(
            {
                "status": "failed",
                "progress": "failed",
                "error": str(exc),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        with (job_dir / "error.json").open("w", encoding="utf-8") as f:
            json.dump({"error": str(exc)}, f, indent=2)


def _job_or_404(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        summary_path = JOBS_DIR / job_id / "summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as f:
                summary = json.load(f)
            job = {"id": job_id, "status": "done", "summary": summary, "created_at": summary_path.stat().st_mtime}
            JOBS[job_id] = job
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _result_links(job_id: str) -> dict[str, str]:
    base = f"/api/jobs/{job_id}/files"
    return {
        "video": f"{base}/tracked_output.mp4",
        "trajectory_map": f"{base}/trajectory_map.png",
        "json": f"{base}/trajectories.json",
        "csv": f"{base}/trajectories.csv",
        "summary": f"{base}/summary.json",
    }

