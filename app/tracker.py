from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
FALLBACK_MODEL = "yolov8n.pt"


@dataclass
class TrackingOptions:
    model_name: str = DEFAULT_MODEL
    confidence: float = 0.35
    iou: float = 0.45
    image_size: int = 640
    max_frames: int = 0
    max_age: int = 30
    n_init: int = 3


def validate_video_path(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video format. Supported formats: {supported}")


def process_video(input_path: Path, output_dir: Path, options: TrackingOptions) -> dict[str, Any]:
    validate_video_path(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    model = _load_model(options.model_name)
    tracker = DeepSort(max_age=options.max_age, n_init=options.n_init, nms_max_overlap=1.0)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("Video has invalid frame dimensions.")

    panel_width = max(320, min(420, frame_width // 2))
    output_size = (frame_width + panel_width, frame_height)
    output_video = output_dir / "tracked_output.mp4"
    writer = _open_writer(output_video, source_fps, output_size)

    trajectories: dict[int, list[dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    frame_index = 0
    processed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if options.max_frames and processed_frames >= options.max_frames:
            break

        detections = _detect_people(model, frame, options)
        tracks = tracker.update_tracks(detections, frame=frame)
        annotated = frame.copy()

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            foot_x = (x1 + x2) / 2.0
            foot_y = float(y2)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            point = {
                "frame": float(frame_index),
                "time": float(frame_index / source_fps),
                "x": float(foot_x),
                "y": float(foot_y),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
            trajectories.setdefault(track_id, []).append(point)
            rows.append({"track_id": track_id, **point})
            _draw_track(annotated, track_id, (x1, y1, x2, y2), trajectories[track_id])

        panel = _draw_trajectory_panel(frame_width, frame_height, panel_width, trajectories)
        combined = np.hstack([annotated, panel])
        writer.write(combined)

        frame_index += 1
        processed_frames += 1

    cap.release()
    writer.release()

    trajectory_json = output_dir / "trajectories.json"
    trajectory_csv = output_dir / "trajectories.csv"
    trajectory_map = output_dir / "trajectory_map.png"

    with trajectory_json.open("w", encoding="utf-8") as f:
        json.dump(_serialize_trajectories(trajectories), f, indent=2)

    _write_csv(trajectory_csv, rows)
    final_panel = _draw_trajectory_panel(frame_width, frame_height, max(900, frame_width), trajectories, final=True)
    cv2.imwrite(str(trajectory_map), final_panel)

    summary = {
        "model": getattr(model, "model_name", options.model_name),
        "input_filename": input_path.name,
        "source_fps": source_fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "processed_frames": processed_frames,
        "tracked_people": len(trajectories),
        "duration_seconds": processed_frames / source_fps if source_fps else 0,
        "processing_seconds": round(time.time() - started, 2),
        "outputs": {
            "video": output_video.name,
            "trajectory_map": trajectory_map.name,
            "trajectories_json": trajectory_json.name,
            "trajectories_csv": trajectory_csv.name,
        },
        "options": asdict(options),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _load_model(model_name: str) -> YOLO:
    try:
        model = YOLO(model_name)
        model.model_name = model_name
        return model
    except Exception:
        if model_name == FALLBACK_MODEL:
            raise
        model = YOLO(FALLBACK_MODEL)
        model.model_name = FALLBACK_MODEL
        return model


def _detect_people(model: YOLO, frame: np.ndarray, options: TrackingOptions) -> list[tuple[list[float], float, str]]:
    result = model.predict(
        frame,
        classes=[0],
        conf=options.confidence,
        iou=options.iou,
        imgsz=options.image_size,
        verbose=False,
    )[0]
    detections: list[tuple[list[float], float, str]] = []
    if result.boxes is None:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, score in zip(xyxy, confs):
        x1, y1, x2, y2 = [float(v) for v in box]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width < 4 or height < 8:
            continue
        detections.append(([x1, y1, width, height], float(score), "person"))
    return detections


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for fourcc_name in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError("Could not create an MP4 video writer.")


def _draw_track(frame: np.ndarray, track_id: int, box: tuple[int, int, int, int], history: list[dict[str, float]]) -> None:
    color = _track_color(track_id)
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 96, y1), color, -1)
    cv2.putText(frame, f"ID {track_id}", (x1 + 8, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    pts = [(int(p["x"]), int(p["y"])) for p in history[-60:]]
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, color, 2)
    if pts:
        cv2.circle(frame, pts[-1], 4, color, -1)


def _draw_trajectory_panel(
    frame_width: int,
    frame_height: int,
    panel_width: int,
    trajectories: dict[int, list[dict[str, float]]],
    final: bool = False,
) -> np.ndarray:
    panel = np.full((frame_height, panel_width, 3), 246, dtype=np.uint8)
    margin_x = 24
    margin_y = 44
    plot_w = max(1, panel_width - margin_x * 2)
    plot_h = max(1, frame_height - margin_y * 2)

    cv2.rectangle(panel, (margin_x, margin_y), (margin_x + plot_w, margin_y + plot_h), (210, 214, 220), 1)
    cv2.putText(panel, "Trajectory Map", (margin_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (31, 41, 55), 2)
    cv2.putText(panel, "x / y position", (margin_x, frame_height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1)

    for track_id, points in sorted(trajectories.items()):
        if len(points) == 0:
            continue
        color = _track_color(track_id)
        projected = [_project_point(p["x"], p["y"], frame_width, frame_height, margin_x, margin_y, plot_w, plot_h) for p in points]
        for a, b in zip(projected, projected[1:]):
            cv2.line(panel, a, b, color, 2 if final else 1)
        cv2.circle(panel, projected[-1], 5, color, -1)
        cv2.putText(panel, str(track_id), (projected[-1][0] + 6, projected[-1][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if not trajectories:
        cv2.putText(panel, "No confirmed tracks yet", (margin_x, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 116, 139), 1)
    return panel


def _project_point(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
    margin_x: int,
    margin_y: int,
    plot_w: int,
    plot_h: int,
) -> tuple[int, int]:
    px = margin_x + int(np.clip(x / max(1, frame_width), 0, 1) * plot_w)
    py = margin_y + int(np.clip(y / max(1, frame_height), 0, 1) * plot_h)
    return px, py


def _track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47) % 180
    color = np.uint8([[[hue, 210, 230]]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _serialize_trajectories(trajectories: dict[int, list[dict[str, float]]]) -> dict[str, Any]:
    return {
        "tracks": [
            {
                "track_id": track_id,
                "points": points,
                "distance_pixels": _distance(points),
            }
            for track_id, points in sorted(trajectories.items())
        ]
    }


def _distance(points: list[dict[str, float]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    return round(total, 2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["track_id", "frame", "time", "x", "y", "center_x", "center_y", "x1", "y1", "x2", "y2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

