from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out_dir = Path("sample_data")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "synthetic_crowd.mp4"
    width, height, fps = 640, 360, 24
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    asset = _load_people_asset(width, height)
    if asset is not None:
        for frame_idx in range(96):
            shift = int(np.sin(frame_idx / 11) * 18)
            matrix = np.float32([[1, 0, shift], [0, 1, 0]])
            frame = cv2.warpAffine(asset, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
            writer.write(frame)
        writer.release()
        print(path)
        return

    for frame_idx in range(144):
        frame = np.full((height, width, 3), 238, dtype=np.uint8)
        cv2.line(frame, (40, 310), (600, 310), (160, 172, 184), 2)

        people = [
            (80 + frame_idx * 2, 120 + int(np.sin(frame_idx / 14) * 18), (40, 100, 220)),
            (520 - frame_idx * 2, 150 + int(np.cos(frame_idx / 16) * 22), (30, 150, 90)),
        ]
        for x, y, color in people:
            cv2.circle(frame, (x, y), 14, color, -1)
            cv2.rectangle(frame, (x - 14, y + 14), (x + 14, y + 76), color, -1)
            cv2.line(frame, (x - 12, y + 76), (x - 22, y + 112), color, 7)
            cv2.line(frame, (x + 12, y + 76), (x + 22, y + 112), color, 7)
        writer.write(frame)

    writer.release()
    print(path)


def _load_people_asset(width: int, height: int) -> np.ndarray | None:
    try:
        from ultralytics.utils import ASSETS
    except Exception:
        return None

    image_path = Path(ASSETS) / "bus.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
