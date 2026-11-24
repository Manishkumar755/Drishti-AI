"""Phase 3 demo: MOG2 background subtraction on a synthetic video.

A blue box moves left to right over a static road-like background.
The script writes an annotated video and a sample frame, then prints a summary.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from cv_engine.detection import classical as c  # noqa: E402

SAMPLE_VIDEO = ROOT / "data" / "samples" / "motion.mp4"
OUTPUT_VIDEO = ROOT / "results" / "motion_demo.mp4"
SNAPSHOT = ROOT / "results" / "motion_frame.png"
W, H, FPS, FRAMES = 320, 240, 25, 120
WARMUP = 15  # MOG2 needs a few frames to learn the background


def make_synthetic_video(path: Path) -> None:
    """Create a static background with one moving blue box, if the file does not exist yet."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("OpenCV could not open the video writer (mp4v codec).")
    background = np.full((H, W, 3), 90, dtype=np.uint8)
    cv2.rectangle(background, (0, 150), (W, H), (70, 70, 70), -1)  # road
    rng = np.random.default_rng(0)
    for i in range(FRAMES):
        frame = background.copy()
        x = int(-60 + i * 3)  # moves from left edge to the right
        cv2.rectangle(frame, (x, 170), (x + 60, 200), (200, 80, 40), -1)
        noise = rng.normal(0, 3, frame.shape)
        writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    writer.release()


def main() -> None:
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_video(SAMPLE_VIDEO)

    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {SAMPLE_VIDEO}")

    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    subtractor = c.make_background_subtractor()

    frame_index = 0
    frames_with_objects = 0
    counted_frames = 0
    total_boxes = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = c.clean_mask(c.motion_mask(subtractor, frame))
        boxes = c.find_objects(mask, min_area=300)

        annotated = frame.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"frame {frame_index} boxes {len(boxes)}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(annotated)

        if frame_index == FRAMES // 2:
            cv2.imwrite(str(SNAPSHOT), annotated)

        if frame_index >= WARMUP:
            counted_frames += 1
            total_boxes += len(boxes)
            if boxes:
                frames_with_objects += 1
        frame_index += 1

    cap.release()
    writer.release()

    print(f"Frames processed: {frame_index}")
    print(f"Counted after warm-up: {counted_frames}")
    print(f"Frames with at least one moving object: {frames_with_objects}")
    if counted_frames:
        print(f"Average boxes per counted frame: {total_boxes / counted_frames:.2f}")
    print(f"Annotated video: {OUTPUT_VIDEO.relative_to(ROOT)}")
    print(f"Sample frame:    {SNAPSHOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()