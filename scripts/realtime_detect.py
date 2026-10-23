"""Real-time lettuce health detection from a webcam, video file, or image folder.

Usage:
  .venv\\Scripts\\python.exe scripts\\realtime_detect.py
  .venv\\Scripts\\python.exe scripts\\realtime_detect.py --source 0
  .venv\\Scripts\\python.exe scripts\\realtime_detect.py --source path\\to\\video.mp4
  .venv\\Scripts\\python.exe scripts\\realtime_detect.py --source path\\to\\image_or_folder --save

Press 'q' to quit the live window.
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "lettuce_health_v1" / "weights" / "best.pt"

COLORS = {
    "healthy": (0, 200, 0),
    "not healthy": (0, 0, 255),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained .pt weights")
    p.add_argument("--source", default="0", help="Webcam index, video path, image path, or folder")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default=None, help="'cpu', '0' for first GPU, etc. Autodetects if omitted")
    p.add_argument("--save", action="store_true", help="Save annotated output video/image to runs/realtime/")
    return p.parse_args()


def draw_detections(frame, result):
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS.get(label, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def run(args):
    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = None
    if args.save:
        out_dir = ROOT / "runs" / "realtime"
        out_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_dir / "output.mp4"), fourcc, fps, (w, h))
        print(f"Saving annotated output to {out_dir / 'output.mp4'}")

    prev_time = time.time()
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream / cannot read frame.")
            break

        results = model.predict(
            frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
        )
        result = results[0]
        frame = draw_detections(frame, result)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Lettuce Health Detection", frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())
