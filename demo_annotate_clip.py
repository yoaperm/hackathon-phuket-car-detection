#!/usr/bin/env python3
"""
Hackathon demo: annotate a Phuket CCTV clip with COCO YOLOv8 vehicle detections.

Reads a local clip (downloaded from S3), samples frames at --fps, detects
vehicles (car/motorcycle/bus/truck/bicycle) with a pretrained COCO YOLOv8
model, draws boxes + a per-class count HUD, and writes an annotated MP4 and
a per-frame counts CSV.

Example:
  python3 demo_annotate_clip.py --video clip.mp4 --out-dir demo-output \
      --model yolov8s.pt --fps 5 --scale 1280 --conf 0.35
"""
import argparse
import csv
import os
from collections import Counter

import cv2
from ultralytics import YOLO

# COCO class ids for traffic-relevant objects
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {"car": (0, 220, 0), "motorcycle": (255, 140, 0), "bus": (0, 165, 255),
          "truck": (0, 0, 230), "bicycle": (230, 0, 230)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default="demo-output")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--fps", type=float, default=5.0, help="sampled output fps")
    ap.add_argument("--scale", type=int, default=1280, help="output frame width")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / a.fps))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = a.scale
    out_h = int(h * out_w / w) // 2 * 2

    model = YOLO(a.model)
    out_mp4 = os.path.join(a.out_dir, f"{stem}_annotated.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (out_w, out_h))

    rows, agg = [], Counter()
    idx = out_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step:
            idx += 1
            continue
        small = cv2.resize(frame, (out_w, out_h))
        r = model.predict(small, imgsz=a.imgsz, conf=a.conf,
                          classes=list(VEHICLE_CLASSES), verbose=False)[0]
        counts = Counter()
        for box, cls, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = VEHICLE_CLASSES[int(cls)]
            counts[name] += 1
            x1, y1, x2, y2 = (int(v) for v in box)
            c = COLORS[name]
            cv2.rectangle(small, (x1, y1), (x2, y2), c, 2)
            label = f"{name} {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(small, (x1, y1 - th - 4), (x1 + tw, y1), c, -1)
            cv2.putText(small, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        agg.update(counts)
        hud = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no detections"
        cv2.rectangle(small, (0, 0), (out_w, 26), (0, 0, 0), -1)
        cv2.putText(small, f"t={idx / src_fps:5.1f}s  {hud}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(small)
        rows.append({"frame_index": idx, "timestamp_sec": round(idx / src_fps, 2),
                     "total": sum(counts.values()),
                     **{k: counts.get(k, 0) for k in VEHICLE_CLASSES.values()}})
        out_idx += 1
        if out_idx % 25 == 0:
            print(f"  {out_idx} frames annotated", flush=True)
        idx += 1

    cap.release()
    writer.release()

    csv_path = os.path.join(a.out_dir, f"{stem}_counts.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"annotated video : {out_mp4}")
    print(f"per-frame counts: {csv_path}")
    print(f"aggregate detections over {out_idx} sampled frames: {dict(agg)}")


if __name__ == "__main__":
    main()
