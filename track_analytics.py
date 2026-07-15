#!/usr/bin/env python3
"""
Vehicle detection + tracking analytics for CCTV traffic footage.

Runs a COCO-pretrained YOLO detector with ByteTrack multi-object tracking
(via ultralytics) over a traffic video and produces:

  <out-dir>/<name>_tracked.mp4      annotated video: boxes, track IDs, trails,
                                    counting line, live HUD
  <out-dir>/<name>_analytics.json   unique vehicles per class, directional flow
                                    counts, density time series, congestion states
  <out-dir>/<name>_frames.csv       per-frame: active tracks, cumulative counts,
                                    congestion state

Unlike infer.py (per-frame density counts), tracking gives each vehicle a
persistent identity, so "how many vehicles passed" and "which direction"
become answerable instead of just "how many are visible right now".

Example:
  python3 track_analytics.py --video clip.mp4 --out-dir analytics-out \
      --model yolo11s.pt --conf 0.35 --seconds 20 --line 0,0.55,1,0.55
"""
import argparse
import json
import os
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_NAMES = {"bicycle", "car", "motorcycle", "bus", "truck"}


def vehicle_classes(model):
    """Map class id -> name for the vehicle classes of THIS model.

    COCO checkpoints carry 80 classes (car=2, ...); the fine-tuned Phuket
    checkpoints carry 5 remapped ones (car=0, ...). Reading model.names keeps
    the id filter correct for both.
    """
    return {i: n for i, n in model.names.items() if n in VEHICLE_NAMES}
COLORS = {"car": (0, 220, 0), "motorcycle": (255, 140, 0), "bus": (0, 165, 255),
          "truck": (0, 0, 230), "bicycle": (230, 0, 230)}
# Congestion thresholds on the rolling mean of simultaneously-tracked vehicles.
CONGESTION_LEVELS = [(5, "Free"), (12, "Moderate"), (float("inf"), "Heavy")]
CONGESTION_COLORS = {"Free": (0, 200, 0), "Moderate": (0, 200, 255), "Heavy": (0, 0, 255)}


def side_of_line(pt, a, b):
    """Signed side of point pt relative to line a->b (0 = on the line)."""
    return np.sign((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]))


def congestion_state(mean_active: float) -> str:
    for threshold, name in CONGESTION_LEVELS:
        if mean_active <= threshold:
            return name
    return CONGESTION_LEVELS[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default="analytics-out")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--seconds", type=float, default=0, help="0 = whole video")
    ap.add_argument("--scale", type=int, default=1280, help="processing/output width")
    ap.add_argument("--line", default="0,0.55,1,0.55",
                    help="counting line as x1,y1,x2,y2 fractions of the frame")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="ultralytics tracker config (bytetrack.yaml or botsort.yaml)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = min(a.scale, w)
    out_h = int(h * out_w / w) // 2 * 2
    max_frames = int(a.seconds * src_fps) if a.seconds else None

    fx = [float(v) for v in a.line.split(",")]
    line_a = (int(fx[0] * out_w), int(fx[1] * out_h))
    line_b = (int(fx[2] * out_w), int(fx[3] * out_h))

    model = YOLO(a.model)
    classes = vehicle_classes(model)
    out_mp4 = os.path.join(a.out_dir, f"{stem}_tracked.mp4")
    # avc1 (H.264) plays in browsers; mp4v does not. Fall back if unavailable.
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"avc1"),
                             src_fps, (out_w, out_h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (out_w, out_h))

    trails = defaultdict(lambda: deque(maxlen=30))   # track id -> recent centers
    last_side = {}                                    # track id -> last line side
    counted = set()                                   # track ids already counted
    track_class = {}                                  # track id -> class name
    flow = Counter()                                  # "a_to_b" / "b_to_a"
    unique_by_class = Counter()
    seen_ids = set()
    active_window = deque(maxlen=int(5 * src_fps))    # rolling 5 s of active counts
    rows = []

    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (out_w, out_h))

        r = model.track(frame, persist=True, tracker=a.tracker,
                        classes=list(classes), conf=a.conf,
                        imgsz=a.imgsz, verbose=False)[0]

        active = 0
        if r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            clss = r.boxes.cls.int().tolist()
            confs = r.boxes.conf.tolist()
            boxes = r.boxes.xyxy.int().tolist()
            active = len(ids)
            for tid, cls, cf, (x1, y1, x2, y2) in zip(ids, clss, confs, boxes):
                name = classes[cls]
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    unique_by_class[name] += 1
                    track_class[tid] = name
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                trails[tid].append(center)

                side = side_of_line(center, line_a, line_b)
                prev = last_side.get(tid)
                if prev is not None and side != 0 and prev != 0 and side != prev \
                        and tid not in counted:
                    counted.add(tid)
                    flow["a_to_b" if prev < side else "b_to_a"] += 1
                if side != 0:
                    last_side[tid] = side

                c = COLORS[name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                label = f"#{tid} {name} {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), c, -1)
                cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 1, cv2.LINE_AA)
                pts = np.array(trails[tid], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, c, 1, cv2.LINE_AA)

        active_window.append(active)
        mean_active = sum(active_window) / len(active_window)
        state = congestion_state(mean_active)

        cv2.line(frame, line_a, line_b, (255, 255, 0), 2)
        hud1 = (f"active:{active}  unique:{len(seen_ids)}  "
                f"flow A>B:{flow['a_to_b']} B>A:{flow['b_to_a']}")
        hud2 = f"congestion: {state} (5s avg {mean_active:.1f})"
        cv2.rectangle(frame, (0, 0), (out_w, 46), (0, 0, 0), -1)
        cv2.putText(frame, hud1, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, hud2, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    CONGESTION_COLORS[state], 1, cv2.LINE_AA)
        writer.write(frame)

        rows.append({"frame_index": frame_idx,
                     "timestamp_sec": round(frame_idx / src_fps, 3),
                     "active_tracks": active,
                     "unique_total": len(seen_ids),
                     "flow_a_to_b": flow["a_to_b"],
                     "flow_b_to_a": flow["b_to_a"],
                     "congestion": state})
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames, {len(seen_ids)} unique vehicles", flush=True)

    cap.release()
    writer.release()

    flow_by_class = Counter(track_class[tid] for tid in counted)
    analytics = {
        "video": os.path.basename(a.video),
        "model": a.model,
        "tracker": a.tracker,
        "frames_processed": frame_idx,
        "duration_sec": round(frame_idx / src_fps, 2),
        "unique_vehicles_total": len(seen_ids),
        "unique_vehicles_by_class": dict(unique_by_class),
        "line_crossings": {"a_to_b": flow["a_to_b"], "b_to_a": flow["b_to_a"],
                           "by_class": dict(flow_by_class)},
        "congestion_final": rows[-1]["congestion"] if rows else "n/a",
        "congestion_share": {s: round(sum(r["congestion"] == s for r in rows)
                                      / max(len(rows), 1), 3)
                             for s in ("Free", "Moderate", "Heavy")},
    }
    json_path = os.path.join(a.out_dir, f"{stem}_analytics.json")
    with open(json_path, "w") as f:
        json.dump(analytics, f, indent=2)

    csv_path = os.path.join(a.out_dir, f"{stem}_frames.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        wr = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(json.dumps(analytics, indent=2))
    print(f"\ntracked video : {out_mp4}\nanalytics json: {json_path}\nframes csv    : {csv_path}")


if __name__ == "__main__":
    main()
