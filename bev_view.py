#!/usr/bin/env python3
"""
Bird's-eye (top-down) view of CCTV traffic via planar homography (IPM).

Approach: the road is (near-)planar, so a single fixed camera admits an exact
plane-to-plane homography. We warp a MEDIAN BACKGROUND (vehicle-free road) as
the clean base map and render tracked vehicles as metric footprints at their
ground-contact points — nothing above the road plane is warped, so the output
never smears or hallucinates vehicles. This is the geometrically-faithful BEV;
see docs/research on why NeRF/3DGS/diffusion NVS are unsuitable for live
monitoring from one fixed viewpoint.

The road-plane quad comes from camera_lines.json ("<camera>_bev" keys):
  [[x,y]*4 image fractions: far-left, far-right, near-right, near-left]
plus assumed real dimensions (--width-m, --length-m). Positions are exact up
to the quad calibration; the metric scale is approximate until surveyed.

Example:
  python3 bev_view.py --video data/clips/chalong_c1_0700_demo30s.mp4 \
      --model models/phuket-yolo26s.pt --seconds 12 --out-dir bev-out
"""
import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

from track_analytics import vehicle_classes

BGR = {"car": (0, 0, 255), "motorcycle": (255, 140, 0), "truck": (0, 215, 255),
       "bus": (255, 0, 255), "bicycle": (0, 255, 0)}
FOOTPRINT_M = {"car": (1.8, 4.5), "motorcycle": (0.8, 2.0), "bicycle": (0.8, 2.0),
               "truck": (2.4, 7.0), "bus": (2.5, 11.0)}


def load_quad(stem: str) -> list[list[float]] | None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_lines.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        table = json.load(f)
    keys = [k[:-4] for k in table if k.endswith("_bev")]
    best = max((k for k in keys if stem.startswith(k)), key=len, default=None)
    return table[best + "_bev"] if best else None


def median_background(video: str, w: int, h: int, samples: int = 60) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames.append(cv2.resize(f, (w, h)))
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default="bev-out")
    ap.add_argument("--model", default="models/phuket-yolo26s.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--seconds", type=float, default=0, help="0 = whole video")
    ap.add_argument("--width-m", type=float, default=15, help="road-quad width (m)")
    ap.add_argument("--length-m", type=float, default=40, help="road-quad length (m)")
    ap.add_argument("--ppm", type=float, default=18, help="BEV pixels per metre")
    ap.add_argument("--quad", default="auto",
                    help="'auto' (camera_lines.json <camera>_bev) or 8 comma-separated "
                         "fractions: far-left x,y, far-right x,y, near-right x,y, near-left x,y")
    a = ap.parse_args()

    stem = os.path.splitext(os.path.basename(a.video))[0]
    quad = load_quad(stem) if a.quad == "auto" else \
        np.array([float(v) for v in a.quad.split(",")]).reshape(4, 2).tolist()
    if quad is None:
        raise SystemExit(f"no BEV quad for {stem}: add '<camera>_bev' to "
                         "camera_lines.json or pass --quad")

    W, Hh = 1280, 720
    src = np.float32(quad) * np.float32([W, Hh])
    bw, bh = int(a.width_m * a.ppm), int(a.length_m * a.ppm)
    dst = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    Hm = cv2.getPerspectiveTransform(src, dst)

    print("building median background (vehicle-free base map)…", flush=True)
    base = cv2.warpPerspective(median_background(a.video, W, Hh), Hm, (bw, bh))

    model = YOLO(a.model)
    classes = vehicle_classes(model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = int(a.seconds * fps) if a.seconds else None
    side_w = int(W * bh / Hh)
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{stem}_bev.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"),
                          fps, (side_w + bw + 20, bh))

    trails: dict[int, list] = {}
    fi = 0
    while True:
        if max_frames is not None and fi >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (W, Hh))
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        classes=list(classes), conf=a.conf, imgsz=a.imgsz,
                        verbose=False)[0]
        bev = base.copy()
        if r.boxes.id is not None:
            for tid, box, cls in zip(r.boxes.id.int().tolist(),
                                     r.boxes.xyxy.tolist(),
                                     r.boxes.cls.int().tolist()):
                x1, y1, x2, y2 = box
                name = classes[cls]
                p = cv2.perspectiveTransform(
                    np.float32([[[(x1 + x2) / 2, y2]]]), Hm)[0][0]
                c = BGR.get(name, (0, 255, 0))
                if -20 <= p[0] <= bw + 20 and -20 <= p[1] <= bh + 20:
                    fw, fl = FOOTPRINT_M.get(name, (1.8, 4.5))
                    cv2.rectangle(
                        bev, (int(p[0] - fw * a.ppm / 2), int(p[1] - fl * a.ppm)),
                        (int(p[0] + fw * a.ppm / 2), int(p[1])), c, -1)
                    trails.setdefault(tid, []).append((int(p[0]), int(p[1])))
                    pts = np.array(trails[tid][-30:], np.int32)
                    if len(pts) > 1:
                        cv2.polylines(bev, [pts], False, c, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.polylines(frame, [src.astype(np.int32)], True, (0, 255, 255), 2)
        canvas = np.zeros((bh, side_w + bw + 20, 3), np.uint8)
        canvas[:, :side_w] = cv2.resize(frame, (side_w, bh))
        canvas[:, side_w + 20:] = bev
        cv2.putText(canvas, "BEV: homography + tracked footprints (no hallucination)",
                    (side_w + 22, 18), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        out.write(canvas)
        fi += 1
        if fi % 100 == 0:
            print(f"  {fi} frames", flush=True)
    out.release()
    cap.release()
    print(f"BEV video: {out_path}")


if __name__ == "__main__":
    main()
