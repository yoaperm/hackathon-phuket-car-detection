#!/usr/bin/env python3
"""
Build a YOLO-format fine-tuning dataset from Phuket CCTV clips by auto-labeling
frames with a large COCO-pretrained teacher model.

Pipeline: sample frames from each clip at --fps, run the teacher (default
yolo11l.pt) at high resolution, keep vehicle detections above --conf, and write
images/ + labels/ + dataset.yaml ready for `train_phuket.py`. The train/val
split is BY SOURCE VIDEO (not random frames) so near-duplicate frames from the
same clip never leak across the split.

Auto-labels are noisy ground truth: the student can only approach the teacher's
quality. Use a teacher at least two sizes larger than the student (l/x teacher
for an s student), and treat a human-reviewed pass (CVAT/Label Studio) as the
upgrade path for boxes the teacher misses (rain, night, heavy occlusion).

Example:
  python3 finetune/build_dataset.py --clips-dir data/clips --out data/phuket-yolo \
      --teacher yolo11l.pt --fps 1 --conf 0.4 --val-stems chalong_c1_2200_demo20s
"""
import argparse
import glob
import os
import sys

import cv2
import yaml
from ultralytics import YOLO

# COCO class id -> (dataset class id, name). Vehicle classes only.
CLASS_MAP = {2: (0, "car"), 3: (1, "motorcycle"), 5: (2, "bus"),
             7: (3, "truck"), 1: (4, "bicycle")}
NAMES = [name for _, name in sorted(CLASS_MAP.values())]


def sample_frames(video_path: str, fps: float):
    """Yield (frame_index, frame) sampled at roughly `fps` frames per second."""
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(native / fps)), 1)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield idx, frame
        idx += 1
    cap.release()


def label_frame(model, frame, conf: float, imgsz: int):
    """Run the teacher and return YOLO label lines for vehicle classes."""
    result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    h, w = frame.shape[:2]
    lines = []
    for box in result.boxes:
        coco_id = int(box.cls)
        if coco_id not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[coco_id][0]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True, help="directory of source .mp4 clips")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--teacher", default="yolo11l.pt",
                    help="teacher model (bare ultralytics names auto-download)")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second")
    ap.add_argument("--conf", type=float, default=0.4, help="teacher confidence floor")
    ap.add_argument("--imgsz", type=int, default=1920, help="teacher inference size")
    ap.add_argument("--save-width", type=int, default=1920,
                    help="width images are saved at (0 = native)")
    ap.add_argument("--val-stems", default="",
                    help="comma-separated clip stems held out for validation; "
                         "default holds out the last clip alphabetically")
    ap.add_argument("--min-boxes", type=int, default=1,
                    help="skip frames with fewer teacher detections than this")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(args.clips_dir, "*.mp4")))
    if not clips:
        sys.exit(f"no .mp4 clips found in {args.clips_dir}")
    val_stems = ({s.strip() for s in args.val_stems.split(",") if s.strip()}
                 or {os.path.splitext(os.path.basename(clips[-1]))[0]})

    model = YOLO(args.teacher)
    counts = {"train": 0, "val": 0}
    for split in counts:
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    for clip in clips:
        stem = os.path.splitext(os.path.basename(clip))[0]
        split = "val" if stem in val_stems else "train"
        kept = skipped = 0
        for idx, frame in sample_frames(clip, args.fps):
            lines = label_frame(model, frame, args.conf, args.imgsz)
            if len(lines) < args.min_boxes:
                skipped += 1
                continue
            if args.save_width and frame.shape[1] > args.save_width:
                h = int(frame.shape[0] * args.save_width / frame.shape[1])
                frame = cv2.resize(frame, (args.save_width, h))
            name = f"{stem}_f{idx:06d}"
            cv2.imwrite(os.path.join(args.out, "images", split, name + ".jpg"),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            kept += 1
            counts[split] += 1
        print(f"{stem}: {kept} frames -> {split} ({skipped} skipped, <{args.min_boxes} boxes)",
              flush=True)

    with open(os.path.join(args.out, "dataset.yaml"), "w") as f:
        yaml.safe_dump({"path": os.path.abspath(args.out),
                        "train": "images/train", "val": "images/val",
                        "names": dict(enumerate(NAMES))}, f, sort_keys=False)
    print(f"dataset ready: {counts['train']} train / {counts['val']} val -> {args.out}/dataset.yaml")


if __name__ == "__main__":
    main()
