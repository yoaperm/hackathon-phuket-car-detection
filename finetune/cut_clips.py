#!/usr/bin/env python3
"""
Cut short demo/training clips from the huge S3 CCTV recordings without
downloading them: byte-range fetch the file head (the Hikvision IMKH
"mp4" is an MPEG-PS stream, so the head decodes standalone), then re-wrap
the first N seconds with OpenCV. No ffmpeg needed.

Clip spec format (one per line, in a file or inline via --spec):
  <location>/<timeband>/<filename>  ->  <out_stem>

Example:
  python3 finetune/cut_clips.py --out-dir data/clips --seconds 30 \
      --spec "chalong/0700-0900/05_Chalong_C10-CHL-SB-02.mp4 -> chalong_c10_0700_demo30s"
"""
import argparse
import os
import subprocess
import tempfile

import cv2

BUCKET = "chula-aigov-car-video-training-487984284636"
PREFIX = "01_traffic/phuket-eye/location-"
HEAD_BYTES_PER_SEC = 1_500_000  # ~2x the observed ~750 KB/s stream rate


def cut(src_key: str, out_path: str, seconds: float) -> str:
    head_bytes = int(seconds * HEAD_BYTES_PER_SEC) + 20_000_000
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        head = tf.name
    try:
        subprocess.run(
            ["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", src_key,
             "--range", f"bytes=0-{head_bytes}", head],
            check=True, stdout=subprocess.DEVNULL)
        cap = cv2.VideoCapture(head)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(3)), int(cap.get(4))
        want = int(fps * seconds)
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        written = 0
        while written < want:
            ret, frame = cap.read()
            if not ret:
                break
            vw.write(frame)
            written += 1
        cap.release()
        vw.release()
        if written < want * 0.9:
            raise RuntimeError(f"only decoded {written}/{want} frames from head")
        return f"{w}x{h}@{fps:.0f} frames={written}"
    finally:
        os.unlink(head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/clips")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--spec", action="append", default=[],
                    help="'<loc>/<timeband>/<file> -> <out_stem>' (repeatable)")
    ap.add_argument("--spec-file", help="file with one spec per line")
    args = ap.parse_args()

    specs = list(args.spec)
    if args.spec_file:
        with open(args.spec_file) as f:
            specs += [l.strip() for l in f if l.strip() and not l.startswith("#")]

    os.makedirs(args.out_dir, exist_ok=True)
    for spec in specs:
        src, stem = (s.strip() for s in spec.split("->"))
        out = os.path.join(args.out_dir, stem + ".mp4")
        if os.path.exists(out):
            print(f"{stem}: exists, skipping", flush=True)
            continue
        info = cut(PREFIX + src, out, args.seconds)
        print(f"{stem}: {info}", flush=True)


if __name__ == "__main__":
    main()
