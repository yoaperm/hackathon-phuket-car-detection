#!/usr/bin/env python3
"""
Fine-tune a YOLO11 detector on the Phuket CCTV dataset produced by
build_dataset.py. GPU-aware: uses CUDA on RunPod / EC2, falls back to CPU for
a local smoke test (--smoke shrinks everything so the loop finishes in minutes).

Freezing the backbone (--freeze 10) is the safe default for small auto-labeled
datasets: it adapts the detection head to the CCTV viewpoint without letting a
few thousand noisy labels wash out COCO features.

Examples:
  # local CPU smoke test (~minutes)
  python3 finetune/train_phuket.py --data data/phuket-yolo/dataset.yaml --smoke

  # RunPod / GPU real run, upload best.pt to S3
  python3 finetune/train_phuket.py --data data/phuket-yolo/dataset.yaml \
      --model yolo11s.pt --epochs 40 --imgsz 1280 --batch 8 \
      --out-s3 s3://chula-aigov-car-video-training-487984284636/models/phuket-yolo11s/
"""
import argparse
import subprocess

import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.yaml from build_dataset.py")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--freeze", type=int, default=10,
                    help="backbone layers to freeze (0 = full fine-tune)")
    ap.add_argument("--name", default="phuket-finetune")
    ap.add_argument("--out-s3", default="", help="S3 prefix to upload best.pt after training")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU-friendly run: yolo11n, 1 epoch, imgsz 640")
    args = ap.parse_args()

    if args.smoke:
        args.model, args.epochs, args.imgsz, args.batch = "yolo11n.pt", 1, 640, 2

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} epochs={args.epochs} imgsz={args.imgsz}",
          flush=True)

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, freeze=args.freeze or None,
                          device=device, name=args.name, patience=10, plots=True)

    best = str(results.save_dir / "weights" / "best.pt")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device)
    print(f"best={best}  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")

    if args.out_s3:
        dest = args.out_s3.rstrip("/") + "/best.pt"
        subprocess.run(["aws", "s3", "cp", best, dest, "--only-show-errors"], check=True)
        print(f"uploaded {dest}")


if __name__ == "__main__":
    main()
