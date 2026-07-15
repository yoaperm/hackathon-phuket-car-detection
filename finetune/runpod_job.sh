#!/usr/bin/env bash
# Unattended fine-tune job for a RunPod PyTorch/CUDA pod.
#
# Launched as the pod's start command via:
#   pip -q install awscli && aws s3 cp s3://<bucket>/code/runpod_job.sh - | bash
# Expects AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the environment
# (the runpod-s3-reader scoped key: read-all, write to models/ only).
#
# Trains YOLO11s on the staged phuket-yolo-v1 dataset, uploads best.pt +
# training curves to s3://<bucket>/models/phuket-yolo11s/, then writes a
# _DONE (or _FAILED) marker the launcher polls for. The launcher terminates
# the pod when the marker appears, so a hung job can't bill forever.
set -uo pipefail

BUCKET=chula-aigov-car-video-training-487984284636
DATASET=s3://$BUCKET/datasets/phuket-yolo-v1.tar.gz
OUT=s3://$BUCKET/models/phuket-yolo11s
export AWS_DEFAULT_REGION=ap-southeast-1

fail() {
  echo "JOB FAILED: $1"
  echo "$1" | aws s3 cp - "$OUT/_FAILED" || true
  exit 1
}

cd /workspace 2>/dev/null || cd /root
echo "== env: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'NO GPU')"
python3 -c "import torch; assert torch.cuda.is_available()" || fail "CUDA not available in this pod"

pip -q install "ultralytics>=8.1,<9" || fail "pip install ultralytics"

echo "== fetching dataset"
aws s3 cp "$DATASET" - | tar xz --no-same-owner --exclude '._*' || fail "dataset download"
[ -f phuket-yolo/dataset.yaml ] || fail "dataset.yaml missing after extract"
# dataset.yaml carries the absolute path of the machine that built it — repoint it here
sed -i "s|^path:.*|path: $(pwd)/phuket-yolo|" phuket-yolo/dataset.yaml

echo "== training"
python3 - <<'EOF' || fail "training crashed"
import shutil
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(data="phuket-yolo/dataset.yaml", epochs=40, imgsz=1280,
                      batch=8, freeze=10, device=0, name="phuket-yolo11s",
                      patience=10, plots=True, exist_ok=True)
metrics = model.val(data="phuket-yolo/dataset.yaml", imgsz=1280, device=0)
print(f"FINAL mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")
shutil.copy(str(results.save_dir / "weights" / "best.pt"), "best.pt")
shutil.copy(str(results.save_dir / "results.csv"), "results.csv")
with open("metrics.txt", "w") as f:
    f.write(f"mAP50={metrics.box.map50:.4f}\nmAP50-95={metrics.box.map:.4f}\n")
    for i, name in metrics.names.items():
        f.write(f"{name}: mAP50-95={metrics.box.maps[i]:.4f}\n")
EOF

echo "== uploading results"
aws s3 cp best.pt "$OUT/best.pt" || fail "upload best.pt"
aws s3 cp results.csv "$OUT/results.csv" || true
aws s3 cp metrics.txt "$OUT/metrics.txt" || true
aws s3 cp runs/detect/phuket-yolo11s/confusion_matrix_normalized.png "$OUT/confusion_matrix.png" || true
aws s3 cp runs/detect/phuket-yolo11s/results.png "$OUT/curves.png" || true

cat metrics.txt | aws s3 cp - "$OUT/_DONE" || fail "write _DONE"
echo "== JOB COMPLETE"
