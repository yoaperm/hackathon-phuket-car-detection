# Running the traffic-density inference job on RunPod

A one-shot, headless batch job: download `.mp4` videos from S3, run YOLOv8 GPU
inference, write annotated videos + preview images + CSV summaries to
`/workspace/output`, then exit. No server, no GUI, no infinite loop.

## Prerequisites

- A **RunPod GPU Pod** launched from an official PyTorch/CUDA template
  (e.g. `runpod/pytorch:2.x-cuda...`). The job does **not** install torch — it
  relies on the base image's CUDA-matched PyTorch build.
- Your SSH key added to RunPod, and the Pod's SSH id from the dashboard.
- An S3 bucket containing the `.mp4` videos, plus AWS credentials (or an
  IAM role attached to the Pod).

## End-to-end command sequence

```bash
# ============ On your LOCAL machine ============
# 1. Get the repo (already includes models/best.pt + the deploy files).
git clone https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation.git yolov8-traffic-runpod
cd yolov8-traffic-runpod
#    (infer.py, run.sh, requirements.txt, .env.example live here alongside models/best.pt)

# 2. Sync the whole directory to the Pod's persistent /workspace volume.
#    Replace <pod-id> with your Pod's SSH id from the RunPod dashboard.
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" ./ <pod-id>@ssh.runpod.io:/workspace/yolov8-traffic-runpod/

# ============ On the POD ============
# 3. SSH in.
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/yolov8-traffic-runpod
chmod +x run.sh

# 4. Configure this run (see .env.example for all options).
#    NOTE: models/best.pt is fine-tuned on TOP-VIEW aerial imagery. For
#    street-view CCTV (e.g. the Phuket phuket-eye footage) use a COCO model
#    with full-frame detection and a vehicle class filter instead:
#      export MODEL_PATH=yolov8s.pt DISABLE_ROI=true \
#             DETECT_CLASSES=car,motorcycle,bus,truck,bicycle CONF_THRESHOLD=0.35
export S3_INPUT="s3://my-bucket/incoming-videos/"   # object(s) or a prefix ending in "/"
export OUTPUT_DIR="/workspace/output"
export AWS_ACCESS_KEY_ID="..."        # omit all three if the Pod has an IAM role
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
export ENABLE_GUI=false
export ALLOW_CPU=false

# 5. Run. tmux/nohup is recommended so the job survives an SSH drop.
nohup ./run.sh > run.log 2>&1 &
tail -f run.log

# 6. Inspect results on the Pod.
ls -R /workspace/output

# ============ Back on your LOCAL machine ============
# 7. Retrieve results.
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" <pod-id>@ssh.runpod.io:/workspace/output/ ./output-from-runpod/
```

### Quick local smoke test (no GPU, no S3)

```bash
LOCAL_VIDEO_DIR=. ALLOW_CPU=true ENABLE_GUI=false OUTPUT_DIR=./output python3 infer.py
```

Runs against the bundled `sample_video.mp4` and produces an annotated `.mp4`,
a preview `.jpg`, and per-frame + summary CSVs under `./output`.

## Outputs (under `OUTPUT_DIR`, default `/workspace/output`)

```
videos/<name>_annotated.mp4     annotated video (mp4v)
previews/<name>_preview.jpg     one annotated sample frame per video
csv/<name>_frame_counts.csv     per-frame: frame_index, timestamp_sec, left/right/total counts, statuses
csv/summary.csv                 one row per video: averages, maxima, % heavy, output paths, device_used
```

Optionally, set `S3_OUTPUT=s3://my-bucket/results/run1/` to also upload everything
under `OUTPUT_DIR` (videos/, previews/, csv/ — not the `_staging/` downloads) to
that S3 prefix after processing finishes. Off by default (local-only).

## Environment variables

See [`.env.example`](.env.example) for the full list with defaults. Key ones:
`S3_INPUT` / `LOCAL_VIDEO_DIR` (input), `S3_OUTPUT` (optional upload of results),
`OUTPUT_DIR`, `MODEL_PATH`, `CONF_THRESHOLD`, `HEAVY_TRAFFIC_THRESHOLD`,
`ENABLE_GUI`, `ALLOW_CPU`, `MAX_SECONDS`.

## Known limitations

1. **ROI/lane geometry** is a linear scale-up from `sample_video.mp4`'s framing
   (1280×720). It is correct for similarly-framed footage, not perspective-calibrated
   for arbitrary camera angles.
2. **S3 prefix listing is single-page** (≤1000 objects) — fine for a batch job.
3. **No retries/backoff.** A failed download or video is logged to stderr and
   skipped; the batch continues. The job exits non-zero only if *nothing* succeeded.
4. **torch is unpinned/uninstalled by design** — provided by the RunPod base image.
5. **`ENABLE_GUI=true`** only does anything with a real display attached (local dev);
   it is meaningless over a headless SSH session (and `opencv-python-headless` can't
   render a window anyway).

## Security note

If you shared a RunPod API key while setting this up, rotate it in the RunPod
dashboard — this job is SSH-only and never needs the API key.
