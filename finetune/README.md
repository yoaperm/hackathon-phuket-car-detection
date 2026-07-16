# Fine-tuning a Phuket-specific detector

Why: COCO-pretrained models work but leave accuracy on the table — published results
on developing-city angled CCTV show ~2.5× mAP improvement from in-domain training
(see `docs/research/2026-07-15-cctv-vehicle-detection-sota.md`). This pipeline
bootstraps an in-domain dataset with zero manual labeling by distilling a large
COCO teacher into auto-labels, then fine-tunes a small fast student.

## Pipeline

```
S3 CCTV clips ──> build_dataset.py ──> data/phuket-yolo/ ──> train_phuket.py ──> best.pt
                  (YOLO11l teacher       images/ labels/       (YOLO11s student,
                   auto-labels frames)   dataset.yaml           GPU on RunPod)
```

1. **Build the dataset** (CPU-friendly; ~90 frames from the 4 demo clips takes minutes):

   ```bash
   python3 finetune/build_dataset.py --clips-dir data/clips --out data/phuket-yolo \
       --teacher yolo11l.pt --fps 1 --conf 0.4
   ```

   Train/val is split **by source video** so near-duplicate frames never leak across
   the split. Default holds out the last clip alphabetically; pick explicitly with
   `--val-stems`.

2. **Train.** Local smoke test first, real run on GPU (RunPod / EC2 g4dn):

   ```bash
   # local: prove the loop (~5 min CPU)
   python3 finetune/train_phuket.py --data data/phuket-yolo/dataset.yaml --smoke

   # GPU: the real configuration
   python3 finetune/train_phuket.py --data data/phuket-yolo/dataset.yaml \
       --model yolo11s.pt --epochs 40 --imgsz 1280 --batch 8 \
       --out-s3 s3://chula-aigov-car-video-training-487984284636/models/phuket-yolo11s/
   ```

3. **Use it.** Point the existing tools at the fine-tuned weights:
   `MODEL_PATH=models/phuket-yolo11s.pt` for `infer.py`, or
   `--model runs/detect/phuket-finetune/weights/best.pt` for `track_analytics.py`.

## Staged dataset: phuket-yolo-v1 (2026-07-15)

Ready-to-train dataset on S3 — no need to rebuild it on the GPU box:

- `s3://chula-aigov-car-video-training-487984284636/datasets/phuket-yolo-v1.tar.gz` (334 MB)
- 310 train / 89 val frames from **15 clips / 12 distinct cameras**, all 3 locations,
  timebands 00:00–23:00 (incl. two night cameras and one 2688×1520@15fps HEVC unit)
- 4,702 YOLO11l auto-labeled boxes: 2,791 car / 1,333 motorcycle / 499 truck / 70 bus / 9 bicycle
- Val = 3 cameras that appear nowhere in train (Chalong C13 night, Kathu C12 day,
  Sakhu C15 day) — measures unseen-viewpoint generalization, the deployment scenario.

On the GPU box (RunPod pod or EC2 g4dn once quota clears), the whole run is:

```bash
aws s3 cp s3://chula-aigov-car-video-training-487984284636/datasets/phuket-yolo-v1.tar.gz - | tar xz
python3 finetune/train_phuket.py --data phuket-yolo/dataset.yaml \
    --model yolo11s.pt --epochs 40 --imgsz 1280 --batch 8 \
    --out-s3 s3://chula-aigov-car-video-training-487984284636/models/phuket-yolo11s/
```

To grow the dataset: `finetune/cut_clips.py` cuts new clips from any S3 recording
via byte-range head fetch, then re-run `build_dataset.py`.

## Pilot results (2026-07-15, M4 Pro CPU, YOLO11n @640, 20 epochs)

| Dataset | Val setup | val mAP50 | mAP50-95 |
|---|---|---|---|
| 4 clips, 70 train frames | 1 held-out camera | 0.123 | 0.059 |
| **v1: 15 clips, 310 train frames** | 3 unseen cameras incl. night | **0.280** | **0.167** |

Scaling the dataset 4.4× more than doubled unseen-camera accuracy with the same
tiny student — the in-domain-data lever from the research doc, observed directly.

## GPU result: phuket-yolo11s (2026-07-15, RunPod RTX 3090)

The real run — YOLO11s @1280, 40 epochs on phuket-yolo-v1 via `runpod_job.sh`
(unattended pod, auto-terminated; ~$0.15 of GPU time):

| Metric | val (3 unseen cameras, incl. night) |
|---|---|
| mAP50 | **0.702** |
| mAP50-95 | **0.509** (car 0.654, motorcycle 0.423, truck 0.449) |

Weights: `models/phuket-yolo11s.pt` (also
`s3://chula-aigov-car-video-training-487984284636/models/phuket-yolo11s/best.pt`,
with training curves, confusion matrix, and the full job log alongside).
Use with `MODEL_PATH=models/phuket-yolo11s.pt` (infer.py) or
`--model models/phuket-yolo11s.pt` (track_analytics.py). Note the model predicts
the 5 remapped classes (car/motorcycle/bus/truck/bicycle), not COCO ids.

Caveat: val labels are teacher-generated (YOLO11l @1920), so this measures
agreement with the teacher on unseen cameras, not human ground truth. A small
human-labeled test set is the next rigor upgrade.

## YOLO26 comparison (2026-07-16, same recipe/data/GPU class)

| Student | val mAP50 | val mAP50-95 | car | motorcycle | truck | CPU ms/frame @1280 |
|---|---|---|---|---|---|---|
| yolo11s fine-tune | 0.702 | 0.509 | 0.654 | 0.423 | 0.449 | 230 |
| **yolo26s fine-tune** | **0.722** | **0.552** | 0.689 | 0.459 | 0.509 | 250 |

YOLO26s (end-to-end, NMS-free) wins every class at near-identical size —
**+8.5% relative mAP50-95** — for ~9% slower CPU inference in this setup.
Adopted as the default: `models/phuket-yolo26s.pt`
(S3: `models/phuket-yolo26s/best.pt`). Train any student via
`STUDENT=yolo26s.pt RUN_NAME=phuket-yolo26s` env on `runpod_job.sh`.

## Scaling up the dataset

- More clips → more diversity. Cut 20 s clips from any S3 source with a byte-range
  head fetch (no 5 GB downloads needed) — the IMKH "`.mp4`" heads decode fine with
  OpenCV. See `data/clips/` naming: `<location>_<cam>_<HHMM>_demo20s.mp4`.
- Include night clips in training; the teacher labels them adequately in lit scenes.
  For hard night/rain cases the upgrade path is day→night style-transfer augmentation
  (labeling-free, see research doc) or a human-review pass in CVAT/Label Studio.
- A bigger teacher (`yolo11x.pt`) at `--imgsz 1920` and lower `--conf 0.35` catches
  more distant vehicles at slightly higher label noise.

## Caveats

- Auto-labels inherit teacher blind spots (distant/occluded vehicles at night); the
  student can't exceed the teacher without human-corrected labels.
- `--freeze 10` (default) trains only the head/neck — right for small noisy datasets;
  switch to `--freeze 0` once the dataset is thousands of frames.
- Classes are fixed to car/motorcycle/bus/truck/bicycle (COCO ids remapped to 0–4).
