#!/usr/bin/env python3
"""One-shot, headless YOLOv8 traffic-density batch inference for RunPod.

Pulls .mp4 videos from S3 (or a local dir), runs GPU inference reusing the
lane-counting/annotation logic from ``real_time_traffic_analysis.py``, and writes
annotated videos, preview images, and CSV summaries to OUTPUT_DIR, then exits.

Everything is configured via environment variables (see .env.example). No server,
no GUI dependency by default, no infinite loop.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants ported from real_time_traffic_analysis.py.
#
# These pixel coordinates were hand-tuned for the bundled sample_video.mp4,
# which is 1280x720. scale_roi() scales them linearly for other resolutions.
# NOTE: this is a linear approximation that is only correct when the source
# video shares roughly the same camera framing/perspective as the sample.
# ---------------------------------------------------------------------------
REFERENCE_FRAME_WIDTH = 1280
REFERENCE_FRAME_HEIGHT = 720

# Horizontal band kept for detection. Originally named x1/x2, but they index
# *rows* (detection_frame[:x1, :] = 0), so they are row bounds, not x-coords.
DEFAULT_ROI_ROW_START = 325
DEFAULT_ROI_ROW_END = 635

# Genuine x-coordinate (column) used to split left vs. right lane via box[0].
DEFAULT_LANE_SPLIT_X = 609

# Cosmetic lane-divider polygons drawn on the output frame.
DEFAULT_VERTICES1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.float64)
DEFAULT_VERTICES2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.float64)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255, 255, 255)      # white text
BACKGROUND_COLOR = (0, 0, 255)    # red background


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "y")


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


@dataclass
class Config:
    s3_input: str | None
    local_video_dir: str | None
    output_dir: Path
    model_path: Path
    conf_threshold: float
    img_size: int
    heavy_traffic_threshold: int
    enable_gui: bool
    allow_cpu: bool
    save_annotated_video: bool
    preview_frame_fraction: float
    disable_roi: bool
    detect_classes: list[str]


def load_config() -> Config:
    cfg = Config(
        s3_input=os.environ.get("S3_INPUT") or None,
        local_video_dir=os.environ.get("LOCAL_VIDEO_DIR") or None,
        output_dir=Path(os.environ.get("OUTPUT_DIR", "/workspace/output")),
        model_path=Path(os.environ.get("MODEL_PATH", "models/best.pt")),
        conf_threshold=_env_float("CONF_THRESHOLD", 0.4),
        img_size=_env_int("IMG_SIZE", 640),
        heavy_traffic_threshold=_env_int("HEAVY_TRAFFIC_THRESHOLD", 10),
        enable_gui=_env_bool("ENABLE_GUI", False),
        allow_cpu=_env_bool("ALLOW_CPU", False),
        save_annotated_video=_env_bool("SAVE_ANNOTATED_VIDEO", True),
        preview_frame_fraction=_env_float("PREVIEW_FRAME_FRACTION", 0.5),
        disable_roi=_env_bool("DISABLE_ROI", False),
        detect_classes=[
            c.strip().lower()
            for c in os.environ.get("DETECT_CLASSES", "").split(",")
            if c.strip()
        ],
    )
    if not cfg.s3_input and not cfg.local_video_dir:
        sys.exit(
            "ERROR: no input configured. Set S3_INPUT (comma-separated s3:// URIs "
            "or s3://bucket/prefix/) or LOCAL_VIDEO_DIR (a folder of .mp4 files)."
        )
    if cfg.s3_input and cfg.local_video_dir:
        print(
            "WARNING: both S3_INPUT and LOCAL_VIDEO_DIR are set; using S3_INPUT.",
            file=sys.stderr,
        )
    # Bare well-known names (e.g. "yolov8s.pt", no directory part) are handed to
    # ultralytics as-is, which auto-downloads them; only real paths must exist.
    is_bare_name = cfg.model_path.name == str(cfg.model_path)
    if not is_bare_name and not cfg.model_path.is_file():
        sys.exit(f"ERROR: model weights not found at {cfg.model_path}")
    return cfg


# ---------------------------------------------------------------------------
# GPU enforcement
# ---------------------------------------------------------------------------
def resolve_device(allow_cpu: bool) -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name} (device=cuda:0)")
        return "cuda:0"
    if not allow_cpu:
        sys.exit(
            "ERROR: No CUDA GPU detected. This job requires a GPU pod.\n"
            "Set ALLOW_CPU=true to run on CPU (much slower, not recommended)."
        )
    print(
        "WARNING: No GPU detected — falling back to CPU (ALLOW_CPU=true). "
        "This will be slow.",
        file=sys.stderr,
    )
    return "cpu"


# ---------------------------------------------------------------------------
# S3 layer (boto3 default credential chain: env vars, shared config, or IAM role)
# ---------------------------------------------------------------------------
def get_s3_client():
    import boto3

    return boto3.client("s3")


def resolve_video_uris(s3_client, raw_input: str) -> list[str]:
    """Expand a comma-separated S3 input into a flat list of s3:// object URIs.

    Entries ending in '/' are treated as prefixes and listed (single page,
    up to 1000 objects) for .mp4 files; other entries are taken as direct keys.
    """
    uris: list[str] = []
    for entry in (e.strip() for e in raw_input.split(",")):
        if not entry:
            continue
        parsed = urlparse(entry)
        if parsed.scheme != "s3":
            print(f"WARNING: skipping non-s3 entry {entry!r}", file=sys.stderr)
            continue
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        if entry.endswith("/") or key == "":
            resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=key)
            for obj in resp.get("Contents", []):
                if obj["Key"].lower().endswith(".mp4"):
                    uris.append(f"s3://{bucket}/{obj['Key']}")
        else:
            uris.append(entry)
    return uris


def download_video(s3_client, uri: str, dest_dir: Path) -> Path:
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    parsed = urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    dest = dest_dir / Path(key).name
    print(f"Downloading {uri} -> {dest}")
    try:
        s3_client.download_file(bucket, key, str(dest))
    except (ClientError, BotoCoreError, NoCredentialsError) as exc:
        raise RuntimeError(f"failed to download {uri}: {exc}") from exc
    return dest


# ---------------------------------------------------------------------------
# ROI scaling
# ---------------------------------------------------------------------------
@dataclass
class RoiParams:
    row_start: int
    row_end: int
    lane_split_x: int
    scale_x: float
    scale_y: float
    vertices1: np.ndarray
    vertices2: np.ndarray


def scale_roi(frame_w: int, frame_h: int) -> RoiParams:
    sx = frame_w / REFERENCE_FRAME_WIDTH
    sy = frame_h / REFERENCE_FRAME_HEIGHT
    return RoiParams(
        row_start=int(round(DEFAULT_ROI_ROW_START * sy)),
        row_end=int(round(DEFAULT_ROI_ROW_END * sy)),
        lane_split_x=int(round(DEFAULT_LANE_SPLIT_X * sx)),
        scale_x=sx,
        scale_y=sy,
        vertices1=np.round(DEFAULT_VERTICES1 * np.array([sx, sy])).astype(np.int32),
        vertices2=np.round(DEFAULT_VERTICES2 * np.array([sx, sy])).astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Per-frame counting + annotation
# ---------------------------------------------------------------------------
def count_vehicles_by_lane(boxes_xyxy: np.ndarray, lane_split_x: float) -> tuple[int, int]:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return 0, 0
    left = int(np.count_nonzero(boxes_xyxy[:, 0] < lane_split_x))
    right = len(boxes_xyxy) - left
    return left, right


def _draw_label(frame, text, position, scale, background_color):
    font_scale = scale
    thickness = max(1, int(round(2 * scale)))
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        frame,
        (x - 10, y - th - 10),
        (x + tw + 10, y + 10),
        background_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), FONT, font_scale, FONT_COLOR, thickness, cv2.LINE_AA)


def annotate_frame(frame, roi: RoiParams, left, right, status_left, status_right, draw_lanes=True):
    if draw_lanes:
        cv2.polylines(frame, [roi.vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [roi.vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

    scale = max(0.5, min(roi.scale_x, roi.scale_y))
    left_x = int(round(10 * roi.scale_x))
    right_x = int(round(820 * roi.scale_x))
    row1 = int(round(50 * roi.scale_y))
    row2 = int(round(100 * roi.scale_y))

    _draw_label(frame, f"Vehicles in Left Lane: {left}", (left_x, row1), scale, BACKGROUND_COLOR)
    _draw_label(frame, f"Traffic Intensity: {status_left}", (left_x, row2), scale, BACKGROUND_COLOR)
    _draw_label(frame, f"Vehicles in Right Lane: {right}", (right_x, row1), scale, BACKGROUND_COLOR)
    _draw_label(frame, f"Traffic Intensity: {status_right}", (right_x, row2), scale, BACKGROUND_COLOR)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------
def process_video(
    model,
    video_path: Path,
    cfg: Config,
    device: str,
    s3_uri: str | None,
    class_ids: list[int] | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # 0 or NaN -> fall back
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi = scale_roi(width, height)
    stem = video_path.stem

    writer = None
    out_video_path = ""
    if cfg.save_annotated_video:
        out_video_path = str(cfg.output_dir / "videos" / f"{stem}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {out_video_path}")

    preview_path = str(cfg.output_dir / "previews" / f"{stem}_preview.jpg")
    preview_target = int(total_frames * cfg.preview_frame_fraction) if total_frames > 0 else 0
    preview_saved = False

    records: list[dict] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cfg.disable_roi:
            detection_frame = frame
        else:
            detection_frame = frame.copy()
            detection_frame[: roi.row_start, :] = 0
            detection_frame[roi.row_end :, :] = 0

        results = model.predict(
            detection_frame,
            imgsz=cfg.img_size,
            conf=cfg.conf_threshold,
            device=device,
            classes=class_ids,
            verbose=False,
        )
        processed = results[0].plot(line_width=1)

        if not cfg.disable_roi:
            # Restore the masked-out regions from the original frame for display.
            processed[: roi.row_start, :] = frame[: roi.row_start, :]
            processed[roi.row_end :, :] = frame[roi.row_end :, :]

        boxes = results[0].boxes.xyxy.cpu().numpy()
        left, right = count_vehicles_by_lane(boxes, roi.lane_split_x)
        status_left = "Heavy" if left > cfg.heavy_traffic_threshold else "Smooth"
        status_right = "Heavy" if right > cfg.heavy_traffic_threshold else "Smooth"

        annotate_frame(
            processed, roi, left, right, status_left, status_right,
            draw_lanes=not cfg.disable_roi,
        )

        records.append(
            {
                "frame_index": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 3),
                "left_count": left,
                "right_count": right,
                "total_count": left + right,
                "left_status": status_left,
                "right_status": status_right,
            }
        )

        if writer is not None:
            writer.write(processed)

        if not preview_saved and frame_idx >= preview_target:
            cv2.imwrite(preview_path, processed)
            preview_saved = True

        if cfg.enable_gui:
            cv2.imshow("Real-time Traffic Analysis", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if cfg.enable_gui:
        cv2.destroyAllWindows()

    if not preview_saved and records:
        cv2.imwrite(preview_path, processed)  # fallback: last processed frame
        preview_saved = True

    # Per-frame CSV
    csv_path = cfg.output_dir / "csv" / f"{stem}_frame_counts.csv"
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    processed_frames = len(records)
    heavy_thr = cfg.heavy_traffic_threshold
    return {
        "video_filename": video_path.name,
        "s3_uri": s3_uri or "",
        "total_frames": processed_frames,
        "duration_sec": round(processed_frames / fps, 3) if processed_frames else 0,
        "avg_left_count": round(df["left_count"].mean(), 3) if processed_frames else 0,
        "avg_right_count": round(df["right_count"].mean(), 3) if processed_frames else 0,
        "max_left_count": int(df["left_count"].max()) if processed_frames else 0,
        "max_right_count": int(df["right_count"].max()) if processed_frames else 0,
        "pct_frames_heavy_left": round(100.0 * (df["left_count"] > heavy_thr).mean(), 2)
        if processed_frames
        else 0,
        "pct_frames_heavy_right": round(100.0 * (df["right_count"] > heavy_thr).mean(), 2)
        if processed_frames
        else 0,
        "output_video_path": out_video_path,
        "preview_image_path": preview_path if preview_saved else "",
        "device_used": device,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def resolve_inputs(cfg: Config, staging_dir: Path) -> list[tuple[Path, str | None]]:
    """Return a list of (local_path, s3_uri_or_None) to process."""
    if cfg.s3_input:
        s3 = get_s3_client()
        uris = resolve_video_uris(s3, cfg.s3_input)
        if not uris:
            sys.exit(f"ERROR: no .mp4 files found for S3_INPUT={cfg.s3_input!r}")
        staging_dir.mkdir(parents=True, exist_ok=True)
        return [(download_video(s3, uri, staging_dir), uri) for uri in uris]

    local_dir = Path(cfg.local_video_dir)
    if not local_dir.is_dir():
        sys.exit(f"ERROR: LOCAL_VIDEO_DIR is not a directory: {local_dir}")
    paths = sorted(local_dir.glob("*.mp4"))
    if not paths:
        sys.exit(f"ERROR: no .mp4 files found in LOCAL_VIDEO_DIR={local_dir}")
    return [(p, None) for p in paths]


def main() -> None:
    cfg = load_config()
    device = resolve_device(cfg.allow_cpu)

    for sub in ("videos", "previews", "csv"):
        (cfg.output_dir / sub).mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"Loading model from {cfg.model_path}")
    model = YOLO(str(cfg.model_path))

    class_ids: list[int] | None = None
    if cfg.detect_classes:
        name_to_id = {str(v).lower(): int(k) for k, v in model.names.items()}
        unknown = [c for c in cfg.detect_classes if c not in name_to_id]
        if unknown:
            sys.exit(
                f"ERROR: DETECT_CLASSES names not in model: {unknown}. "
                f"Model classes: {sorted(name_to_id)}"
            )
        class_ids = [name_to_id[c] for c in cfg.detect_classes]
        print(f"Filtering detections to classes: {cfg.detect_classes} -> ids {class_ids}")

    inputs = resolve_inputs(cfg, cfg.output_dir / "_staging")
    print(f"Processing {len(inputs)} video(s)...")

    summary_rows: list[dict] = []
    for local_path, s3_uri in inputs:
        try:
            row = process_video(model, local_path, cfg, device, s3_uri, class_ids)
            summary_rows.append(row)
            print(f"[OK] {local_path.name}: {row['total_frames']} frames")
        except Exception as exc:  # noqa: BLE001 - batch keeps going on per-video failure
            print(f"[ERROR] {local_path.name}: {exc}", file=sys.stderr)
            continue

    if summary_rows:
        summary_path = cfg.output_dir / "csv" / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Wrote summary: {summary_path}")

    if not summary_rows:
        sys.exit("ERROR: no videos processed successfully.")

    print(f"Done. Results in {cfg.output_dir}")


if __name__ == "__main__":
    main()
