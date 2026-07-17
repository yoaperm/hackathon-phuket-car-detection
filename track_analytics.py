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


def resolve_line(spec: str, stem: str) -> list[float]:
    """'auto' -> longest camera_lines.json prefix match for this clip stem;
    otherwise parse the explicit x1,y1,x2,y2 fractions."""
    if spec != "auto":
        return [float(v) for v in spec.split(",")]
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "camera_lines.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            table = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        best = max((k for k in table if stem.startswith(k)), key=len, default=None)
        if best:
            print(f"counting line from camera_lines.json[{best}]: {table[best]}")
            return table[best]
    print("no per-camera line found; using horizontal default")
    return [0, 0.55, 1, 0.55]


def load_bev(stem: str):
    """(quad fractions, (width_m, length_m)) for this camera, or None.

    The '<camera>_bev' quad in camera_lines.json maps the road plane to a
    metric rectangle. Scale is anchored to assumed road dimensions until the
    quad is surveyed — downstream copy says so.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "camera_lines.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        table = json.load(f)
    keys = [k[:-4] for k in table if k.endswith("_bev")]
    best = max((k for k in keys if stem.startswith(k)), key=len, default=None)
    if not best:
        return None
    return table[best + "_bev"], tuple(table.get(best + "_bev_size", [15, 40]))


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
    ap.add_argument("--line", default="auto",
                    help="counting line as x1,y1,x2,y2 fractions of the frame; "
                         "'auto' resolves per camera from camera_lines.json")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="ultralytics tracker config (bytetrack.yaml or botsort.yaml)")
    ap.add_argument("--stationary-secs", type=float, default=8.0,
                    help="a track this old that has not moved is flagged stationary "
                         "(0 disables the stationary/obstruction detector)")
    ap.add_argument("--incident-stop-secs", type=float, default=3.0,
                    help="a previously-moving vehicle stopped this long fires an "
                         "incident-signature event")
    ap.add_argument("--incident-move-speed", type=float, default=0.025,
                    help="fraction of frame width per second that counts as "
                         "'was driving' for the incident detector")
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

    fx = resolve_line(a.line, stem)
    line_a = (int(fx[0] * out_w), int(fx[1] * out_h))
    line_b = (int(fx[2] * out_w), int(fx[3] * out_h))

    # Stationary/obstruction detector: a track older than --stationary-secs
    # whose center stayed within STATIONARY_RADIUS of where it was back then is
    # flagged (parked or stopped in/near the roadway). Camera-relative pixels —
    # no homography, so this is a per-view heuristic, not metric displacement.
    stat_window = int(a.stationary_secs * src_fps)
    STATIONARY_RADIUS = 0.015 * out_w           # ~19 px at 1280
    RELEASE_RADIUS = 2.5 * STATIONARY_RADIUS    # hysteresis so flags don't flicker
    positions = defaultdict(lambda: deque(maxlen=max(stat_window + 1, 2)))
    stationary_now = set()      # track ids currently flagged
    stationary_ever = set()     # unique flagged tracks (reported total)
    stationary_since = {}       # track id -> timestamp first flagged
    events = []                 # obstruction/incident alerts (consumed by dashboards)

    # Incident signatures: a vehicle that WAS moving comes to a sudden stop in
    # the roadway. Rule-based on real tracks — indicative, not a verified
    # accident (signal stops also match; the copy says so). Escalates when
    # several vehicles stop near each other within a short window.
    MOVE_SPEED = a.incident_move_speed * out_w   # px/s that counts as "was driving"
    STOP_SPEED = 0.006 * out_w        # px/s below which the vehicle is stopped
    STOP_PERSIST_S = a.incident_stop_secs        # stay stopped this long to fire
    CLUSTER_PX = 0.10 * out_w         # nearby-stop distance for escalation
    peak_speed = defaultdict(float)   # track id -> lifetime peak speed (px/s)
    stop_frames = Counter()           # track id -> consecutive stopped frames
    incident_fired = {}               # track id -> (t, center) of its incident
    incidents_total = 0

    # Near-miss detection in road-plane metres (needs a BEV quad). Two
    # vehicles closing fast at short range → surrogate-safety event (TTC
    # style). Thresholds are conservative because Thai mixed traffic follows
    # closely: requires genuine closing speed, not just proximity.
    bev_cfg = load_bev(stem)
    bev_H = None
    if bev_cfg:
        quad, (bev_wm, bev_lm) = bev_cfg
        bev_src = np.float32(quad) * np.float32([out_w, out_h])
        bev_dst = np.float32([[0, 0], [bev_wm, 0], [bev_wm, bev_lm], [0, bev_lm]])
        bev_H = cv2.getPerspectiveTransform(bev_src, bev_dst)  # image px -> metres
        print(f"near-miss detector on: BEV quad {bev_wm}x{bev_lm} m")
    NM_DIST_M = 2.5        # centre distance at closest approach
    NM_CLOSING_KMH = 15.0  # relative closing speed
    NM_TTC_S = 1.2         # projected time-to-collision
    bev_hist = defaultdict(lambda: deque(maxlen=int(src_fps) + 2))  # tid->(f,x,y)
    nm_cooldown = {}       # pair -> last fire t
    nm_flash = []          # (until_frame, tid_a, tid_b) — keep annotation visible
    nearmiss_total = 0

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
        bev_now = {}   # tid -> (xm, ym, name) this frame, for the near-miss pass
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

                t_now = frame_idx / src_fps
                pos = positions[tid]
                pos.append((frame_idx, center))

                if bev_H is not None:
                    gx, gy = (x1 + x2) / 2, y2   # ground-contact point
                    m = cv2.perspectiveTransform(
                        np.float32([[[gx, gy]]]), bev_H)[0][0]
                    # only inside the calibrated quad (+2 m margin): outside it
                    # the plane extrapolation degrades and fakes closings
                    if -2 <= m[0] <= bev_wm + 2 and -2 <= m[1] <= bev_lm + 2:
                        bev_hist[tid].append((frame_idx, float(m[0]), float(m[1])))
                        bev_now[tid] = (float(m[0]), float(m[1]), name)

                # ~1s-lookback speed for the incident detector
                if a.stationary_secs > 0:
                    ref = next((p for p in pos
                                if p[0] >= frame_idx - int(src_fps)), pos[0])
                    span_s = max((frame_idx - ref[0]) / src_fps, 1e-6)
                    if frame_idx - ref[0] >= int(src_fps * 0.6):
                        speed = ((center[0] - ref[1][0]) ** 2
                                 + (center[1] - ref[1][1]) ** 2) ** 0.5 / span_s
                        peak_speed[tid] = max(peak_speed[tid], speed)
                        stop_frames[tid] = stop_frames[tid] + 1 \
                            if speed < STOP_SPEED else 0
                        if (tid not in incident_fired
                                and peak_speed[tid] >= MOVE_SPEED
                                and stop_frames[tid] >= STOP_PERSIST_S * src_fps):
                            incident_fired[tid] = (t_now, center)
                            incidents_total += 1
                            near = [o for o, (ot, oc) in incident_fired.items()
                                    if o != tid and t_now - ot <= 10
                                    and ((center[0] - oc[0]) ** 2
                                         + (center[1] - oc[1]) ** 2) ** 0.5 <= CLUSTER_PX]
                            events.append({
                                "t": round(t_now, 1), "type": "incident",
                                "severity": "emergency",
                                "msg": f"Sudden stop in roadway — {name} #{tid} was "
                                       f"moving, now stopped {STOP_PERSIST_S:.0f}s+ "
                                       "(possible incident or signal stop)"})
                            if near:
                                events.append({
                                    "t": round(t_now, 1), "type": "incident",
                                    "severity": "emergency",
                                    "msg": f"{len(near) + 1} vehicles stopped together "
                                           "near the same spot — possible collision "
                                           "or blockage"})
                if a.stationary_secs > 0 and len(pos) > 1:
                    oldest_frame, oldest_c = pos[0]
                    span_ok = frame_idx - oldest_frame >= stat_window * 0.9
                    moved = ((center[0] - oldest_c[0]) ** 2
                             + (center[1] - oldest_c[1]) ** 2) ** 0.5
                    if span_ok and moved < STATIONARY_RADIUS \
                            and tid not in stationary_now:
                        stationary_now.add(tid)
                        stationary_since[tid] = t_now
                        if tid not in stationary_ever:
                            stationary_ever.add(tid)
                            events.append({
                                "t": round(t_now, 1), "type": "obstruction",
                                "severity": "alert",
                                "msg": f"Stationary {name} #{tid} — no movement "
                                       f"for {a.stationary_secs:.0f}s+"})
                    elif tid in stationary_now and moved > RELEASE_RADIUS:
                        stationary_now.discard(tid)
                        events.append({
                            "t": round(t_now, 1), "type": "obstruction",
                            "severity": "info",
                            "msg": f"{name} #{tid} moving again after "
                                   f"{t_now - stationary_since.get(tid, t_now):.0f}s stationary"})

                side = side_of_line(center, line_a, line_b)
                prev = last_side.get(tid)
                if prev is not None and side != 0 and prev != 0 and side != prev \
                        and tid not in counted:
                    counted.add(tid)
                    flow["a_to_b" if prev < side else "b_to_a"] += 1
                if side != 0:
                    last_side[tid] = side

                c = COLORS[name]
                if tid in incident_fired and stop_frames[tid] > 0:
                    # flashing marker while the suddenly-stopped vehicle stays put
                    if (frame_idx // int(src_fps / 2 or 1)) % 2 == 0:
                        cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6),
                                      (0, 0, 255), 3)
                    cv2.putText(frame, "! INCIDENT?", (x1, y1 - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                                cv2.LINE_AA)
                if tid in stationary_now:
                    c = (0, 0, 255)
                    dwell = frame_idx / src_fps - stationary_since.get(tid, 0)
                    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), c, 3)
                    cv2.putText(frame, f"STATIONARY {dwell:.0f}s", (x1, y2 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                label = f"#{tid} {name} {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), c, -1)
                cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 1, cv2.LINE_AA)
                pts = np.array(trails[tid], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, c, 1, cv2.LINE_AA)

        # near-miss pass: pairwise closing-speed / TTC in road-plane metres
        if bev_H is not None and len(bev_now) >= 2:
            t_now = frame_idx / src_fps
            vels = {}
            for tid, (xm, ym, _) in bev_now.items():
                h = bev_hist[tid]
                ref = next((e for e in h if e[0] >= frame_idx - int(src_fps * 0.8)),
                           h[0])
                span = (frame_idx - ref[0]) / src_fps
                if span >= 0.5:
                    vels[tid] = ((xm - ref[1]) / span, (ym - ref[2]) / span)
            tids = [t for t in bev_now if t in vels]
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    ta, tb = tids[i], tids[j]
                    ax, ay, an = bev_now[ta]
                    bx, by, bn = bev_now[tb]
                    dx, dy = bx - ax, by - ay
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > 12 or dist < 1e-6:
                        continue
                    rvx = vels[tb][0] - vels[ta][0]
                    rvy = vels[tb][1] - vels[ta][1]
                    closing = -(dx * rvx + dy * rvy) / dist   # m/s, >0 approaching
                    if closing * 3.6 < NM_CLOSING_KMH:
                        continue
                    ttc = dist / closing
                    # predicted miss distance at closest approach under linear
                    # motion — filters opposing streams that pass a lane apart
                    rv = (rvx * rvx + rvy * rvy) ** 0.5
                    along = abs(dx * rvx + dy * rvy) / max(rv, 1e-6)
                    d_miss = max(dist * dist - along * along, 0.0) ** 0.5
                    danger_now = dist < NM_DIST_M and closing * 3.6 >= NM_CLOSING_KMH
                    danger_pred = ttc < NM_TTC_S and d_miss < 1.2 and dist < 10.0
                    if not (danger_now or danger_pred):
                        continue
                    pair = (min(ta, tb), max(ta, tb))
                    if t_now - nm_cooldown.get(pair, -99) < 5.0:
                        continue
                    nm_cooldown[pair] = t_now
                    nearmiss_total += 1
                    events.append({
                        "t": round(t_now, 1), "type": "nearmiss",
                        "severity": "alert",
                        "msg": f"Near-miss signature: {an} #{ta} & {bn} #{tb} — "
                               f"{dist:.1f} m apart, closing {closing * 3.6:.0f} km/h "
                               f"(TTC {ttc:.1f}s)"})
                    nm_flash.append((frame_idx + int(1.5 * src_fps), ta, tb))
            nm_flash = [f for f in nm_flash if f[0] >= frame_idx]
            for _, ta, tb in nm_flash:
                if ta in bev_now and tb in bev_now:
                    pa, pb = trails[ta][-1], trails[tb][-1]
                    cv2.line(frame, pa, pb, (0, 200, 255), 3)
                    mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
                    cv2.putText(frame, "NEAR MISS", (mid[0] - 40, mid[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
                                cv2.LINE_AA)

        active_window.append(active)
        mean_active = sum(active_window) / len(active_window)
        state = congestion_state(mean_active)

        cv2.line(frame, line_a, line_b, (255, 255, 0), 2)
        hud1 = (f"active:{active}  unique:{len(seen_ids)}  "
                f"flow A>B:{flow['a_to_b']} B>A:{flow['b_to_a']}  "
                f"stationary:{len(stationary_now)}")
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
                     "stationary_active": len(stationary_now),
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
        "counting_line": fx,
        "stationary_vehicles_total": len(stationary_ever),
        "stationary_by_class": dict(Counter(
            track_class[tid] for tid in stationary_ever if tid in track_class)),
        "incidents_total": incidents_total,
        "nearmiss_total": nearmiss_total,
        "events": events,
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
