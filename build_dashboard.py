#!/usr/bin/env python3
"""
Build a self-contained HTML traffic-analytics dashboard from track_analytics.py
outputs. Scans --analytics-dir for <name>_analytics.json + <name>_frames.csv
(+ optional <name>_tracked.mp4 for a preview frame) and writes one dashboard
HTML with all images inlined — no external requests, shareable as a single file.

Example:
  python3 build_dashboard.py --analytics-dir analytics-out --out dashboard.html
"""
import argparse
import base64
import csv
import glob
import html
import json
import os

import cv2

CONGESTION_COLOR = {"Free": "#3fb96e", "Moderate": "#f5a623", "Heavy": "#e5484d"}
CLASS_COLOR = {"car": "#4c9be8", "motorcycle": "#f5a623", "truck": "#e5484d",
               "bus": "#b07ce8", "bicycle": "#3fb96e"}

# Friendly labels for known demo clips; unknown stems fall back to the stem.
CAMERA_LABELS = {
    "chalong_c1_0700_demo30s": ("Chalong C1", "07:00 · morning rain"),
    "chalong_c1_2200_demo20s": ("Chalong C1", "22:00 · night"),
    "kathu_c10_1200_demo20s": ("Kathu C10", "12:00 · intersection"),
    "sakhu_c1_0800_demo20s": ("Sakhu C1", "08:00 · morning"),
}


def preview_data_uri(video_path: str, frac: float = 0.6) -> str:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ""
    frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def density_svg(rows: list[dict], w: int = 560, h: int = 96) -> str:
    ys = [int(r["active_tracks"]) for r in rows]
    if not ys:
        return ""
    ymax = max(max(ys), 1)
    pts = [(i * w / max(len(ys) - 1, 1), h - y * (h - 8) / ymax) for i, y in enumerate(ys)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} " + line + f" {w},{h}"
    grid = "".join(
        f'<line x1="0" y1="{h - lvl * (h - 8) / ymax:.1f}" x2="{w}" '
        f'y2="{h - lvl * (h - 8) / ymax:.1f}" stroke="#242d38" stroke-width="1"/>'
        for lvl in range(0, ymax + 1, max(ymax // 3, 1))
    )
    ex, ey = pts[-1]
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" class="spark">'
        f"{grid}"
        f'<polygon points="{area}" fill="rgba(245,166,35,0.12)"/>'
        f'<polyline points="{line}" fill="none" stroke="#f5a623" stroke-width="1.5"/>'
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="#f5a623"/>'
        f'<text x="{w - 6}" y="12" text-anchor="end" class="sparkmax">peak {ymax}</text>'
        "</svg>"
    )


def congestion_strip(rows: list[dict], w: int = 560, h: int = 10) -> str:
    if not rows:
        return ""
    n = len(rows)
    segs, start = [], 0
    for i in range(1, n + 1):
        if i == n or rows[i]["congestion"] != rows[start]["congestion"]:
            color = CONGESTION_COLOR[rows[start]["congestion"]]
            segs.append(f'<rect x="{start * w / n:.1f}" y="0" '
                        f'width="{(i - start) * w / n:.1f}" height="{h}" fill="{color}"/>')
            start = i
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'class="strip">{"".join(segs)}</svg>')


def class_bar(by_class: dict, total: int) -> str:
    if not total:
        return ""
    cells = "".join(
        f'<div class="mixseg" style="flex:{n};background:{CLASS_COLOR.get(c, "#8b96a5")}" '
        f'title="{c}: {n}"></div>'
        for c, n in sorted(by_class.items(), key=lambda kv: -kv[1])
    )
    legend = "".join(
        f'<span class="chip"><i style="background:{CLASS_COLOR.get(c, "#8b96a5")}"></i>'
        f"{c} {n}</span>"
        for c, n in sorted(by_class.items(), key=lambda kv: -kv[1])
    )
    return f'<div class="mix">{cells}</div><div class="chips">{legend}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytics-dir", default="analytics-out")
    ap.add_argument("--out", default="dashboard.html")
    a = ap.parse_args()

    cams = []
    for jpath in sorted(glob.glob(os.path.join(a.analytics_dir, "*_analytics.json"))):
        stem = os.path.basename(jpath)[: -len("_analytics.json")]
        with open(jpath) as f:
            meta = json.load(f)
        rows = []
        cpath = os.path.join(a.analytics_dir, f"{stem}_frames.csv")
        if os.path.exists(cpath):
            with open(cpath) as f:
                rows = list(csv.DictReader(f))
        vpath = os.path.join(a.analytics_dir, f"{stem}_tracked.mp4")
        img = preview_data_uri(vpath) if os.path.exists(vpath) else ""
        cams.append((stem, meta, rows, img))

    if not cams:
        raise SystemExit(f"no *_analytics.json found in {a.analytics_dir}")

    total_unique = sum(m["unique_vehicles_total"] for _, m, _, _ in cams)
    total_crossings = sum(m["line_crossings"]["a_to_b"] + m["line_crossings"]["b_to_a"]
                          for _, m, _, _ in cams)
    busiest_stem, busiest_meta = max(
        ((s, m) for s, m, _, _ in cams), key=lambda x: x[1]["unique_vehicles_total"])
    busiest_label = CAMERA_LABELS.get(busiest_stem, (busiest_stem, ""))[0]
    model = cams[0][1].get("model", "?")
    tracker = cams[0][1].get("tracker", "?").replace(".yaml", "")

    cards = []
    for stem, meta, rows, img in cams:
        label, sub = CAMERA_LABELS.get(stem, (stem, ""))
        heavy_pct = round(meta["congestion_share"].get("Heavy", 0) * 100)
        flow = meta["line_crossings"]
        state = meta.get("congestion_final", "n/a")
        state_c = CONGESTION_COLOR.get(state, "#8b96a5")
        cards.append(f"""
    <section class="card">
      <header class="cardhead">
        <div>
          <h2>{html.escape(label)}</h2>
          <p class="sub">{html.escape(sub)} · {meta["duration_sec"]:.0f}s sample</p>
        </div>
        <span class="state" style="--c:{state_c}">{state}</span>
      </header>
      {f'<img class="shot" src="{img}" alt="annotated frame from {html.escape(label)}">' if img else ""}
      <div class="metrics">
        <div class="metric"><b>{meta["unique_vehicles_total"]}</b><span>unique vehicles</span></div>
        <div class="metric"><b>{flow["a_to_b"] + flow["b_to_a"]}</b><span>line crossings</span></div>
        <div class="metric"><b>{heavy_pct}%</b><span>time heavy</span></div>
      </div>
      {class_bar(meta["unique_vehicles_by_class"], meta["unique_vehicles_total"])}
      <p class="chart-label">Active vehicles over time</p>
      {density_svg(rows)}
      <p class="chart-label">Congestion timeline</p>
      {congestion_strip(rows)}
    </section>""")

    doc = f"""<title>Phuket Traffic Analytics — CCTV Vehicle Tracking</title>
<style>
  :root {{
    --bg: #12161c; --panel: #1a212a; --edge: #242d38;
    --text: #e8edf3; --muted: #8b96a5; --accent: #f5a623;
    font-size: 16px;
  }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 32px 24px 56px;
  }}
  .wrap {{ max-width: 1240px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; font-weight: 650; margin: 0; letter-spacing: -0.01em; }}
  .tagline {{ color: var(--muted); margin: 6px 0 0; font-size: 0.9rem; }}
  .tagline b {{ color: var(--text); font-weight: 600; }}
  .kpis {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 24px 0 28px; }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--edge); border-radius: 6px;
    padding: 14px 20px; min-width: 150px;
  }}
  .kpi b {{
    display: block; font-size: 1.7rem; font-weight: 650;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums; color: var(--accent);
  }}
  .kpi span {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--edge); border-radius: 8px;
    padding: 18px; display: flex; flex-direction: column; gap: 12px;
  }}
  .cardhead {{ display: flex; justify-content: space-between; align-items: start; }}
  .card h2 {{ margin: 0; font-size: 1.05rem; font-weight: 650; }}
  .sub {{ margin: 2px 0 0; color: var(--muted); font-size: 0.82rem; }}
  .state {{
    color: var(--c); border: 1px solid var(--c); border-radius: 99px;
    padding: 3px 12px; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .shot {{ width: 100%; border-radius: 5px; border: 1px solid var(--edge); }}
  .metrics {{ display: flex; gap: 10px; }}
  .metric {{
    flex: 1; background: var(--bg); border: 1px solid var(--edge);
    border-radius: 5px; padding: 10px 12px;
  }}
  .metric b {{
    display: block; font-size: 1.25rem; font-weight: 650;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums;
  }}
  .metric span {{ color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .mix {{ display: flex; height: 10px; border-radius: 3px; overflow: hidden; gap: 2px; }}
  .mixseg {{ min-width: 3px; }}
  .chips {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .chip {{ color: var(--muted); font-size: 0.78rem; display: inline-flex; align-items: center; gap: 5px; }}
  .chip i {{ width: 8px; height: 8px; border-radius: 2px; display: inline-block; }}
  .chart-label {{
    margin: 4px 0 -6px; color: var(--muted); font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.06em;
  }}
  .spark {{ width: 100%; height: 96px; display: block; }}
  .sparkmax {{ fill: #8b96a5; font-size: 11px; font-family: ui-monospace, Menlo, monospace; }}
  .strip {{ width: 100%; height: 10px; display: block; border-radius: 3px; }}
  .legend {{ display: flex; gap: 16px; margin-top: 26px; color: var(--muted); font-size: 0.8rem; }}
  .legend i {{ width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 5px; }}
  footer {{ color: var(--muted); font-size: 0.78rem; margin-top: 30px; border-top: 1px solid var(--edge); padding-top: 14px; }}
</style>
<div class="wrap">
  <h1>Phuket Traffic Analytics</h1>
  <p class="tagline">CCTV vehicle tracking · <b>{html.escape(model)}</b> + <b>{html.escape(tracker)}</b> · phuket-eye cameras, 20 s samples per site</p>
  <div class="kpis">
    <div class="kpi"><b>{total_unique}</b><span>unique vehicles</span></div>
    <div class="kpi"><b>{total_crossings}</b><span>line crossings</span></div>
    <div class="kpi"><b>{len(cams)}</b><span>cameras</span></div>
    <div class="kpi"><b>{html.escape(busiest_label)}</b><span>busiest site · {busiest_meta["unique_vehicles_total"]} vehicles</span></div>
  </div>
  <div class="grid">{"".join(cards)}
  </div>
  <div class="legend">
    <span><i style="background:#3fb96e"></i>Free</span>
    <span><i style="background:#f5a623"></i>Moderate</span>
    <span><i style="background:#e5484d"></i>Heavy</span>
  </div>
  <footer>Source: s3://chula-aigov-car-video-training-487984284636/01_traffic/phuket-eye/ · generated by track_analytics.py + build_dashboard.py · counts are unique track IDs and may include some ID-switch inflation in dense scenes.</footer>
</div>
"""
    with open(a.out, "w") as f:
        f.write(doc)
    print(f"wrote {a.out} ({os.path.getsize(a.out) // 1024} KB, {len(cams)} cameras)")


if __name__ == "__main__":
    main()
