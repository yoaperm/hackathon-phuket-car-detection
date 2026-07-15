#!/usr/bin/env python3
"""
Live traffic-analytics dashboard server. Reads per-clip analytics JSON from S3
and serves a dashboard whose <video> players stream the tracked videos straight
from S3 via presigned URLs (regenerated on every page load, so links never go
stale while the server runs).

Usage:
  python3 serve_dashboard.py                 # http://localhost:8899
  python3 serve_dashboard.py --port 8080 --prefix demo/analytics-ft/

Requires AWS credentials with read access to the bucket (env or ~/.aws).
"""
import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3

BUCKET = "chula-aigov-car-video-training-487984284636"

CAMERA_LABELS = {
    "chalong_c1_0700_demo30s": ("Chalong C1", "07:00 · morning rain"),
    "chalong_c1_2200_demo20s": ("Chalong C1", "22:00 · night"),
    "kathu_c10_1200_demo20s": ("Kathu C10", "12:00 · intersection"),
    "kathu_c10_2100_demo20s": ("Kathu C10", "21:00 · night"),
    "sakhu_c1_0800_demo20s": ("Sakhu C1", "08:00 · morning"),
    "sakhu_c1_1400_demo20s": ("Sakhu C1", "14:00 · afternoon"),
}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phuket Traffic Analytics — Live from S3</title>
<style>
  :root {{ --bg:#12161c; --panel:#1a212a; --edge:#242d38; --text:#e8edf3;
           --muted:#8b96a5; --accent:#f5a623; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); margin:0; padding:32px 24px 56px;
         font-family:-apple-system,"Segoe UI",system-ui,sans-serif; }}
  .wrap {{ max-width:1240px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; font-weight:650; margin:0; }}
  .sub {{ color:var(--muted); margin:6px 0 28px; font-size:.95rem; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:20px; }}
  .card {{ background:var(--panel); border:1px solid var(--edge); border-radius:10px;
          overflow:hidden; }}
  .card video {{ width:100%; display:block; background:#000; aspect-ratio:16/9; }}
  .meta {{ padding:14px 16px 16px; }}
  .meta h2 {{ font-size:1.05rem; margin:0; font-weight:600; }}
  .meta .when {{ color:var(--muted); font-size:.85rem; margin:2px 0 10px; }}
  .kpis {{ display:flex; gap:18px; flex-wrap:wrap; font-variant-numeric:tabular-nums; }}
  .kpi b {{ font-size:1.25rem; font-weight:650; display:block; }}
  .kpi span {{ color:var(--muted); font-size:.78rem; text-transform:uppercase;
              letter-spacing:.04em; }}
  .classes {{ margin-top:10px; color:var(--muted); font-size:.85rem; }}
  footer {{ color:var(--muted); font-size:.8rem; margin-top:28px; }}
</style></head><body><div class="wrap">
<h1>Phuket Traffic Analytics <span style="color:var(--accent)">· live from S3</span></h1>
<p class="sub">Model: {model} · tracker: ByteTrack · videos stream from
s3://{bucket}/{prefix} with URLs minted per page load</p>
<div class="grid">{cards}</div>
<footer>Unique-vehicle counts include some track-ID-switch inflation in dense
scenes. Refresh the page if a video stops loading (URLs expire after 1 h).</footer>
</div></body></html>"""

CARD = """<div class="card">
<video controls preload="metadata" src="{url}"></video>
<div class="meta"><h2>{title}</h2><div class="when">{when}</div>
<div class="kpis">
  <div class="kpi"><b>{unique}</b><span>unique vehicles</span></div>
  <div class="kpi"><b>{crossings}</b><span>line crossings</span></div>
  <div class="kpi"><b>{fps:.0f}s</b><span>clip length</span></div>
</div>
<div class="classes">{classes}</div>
</div></div>"""


def build_page(s3, prefix: str) -> str:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    stems = sorted(o["Key"][len(prefix):-len("_analytics.json")]
                   for o in resp.get("Contents", [])
                   if o["Key"].endswith("_analytics.json"))
    cards, model = [], "?"
    for stem in stems:
        data = json.loads(s3.get_object(Bucket=BUCKET,
                                        Key=f"{prefix}{stem}_analytics.json")["Body"].read())
        model = data.get("model", model)
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": BUCKET, "Key": f"{prefix}{stem}_tracked.mp4"},
            ExpiresIn=3600)
        title, when = CAMERA_LABELS.get(stem, (stem, ""))
        flow = data.get("line_crossings", {})
        classes = " · ".join(f"{v} {k}" for k, v in sorted(
            data.get("unique_vehicles_by_class", {}).items(), key=lambda x: -x[1]))
        cards.append(CARD.format(
            url=html.escape(url, quote=True), title=html.escape(title),
            when=html.escape(when), unique=data.get("unique_vehicles_total", "?"),
            crossings=flow.get("a_to_b", 0) + flow.get("b_to_a", 0),
            fps=data.get("duration_sec", 0), classes=html.escape(classes)))
    return PAGE.format(bucket=BUCKET, prefix=prefix, cards="".join(cards),
                       model=html.escape(str(model)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--prefix", default="demo/analytics-ft/")
    args = ap.parse_args()
    s3 = boto3.client("s3")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = build_page(s3, args.prefix).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):
            print(f"{self.client_address[0]} {fmt % a}", flush=True)

    print(f"dashboard: http://localhost:{args.port}  "
          f"(videos stream from s3://{BUCKET}/{args.prefix})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
