# Bird's-Eye View from Fixed CCTV — Research & Implementation Plan

**Goal:** transform the side-angle Phuket CCTV views into top-down (BEV) lane
monitoring, like aerial footage. **Constraint:** monitoring output must never
hallucinate vehicles.
**Method:** deep-research sweep (5 angles, 103 agents; 9 claims survived 3-vote
adversarial verification before the session budget cut verification short — 15
more are cited below as *directional*), plus a working prototype on the actual
Chalong camera (`bev_view.py`).

---

## TL;DR — recommendation hierarchy

| Goal | Recommended technique | Why |
|---|---|---|
| (a) **Geometrically faithful live lane monitoring** ← *our use case* | **Planar homography / IPM** of the road plane + tracked-vehicle ground points rendered as metric footprints on a median-background base map | Exact for planar roads, zero hallucination, runs at tracking speed, deployed-industry standard. Reported accuracy of this classical family: **~0.8 m localization within 50 m of camera** (CAROM [1]) |
| (a+) Upgrade when orientation/height matters | **Roadside monocular 3D detection** (MonoUNI, CoBEV, HeightFormer family) | Direct BEV boxes with yaw from one fixed camera; Rope3D SOTA AP3D 92.45% (MonoUNI [5]); needs GPU + domain fine-tuning |
| (b) Prettier static-scene BEV visualization | **3D Gaussian Splatting** of the *static* scene (needs a one-time multi-view capture, e.g. phone/drone walkaround) + live vehicles composited from tracking | Photoreal base map; static-only so no vehicle hallucination |
| (c) Research-grade full novel view synthesis | Dynamic/4D-GS or NeRF variants | Not deployable from ONE fixed viewpoint; included for completeness only |

**Bottom line:** for this feature, homography IPM is not the fallback — it *is*
the state of the art for deployable single-fixed-camera BEV monitoring. Every
neural alternative either needs data we don't have (multi-view), or generates
content (unacceptable for monitoring).

## Why not the generative/NVS families (the honest physics)

A single fixed camera observes one ray per pixel. True novel views need either
more geometry (multi-view, parallax, depth) or a generative prior that *invents*
the unseen. Assessment of the candidate list (engineering assessment; the
NVS-specific claims were extracted but not fully verified before the budget cut):

| Family | Works from 1 fixed view w/ moving vehicles? | Hallucinates? | Verdict for monitoring |
|---|---|---|---|
| NeRF (incl. pixelNeRF/generalizable) | No — needs many poses; pixelNeRF generalizes but low-fidelity, static scenes | Interpolates unseen | ✗ |
| 3D Gaussian Splatting | No for dynamics from one view; yes for *static* scene with a capture pass | Low (reconstruction) | ✓ for (b) base map only |
| Dynamic/4D GS | Needs multi-view or moving camera for dynamics | Some | ✗ live, research-grade |
| Single-view 3D reconstruction | Per-object plausible shape, not scene-accurate | Yes (priors) | ✗ |
| Multi-View Stereo | Needs multiple views by definition | No | ✗ (no second camera) |
| Monocular depth (Depth Anything, Metric3D, UniDepth) | Yes — per-frame depth → unproject | No, but depth noisy on cars/road (metric scale drift) | possible assist; for a *planar road* homography is strictly more accurate |
| Multi-view diffusion (Zero-1-to-3, Instant3D) | Object-centric, generative | **Yes, by design** | ✗ |
| SDS/DreamFusion | Text/image-to-3D generative, minutes–hours per scene | **Yes** | ✗ |
| Native 3D/mesh diffusion | Generative | **Yes** | ✗ |
| SDF/implicit surfaces | Reconstruction backbone, needs multi-view | No | ✗ (data) |
| Voxel generation | Generative/reconstruction, coarse | Yes | ✗ |

## Verified findings (3-0 adversarial votes unless noted)

1. **CAROM** converts monocular roadside camera video into map-registered BEV
   vehicle data (type, 3D shape, position, velocity) — no scene hallucination;
   localization error **~0.8 m within 50 m**, 1.7 m within 120 m.
   [arxiv.org/abs/2104.00893]
2. **Calibration-free homography from satellite maps**: annotate the same
   landmarks in the CCTV image and Google Maps satellite view → road-plane
   homography in real-world metric coordinates, then detect in the IPM image.
   [arxiv.org/pdf/2103.15293] ← *this is our metric-calibration path*
3. IPM's stated limits: assumes **planar road + negligible lens distortion**;
   curved/non-planar roads are explicit future work. [same source]
4. **Vanishing-point auto-calibration** fails without clear parallel structures,
   degrades on curved urban geometry, needs periodic recalibration.
   [arxiv.org/pdf/2412.00348]
5. **TLCalib** (transformer roadside self-calibration) reaches **3.61 km/h mean
   speed error** on BrnoCompSpeed vs 8.59 for classical DubskaCalib — speed
   estimation from calibrated fixed CCTV is credible. [mdpi.com/1424-8220/23/23/9527]
6. **MonoUNI** is SOTA roadside monocular 3D detection: **Rope3D AP3D 92.45%**
   vs BEVHeight 74.60%. [arxiv.org/pdf/2412.00348]
7. **BEVHeight**: regressing height-to-ground (not depth) is the roadside-specific
   design (2-0). [github.com/ADLab-AutoDrive/BEVHeight]

*Directional (extracted, verification cut short):* CoBEV > BEVHeight on
DAIR-V2X-I and more robust to camera perturbation; HeightFormer (2024) > 
BEVHeight++ on Rope3D; AD-style BEVFormer transfers poorly to single fixed
cameras; BEV-supervised roadside detectors collapse on unseen scenes (domain
gap) — i.e., off-the-shelf weights won't just work on Phuket cameras.

## What we prototyped (working, in this repo)

`bev_view.py` — homography BEV for any camera with a `<camera>_bev` road-plane
quad in `camera_lines.json` (Chalong calibrated):

- **Base map** = IPM warp of the *median background* (vehicle-free road), so
  nothing above the road plane smears.
- **Vehicles** = tracked ground-contact points (bbox bottom-center) projected
  through the homography, drawn as metric footprints (car 1.8×4.5 m, bike
  0.8×2 m) with motion trails. Every marker corresponds to a real tracked
  detection — zero hallucination.
- Output: side-by-side source + BEV video (H.264). 12 s Chalong demo verified:
  lane-correct trails, parked row pinned at the kerb.

## Implementation plan

**P1 — metric calibration + all cameras (next)**
Annotate 4+ landmark pairs per camera against Google Maps satellite (verified
method [2]) → homography in true metres; add `_bev` quads for Sakhu + Kathu;
report positions in metres and derive **speeds (km/h)** per track (TLCalib
benchmarks say ~3–4 km/h error is achievable [5]).

**P2 — pipeline + dashboard integration**
`track_analytics.py --bev`: emit BEV coordinates per track into the frames CSV
and render `<stem>_bev.mp4`; PhuketFlow scene panel gets a BEV toggle per clip
(same DETECTED provenance; footnote: "positions homography-projected onto the
road plane; metric scale approximate until surveyed").

**P3 — optional upgrades**
(i) Fine-tune a roadside mono-3D detector (MonoUNI/CoBEV) on auto-labeled
Phuket data for yaw-accurate footprints — needs labels + GPU, and the domain-gap
finding says pretrained weights alone won't transfer. (ii) One-time multi-view
capture (phone walkaround at each site) → 3DGS static digital twin as a
photoreal BEV base map. (iii) Lens undistortion from the Hikvision intrinsics
to tighten IPM accuracy at frame edges.

**Rejected for this product:** NeRF/diffusion/SDS novel-view synthesis of the
live scene — generative pixels in a monitoring product violate the
no-hallucination constraint and none work from a single fixed viewpoint.
