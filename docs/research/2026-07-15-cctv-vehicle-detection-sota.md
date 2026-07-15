# Vehicle Detection & Tracking on Fixed CCTV — State of the Art (2024–2026)

**Prepared for:** Phuket traffic-analytics stack (4K Hikvision angled street CCTV, day/night/rain; currently YOLOv8s-COCO + ByteTrack, batch GPU inference from S3)
**Method:** Deep-research sweep — 5 angles, 24 sources fetched, 115 claims extracted, top 25 adversarially verified by independent 3-voter panels. **18 claims confirmed** (cited below); 7 were refuted and are excluded. The commercial-platform angle produced sources but no claims that survived verification — that section is flagged as directional.

---

## TL;DR — what to change in the Phuket stack

1. **Move from YOLOv8s to YOLO11s.** A head-to-head vehicle benchmark (KITTI+BDD100K, incl. night/occlusion) puts YOLOv11s at 65.2% precision / 55.7% mAP@0.5 vs YOLOv8s at 64.8% / 52.1% — better accuracy with *fewer* parameters (9.4M vs 9.8M) [5]. A comprehensive YOLOv3→v12 evaluation names the YOLO11 family the best accuracy/efficiency balance [3]; YOLOv12's attention-heavy design adds cost without meaningful gains, so there is no reason to go past YOLO11 today [4].
2. **Fine-tune in-domain — it's the single biggest win available.** Detectors trained on highway data (UA-DETRAC) collapse to 33.6% mAP@0.50:0.95 on developing-city CCTV, vs 83.8% trained in-domain — a ~2.5× improvement [1]. The same cross-dataset gap shows up for transformers: RT-DETRv2 scores 0.474 off-domain vs 0.798 in-domain on the BMD-45 angled-CCTV benchmark [2]. This directly validates the auto-label → fine-tune pipeline (`finetune/`).
3. **Keep ByteTrack; BoT-SORT is the main alternative.** At CVPR 2024's AI City Challenge, most top teams used BoT-SORT, StrongSORT, ByteTrack, or ConfTrack paired with YOLO-family detectors — this is the de facto SOTA tracking stack for camera analytics [10]. YOLOv8+BoT-SORT is a published reference stack for CCTV traffic counting [13].
4. **Night is the biggest robustness gap, and fine-tuning fixes it.** COCO-pretrained YOLO frequently fails on fixed roadside cameras at night [15]; fine-tuning YOLO11 with day→night style-transfer augmentation lifted night mAP50 from 0.259 to 0.758 and recall from 0.246 to 0.883 [14]. The augmentation can be labeling-free: GAN style transfer reuses auto-generated daytime labels on synthesized night images [16].
5. **For small/distant vehicles in 4K frames, use SAHI tiled inference.** Top AI City 2024 teams used Slicing Aided Hyper Inference for small blurred objects and low-light enhancement (GSAD) for night [11].

---

## 1. Detector landscape

**YOLO11 is the current sweet spot.** The first comprehensive experimental evaluation across YOLOv3–v12 finds the YOLO11 family "consistently shows superior performance maintaining a remarkable balance of accuracy and efficiency" [3], and reports YOLOv12 as underwhelming — architectural complexity without significant gains [4]. On a vehicle-specific benchmark including nighttime and occluded scenes, YOLOv11s beats YOLOv8s on precision (65.2 vs 64.8) and mAP@0.5 (55.7 vs 52.1) with a smaller, faster model [5].

**RT-DETR is the credible transformer alternative.** RT-DETR-R50 reports 53.1% AP on COCO at 108 FPS on a T4 GPU (R101: 54.3% at 74 FPS), claimed to outperform contemporary YOLOs on both axes [6]. Being end-to-end (NMS-free) removes NMS post-processing, which the authors show degrades both YOLO speed and accuracy — relevant to latency budgeting in a batch pipeline [7]. Scaled-down RT-DETR variants are claimed competitive with YOLO S/M sizes [8].

**But leaderboard-topping accuracy is an ensemble game.** The 2024 AI City Challenge fisheye-camera detection track was won by an ensemble of CO-DETR + YOLOv9 + YOLOR-W6 + InternImage with pseudo-labels — not a single YOLO [9]. For a production pipeline, that's a signal that squeezing the last few mAP points costs disproportionate complexity; a fine-tuned single model is the right operating point.

**Verdict for Phuket:** YOLO11s as the default; RT-DETR worth a bake-off only after fine-tuning data exists, since the domain gap [1][2] dwarfs the architecture gap [5].

## 2. Domain adaptation: the 2.5× lever

The BMD-45 study (45 cameras, developing-city angled CCTV — the closest published analogue to Phuket's cameras) is unambiguous:

| Training data | mAP@0.50:0.95 on angled-CCTV test |
|---|---|
| UA-DETRAC (highway) | 33.6% |
| In-domain | **83.8%** (~2.5×) [1] |
| TrafficCAM → RT-DETRv2 | 0.474 |
| In-domain → RT-DETRv2 | **0.798** [2] |

Off-the-shelf COCO/highway weights leave most of the achievable accuracy on the table. This matches what we observed locally: the bundled aerial-view `best.pt` detected ~0 vehicles on street-view CCTV while COCO models found ~9/frame — viewpoint mismatch is fatal, and in-domain data is the cure.

## 3. Night, rain, and small objects

- COCO-pretrained YOLO "frequently struggled to distinguish vehicles from the background" on fixed roadside cameras at night, even in favorable lighting [15].
- Fine-tuning YOLO11 on day→night style-transferred data: night mAP50 0.259 → **0.758**, mAP50-95 0.160 → 0.559, recall 0.246 → **0.883** [14].
- The augmentation pipeline is labeling-free: an Efficient-Attention CycleGAN trained on real day images + CARLA-simulated night (with headlights) transfers style, and auto-generated daytime labels carry over unchanged [16] — i.e., our teacher-labeling approach extends to night without human annotation.
- Small/distorted objects: SAHI tiled inference was used by multiple top AI City 2024 teams; night robustness via GSAD low-light enhancement converting night frames to daylight-like input [11]. SAHI matters for 4K Phuket frames where distant vehicles are tens of pixels.
- A non-deep-learning analytical pipeline for nighttime volume estimation from a monocular camera exists as a fallback approach [17], but the fine-tuning results [14] make the neural path clearly stronger.
- Domain-specific YOLO modifications (e.g. CDS-YOLOv8: context-guided downsampling + dilated reparam blocks + Soft-NMS) claim +9 mAP@0.5 on UA-DETRAC [12] — evidence that architecture tweaks help, but again smaller than the in-domain-data lever.

## 4. Tracking

Most top teams in AI City 2024 single-camera tracking used **BoT-SORT, StrongSORT, ByteTrack, or ConfTrack** over YOLO-family detections [10]; YOLOv8+BoT-SORT is a published real-time traffic-counting stack [13]. Note the 2024 challenge had no dedicated vehicle-MOT track (its five tracks were people tracking, VLMs, driver action, fisheye detection, helmet violations) [9a], so treat "AI City SOTA" claims about vehicle MOT with that caveat.

**Verdict:** our YOLO11 + ByteTrack choice in `track_analytics.py` is squarely on the SOTA stack. BoT-SORT (ultralytics `botsort.yaml`) adds camera-motion compensation + appearance ReID — worth an A/B for ID-switch reduction in dense Kathu scenes, at some speed cost.

## 5. Platforms: build vs buy (directional — no claims survived verification)

The sweep fetched platform sources (NVIDIA DeepStream/Jetson traffic-analytics reference apps, Roboflow RTSP tooling, Hikvision solution pages, GoodVision, Miovision, Vivacity [18]) but none of the extracted commercial claims passed the 3-voter verification bar, largely because vendor accuracy/pricing claims aren't independently verifiable. Directionally: commercial offerings (GoodVision, Miovision, Vivacity) sell counting/congestion as managed services priced per camera or per video-hour, while the open-source path (ultralytics + supervision, or NVIDIA DeepStream for edge boxes) covers the same core analytics at hardware cost. For a hackathon-to-pilot trajectory with custom Thai-traffic behavior (motorbike dominance, mixed flow), **build** remains the right call; revisit **buy** only if procurement requires certified accuracy SLAs.

## 6. Mapped recommendations → current repo

| Recommendation | Where it lands |
|---|---|
| YOLO11s fine-tuned in-domain | `finetune/build_dataset.py` (teacher auto-label) + `finetune/train_phuket.py` |
| Keep ByteTrack, A/B BoT-SORT | `track_analytics.py` (`tracker="botsort.yaml"` switch) |
| Night: include night clips in training; later add day→night GAN augmentation | dataset already includes the 22:00 Chalong clip |
| SAHI tiling for distant vehicles | future: wrap teacher labeling & inference with `sahi` package |
| Don't chase YOLOv12 / ensembles | — |

## Sources (confirmed claims)

1,2. BMD-45 CCTV benchmark — https://arxiv.org/html/2604.24419
3,4. YOLOv3→v12 comprehensive evaluation — https://arxiv.org/pdf/2411.00201
5. Vehicle detection benchmark YOLOv8s vs v11s — https://pmc.ncbi.nlm.nih.gov/articles/PMC12158266/
6,7,8. RT-DETR — https://arxiv.org/pdf/2304.08069
9,9a,10,11. 8th AI City Challenge (CVPR 2024) — https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Wang_The_8th_AI_City_Challenge_CVPRW_2024_paper.pdf ; https://www.aicitychallenge.org/2024-challenge-winners/
12. CDS-YOLOv8 — https://www.mdpi.com/2079-9292/13/15/3033
13. Improved YOLOv8 + BoT-SORT counting — https://sciety.org/articles/activity/10.21203/rs.3.rs-4161504/v1
14,15,16. Night fine-tuning of YOLO11 with GAN day→night augmentation — https://arxiv.org/pdf/2412.16478
17. Analytical nighttime volume estimation — https://onlinelibrary.wiley.com/doi/10.1111/mice.13295
18. Platform sources (unverified): NVIDIA Jetson traffic analytics, Roboflow RTSP, Hikvision, GoodVision, Miovision, Vivacity — see report body.

*Refuted during verification (excluded): VisDrone RT-DETR-vs-YOLOv8 numbers, YOLOv11 UA-DETRAC 55.9% figure, MOT17 tracker MOTA comparison table, ByteTrack Kalman-mod mMOTA claim, and two others — their quoted numbers could not be found in the cited sources.*
