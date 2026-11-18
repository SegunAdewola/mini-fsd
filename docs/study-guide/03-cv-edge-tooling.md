# 3. CV models, metrics, edge deployment and tooling

<!-- toc -->
## Contents

- [1. Detection: YOLO26n and RF-DETR Nano](#1-detection-yolo26n-and-rf-detr-nano)
- [2. Segmentation: SegFormer-B0, EfficientViT-B0, frozen DINOv3](#2-segmentation-segformer-b0-efficientvit-b0-frozen-dinov3)
- [3. Monocular depth: Depth Anything V2 Small, Depth Anything 3 Small, distillation](#3-monocular-depth-depth-anything-v2-small-depth-anything-3-small-distillation)
- [4. Multi-task "HydraNet-lite"](#4-multi-task-hydranet-lite)
- [5. Metrics and measuring latency and memory](#5-metrics-and-measuring-latency-and-memory)
- [6. Edge deployment: ONNX, ONNX Runtime, FP16, INT8, CoreML, Docker caps](#6-edge-deployment-onnx-onnx-runtime-fp16-int8-coreml-docker-caps)
- [7. OCR for room signs](#7-ocr-for-room-signs)
- [8. Training infra: Explorer, Colab, Weights & Biases](#8-training-infra-explorer-colab-weights--biases)
- [9. Repo tooling and practices](#9-repo-tooling-and-practices)
<!-- tocstop -->

Facts checked on 2026-11-17. Versions on PyPI that day: ultralytics 8.4.215, rfdetr 1.12.1, transformers 5.23.0, torch 2.15.0, onnxruntime 1.32.0, coremltools 9.0, paddleocr 3.7.0, ruff 0.17.5, mypy 2.4.1, pytest 9.2.0, pre-commit 4.7.0, pixi 0.84.2. Pin what you install. These move fast.

---

## 1. Detection: YOLO26n and RF-DETR Nano

**Why it matters for Mini FSD.** This is E1 (obstacle, mover, door, room sign). First runs launch on Explorer in Week 10, and the winner is picked on real val in Week 11.

**Big picture.** Both models take an image and return boxes with class labels and scores. They get there in different ways.

YOLO26n is a one-stage CNN detector. A convolutional backbone makes feature maps at a few scales. A head predicts, at every grid cell, "is there an object here, what class, and where is the box." That gives thousands of candidate boxes. Classic YOLO then runs non-maximum suppression (NMS) to drop duplicates. YOLO26 adds an optional one-to-one head trained to predict each object once, so you can skip NMS.

RF-DETR Nano is a DETR-style transformer detector. A ViT backbone (DINOv2) makes features. A small transformer decoder holds a fixed set of "object queries." Each query attends to the image and outputs one box or "no object." Training matches predictions to ground truth one-to-one (Hungarian matching), so there are no duplicates and no NMS. Roboflow used neural architecture search to pick the Nano size.

The size gap matters for the edge budget. YOLO26n has 2.4M parameters and 5.5 GFLOPs at 640 px. RF-DETR Nano has 30.5M parameters at 384 px. RF-DETR scores higher on COCO (48.4 AP vs 40.9), and it is fast on an NVIDIA T4 with TensorRT. On a CPU the parameter count shows up. That is why the design doc says the comparison must be measured on the M1.

**In the weeds.**

YOLO26n fine-tune. Ultralytics trains on YOLO text labels, so convert your COCO JSON once:

```python
from ultralytics.data.converter import convert_coco
convert_coco(labels_dir="data/sim/annotations/", save_dir="data/sim_yolo/",
             cls91to80=False)  # must be False for custom classes
```

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")                     # COCO-pretrained
model.train(data="minifsd.yaml", epochs=100, imgsz=640, batch=32,
            device=0, project="runs/e1", name="yolo26n_sim", seed=0)
model.export(format="onnx", imgsz=640, nms=False)   # end-to-end head
```

`minifsd.yaml` lists `path`, `train`, `val`, and `names: {0: obstacle, 1: mover, 2: door, 3: room_sign}`. Interrupted job? `YOLO("runs/e1/yolo26n_sim/weights/last.pt").train(resume=True)`.

RF-DETR Nano fine-tune. It reads COCO directly. The folder needs `train/_annotations.coco.json`, `valid/_annotations.coco.json`, and optionally `test/`, with images next to each JSON.

```bash
pip install "rfdetr[train,onnx]==1.12.1"
```

```python
from rfdetr import RFDETRNano
model = RFDETRNano()
model.train(dataset_dir="data/sim_coco", epochs=50, batch_size=16,
            grad_accum_steps=1, lr=1e-4, output_dir="runs/e1/rfdetr_nano")
model.export()   # writes output/inference_model.onnx
```

Outputs include `checkpoint_best_total.pth` (use this for inference), `last.ckpt` (for resuming), and `training_config.json`. The ONNX input is named `input`, NCHW, with ImageNet mean/std normalization. RF-DETR can also export INT8 through ONNX Runtime with `model.export(format="onnx", quantization="int8", calibration_data="calib_images/")`.

**Gotchas.**
- Name check. YOLO26 is real and current (released January 2026, paper arXiv 2606.03748). The docs now say the **default** path is the one-to-many head with NMS. You get the NMS-free head with `nms=False` at predict, val, and export. Early YOLO26 posts said end-to-end by default. Pick one and use it for both accuracy and latency.
- The YOLO26 family now also ships `yolo26n-sem` (semantic segmentation) and `yolo26n-depth` (monocular depth). These could be cheap extra candidates for E2 and E3, but that is scope creep. A YOLO27 preview page exists. It is not released.
- License. Ultralytics code and weights are AGPL-3.0. RF-DETR Nano through Large and the `rfdetr` package are Apache 2.0. The XL/2XL and the tiny Atto/Femto/Pico models are under a Roboflow platform license. Fine for a class project, worth one line in the report.
- `rfdetr` 1.12.0 (Nov 5, 2026) changed `inference()` and `export()` behavior. Copy examples from the docs version you pinned.
- On the M1, `batch_size="auto"` only works on CUDA. Use an integer on CPU or MPS. Don't train on the M1 anyway. Train on Explorer.
- The NMS-free YOLO output is `(N, 300, 6)`. The NMS head output is `(N, 4 + nc, 8400)` and needs your own NMS outside the model. Time the NMS too.

**Try it (15 min).** On the M1: `pip install ultralytics`, then `yolo predict model=yolo26n.pt source=some_photo.jpg`. Export with `yolo export model=yolo26n.pt format=onnx nms=False`, open the file in Netron (netron.app), and find the output shape.

**Read.**
- Ultralytics YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
- Ultralytics export modes: https://docs.ultralytics.com/modes/export/
- RF-DETR training docs: https://rfdetr.roboflow.com/latest/learn/train/
- RF-DETR paper (ICLR 2026): https://arxiv.org/abs/2511.09554

---

## 2. Segmentation: SegFormer-B0, EfficientViT-B0, frozen DINOv3

**Why it matters for Mini FSD.** E2 floor segmentation, Week 11. The E2 winner that fits the budget becomes the E4 HydraNet backbone in Week 12.

**Big picture.** Floor segmentation labels each pixel "floor" or "not floor." All three candidates share a pattern: an encoder (backbone) turns the image into a grid of feature vectors, and a decoder (head) turns those into per-pixel class scores at lower resolution, which you upsample.

- **SegFormer-B0.** A small hierarchical transformer encoder (MiT-b0) plus an all-MLP decoder. About 3.7M parameters. Easy to fine-tune with Hugging Face.
- **EfficientViT-B0.** MIT Han Lab's design that replaces softmax attention with cheaper linear attention, built for edge hardware. Under 1M parameters for the segmentation model.
- **Frozen DINOv3 ViT-S/16 + light head.** DINOv3 is Meta's self-supervised ViT (August 2025). "Frozen backbone" means you never update its weights. You run it, take the patch features, and train only a tiny head on top: a 1x1 conv (a "linear probe") or two conv layers. Training is fast and cheap because gradients only touch the head. The bet is that features learned from 1.7B web images generalize from sim to real better than features fine-tuned on sim.

**In the weeds.**

SegFormer-B0 with 2 classes:

```python
from transformers import SegformerForSemanticSegmentation
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels=2,
    id2label={0: "other", 1: "floor"}, label2id={"other": 0, "floor": 1})
out = model(pixel_values=x, labels=y)   # out.loss, out.logits at H/4, W/4
```

`nvidia/mit-b0` is the ImageNet-pretrained encoder only. The decoder starts random, which is what you want.

EfficientViT-B0. The Hugging Face Transformers library does not include it. Two routes:
- ImageNet backbone from timm: `timm.create_model("efficientvit_b0.r224_in1k", pretrained=True, features_only=True)`, then add your own small head.
- Full segmentation model from the official repo. B0 segmentation weights exist only for Cityscapes (`efficientvit_seg_b0_cityscapes.pt` on `han-cai/efficientvit-seg`).

The design says "ImageNet-pretrained," so the timm route matches it and is simpler.

Frozen DINOv3 with a linear head:

```python
import torch, torch.nn as nn
from transformers import AutoModel
bb = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").eval()
for p in bb.parameters():
    p.requires_grad = False

class LinearHead(nn.Module):
    def __init__(self, dim=384, n_cls=2):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, n_cls, 1)
    def forward(self, tokens, h, w):
        x = tokens[:, 1 + 4:]                      # drop CLS + 4 register tokens
        x = x.transpose(1, 2).reshape(x.shape[0], -1, h // 16, w // 16)
        return self.conv(self.bn(x))

with torch.no_grad():
    tokens = bb(pixel_values=img).last_hidden_state
logits = head(tokens, H, W)    # upsample to H, W before the loss
```

Since the backbone is frozen, you can cache its features to disk once and train the head in minutes. Check the token layout (1 CLS + 4 registers + patches) against the model card before trusting the slice.

**Gotchas.**
- DINOv3 is **gated** on Hugging Face. Click "agree" on the model page with your account and run `huggingface-cli login` on Explorer and the M1. Approval can take time, so request it now.
- DINOv3 uses the custom **DINOv3 License**, not Apache. It allows research use and has use restrictions (for example, military uses). Say so in the report. Transformers needs 4.56 or newer.
- `nvidia/mit-b0` and the SegFormer checkpoints carry NVIDIA's own license ("other" on the Hub, research use). Fine for a course.
- The EfficientViT repo says it is no longer maintained (since September 2025). Use timm for the backbone.
- ViT-S/16 needs input sides that are multiples of 16. Token count grows with resolution squared. At 512x512 that is 1,024 patches, which hurts CPU latency more than the parameter count suggests.
- Report binary floor IoU, not mIoU over 2 classes. "Other" IoU is always high and hides problems.

**Try it (20 min).** Load DINOv3 ViT-S/16 on the M1. Run one photo of a hallway. Take the patch tokens, run PCA to 3 components, reshape to the patch grid, and show it as an RGB image. The floor should come out as one color with no training.

**Read.**
- SegFormer in Transformers: https://huggingface.co/docs/transformers/model_doc/segformer
- DINOv3 repo and license: https://github.com/facebookresearch/dinov3
- DINOv3 ViT-S/16 model card: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- EfficientViT repo: https://github.com/mit-han-lab/efficientvit

---

## 3. Monocular depth: Depth Anything V2 Small, Depth Anything 3 Small, distillation

**Why it matters for Mini FSD.** E3 in Week 11 (zero-shot on NYUv2 and sim), and the teacher for the distilled depth head in E4, Week 12.

**Big picture.** A monocular depth model guesses distance per pixel from one image. Most foundation models predict **relative** depth: correct ordering and shape, but unknown scale (and sometimes unknown shift). A wall 2 m away and a wall 4 m away can look the same in one image. **Metric** depth is in meters. A robot needs meters to build a costmap, so you either use a metric model or recover the scale yourself.

Your robot gives you a cheap way to recover scale: you know the camera height above the floor. Floor pixels must lie on a plane exactly that far below the camera. Compare the model's floor to the true floor and you get the scale.

Distillation means using a big, slow, accurate "teacher" to label lots of images, then training a small, fast "student" to copy it. Here the student is a depth head on your shared backbone.

**In the weeds.**

Depth Anything V2 Small (24.8M params, Apache 2.0) works through the Transformers pipeline:

```python
from transformers import pipeline
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
out = pipe(pil_image)
pred = out["predicted_depth"]   # relative, larger = closer (inverse depth)
```

The V2 relative models output affine-invariant inverse depth (disparity-like). Align in inverse-depth space, then invert.

Depth Anything 3 (ByteDance Seed, released 2025-11-14, ICLR 2026). DA3-SMALL is 0.08B params, Apache 2.0, and predicts relative depth (plus pose for multi-view input). It is not in Transformers. Use the official package:

```python
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to("cpu")
pred = model.inference(["frame.png"])
depth = pred.depth[0]          # [H, W] float32, relative scale
```

Scale recovery from camera height. With a level camera (zero pitch) at height `h`, a floor pixel at image row `v` is at depth `Z = fy * h / (v - cy)` for rows below the horizon. Then:

```python
floor = floor_mask & (rows > cy + 5)                # from your segmenter
z_geo = fy * cam_h / (rows[floor] - cy)
scale = np.median(z_geo / depth_rel[floor])         # for depth-like output
depth_m = scale * depth_rel
```

If the camera is pitched, fit a plane to the back-projected floor points instead and use `scale = cam_h / plane_distance`. Median is used because it shrugs off a few bad floor pixels.

Distillation plan:
1. Run the teacher (the E3 winner) on all 10k sim frames plus your unlabeled real photos. Save its depth maps.
2. Train the student depth head on teacher outputs with a scale-and-shift-invariant loss (align each prediction to the target with least squares, then take L1). On sim frames you also have exact depth, so you can add a metric loss there.
3. Evaluate the student with the same alignment as the teacher, so the comparison is fair.

**Gotchas.**
- Licenses differ by size. DA V2: Small is Apache 2.0, Base and Large are CC BY-NC 4.0. DA3: Small, Base, Metric-Large, and Mono-Large are Apache 2.0. Large and Giant are CC BY-NC 4.0. The design picks the Small models, which are the clean ones.
- **DA3-SMALL is relative only.** The only DA3 metric models are DA3METRIC-LARGE (convert with `focal * output / 300`) and DA3NESTED-GIANT-LARGE (non-commercial). Neither fits an 8 GB edge budget well. Plan on camera-height scale recovery for both Small models.
- The DA3 package pins `numpy<2`, Python ≤ 3.13, and pulls heavy extras (open3d, pycolmap). Install it in its own pixi environment, export to ONNX once, and keep the main env clean. Whether its ONNX export works cleanly is unverified. Test it early in Week 11.
- Larger DA3 Large/Giant models have `-1.1` retrained checkpoints. The originals are deprecated. Small is unchanged.
- NYUv2 eval convention: cap depth at 10 m and use the standard Eigen crop. Say which alignment (median scale vs least-squares scale and shift) you used. Numbers are not comparable otherwise.

**Try it (15 min).** Run DA V2 Small on a photo of your hallway with the pipeline above. Measure your camera height. Hand-pick 20 floor pixels, apply the formula, and check one distance with a tape measure.

**Read.**
- Depth Anything 3 repo (model table, licenses): https://github.com/ByteDance-Seed/Depth-Anything-3
- DA3-SMALL card: https://huggingface.co/depth-anything/DA3-SMALL
- Depth Anything V2 Small card: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- NYU Depth V2 dataset: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

---

## 4. Multi-task "HydraNet-lite"

**Why it matters for Mini FSD.** E4, Week 12. It answers the report's "Tesla question": does sharing one backbone cost accuracy, and how much latency does it save?

**Big picture.** Running three separate models means three backbones. The backbone is most of the compute. A multi-task network runs one backbone and attaches several small heads: floor mask, detection, depth. You pay for the backbone once per frame.

Tesla calls its version HydraNet. Karpathy's reasons, from his 2019 PyTorch talk and AI Day 2021: the car cannot run a backbone per task (they had dozens of tasks), sharing amortizes compute, and teams can fine-tune heads on cached backbone features without retraining everything. The cost is that tasks can fight over the shared features. One task's gradients can hurt another.

**In the weeds.**

```python
class HydraLite(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.floor = nn.Conv2d(dim, 2, 1)
        self.depth = nn.Sequential(nn.Conv2d(dim, 64, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(64, 1, 1))
        self.det = DetHead(dim, n_cls=4)      # e.g. a small anchor-free head
    def forward(self, x):
        f = self.backbone(x)
        return {"floor": self.floor(f), "depth": self.depth(f), "det": self.det(f)}
```

Loss weighting. You need one number to backprop: `L = w1*L_floor + w2*L_depth + w3*L_det`.
- **Fixed weights.** Start by scaling each loss to roughly the same size at step 0, then tune by hand on val. Simple and often good enough.
- **Uncertainty weighting** (Kendall, Gal, Cipolla 2018). Learn one log-variance `s_i` per task and use `L = Σ exp(-s_i) * L_i + s_i`. Noisy or hard tasks get down-weighted automatically, and the `+ s_i` term stops the weights from going to zero.

```python
self.log_vars = nn.Parameter(torch.zeros(3))
loss = sum(torch.exp(-s) * l + s for s, l in zip(self.log_vars, losses))
```

Fair comparison for E4: same backbone, same input size, same data, same epochs as the specialists. Report accuracy per task and total latency for "3 specialists in sequence" vs "1 HydraNet."

Cheap trick from Tesla: freeze the trained backbone, cache features, and train extra heads fast. If the frozen DINOv3 wins E2, this falls out for free.

**Gotchas.**
- Detection heads are fiddly. If a custom head eats your week, a fallback is a HydraNet with floor + depth heads and the detector kept separate. Report it honestly.
- Different tasks may want different input resolutions. Pick one and say so.
- A shared ONNX graph with three outputs is one session. Your latency harness must handle dict or multiple outputs.

**Try it (15 min).** Take any pretrained timm backbone. Attach two 1x1 conv heads. Time the backbone alone vs backbone + heads on the M1 at batch 1. The heads should be a small fraction of the total. That fraction is the whole argument for sharing.

**Read.**
- Kendall et al., uncertainty weighting: https://arxiv.org/abs/1705.07115
- Karpathy, "PyTorch at Tesla" (2019): https://www.youtube.com/watch?v=oBklltKXtDE
- Ruder, overview of multi-task learning: https://www.ruder.io/multi-task/

---

## 5. Metrics and measuring latency and memory

**Why it matters for Mini FSD.** The shared eval harness is a Week 10 deliverable. Every E1–E6 number goes through it.

**Big picture.** An accuracy number is only useful if it is computed the standard way, and a latency number is only useful if it is measured the same way every time. Write the harness once, test it, and run every model through it.

**In the weeds.**

**IoU / mIoU.** IoU = overlap / union. Accumulate a confusion matrix over the whole split, then compute IoU from the totals. Averaging per-image IoU gives a different number and overweights easy images.

```python
cm += np.bincount(2 * gt.ravel() + pred.ravel(), minlength=4).reshape(2, 2)
iou_floor = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0])
```

**mAP@0.5.** For each class, sort detections by score, mark each as a true positive if it overlaps an unmatched ground truth box with IoU ≥ 0.5, build the precision-recall curve, take the area. Average over classes. COCO's main metric averages over IoU 0.5 to 0.95. Report both. Use a library, not your own code:

```python
from torchmetrics.detection import MeanAveragePrecision
m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
m.update(preds=[{"boxes": b, "scores": s, "labels": l}],
         target=[{"boxes": gb, "labels": gl}])
r = m.compute()   # r["map_50"], r["map"], per class with class_metrics=True
```

The default backend is pycocotools. `backend="faster_coco_eval"` is faster and matches it. Run YOLO and RF-DETR predictions through this same code. Don't compare Ultralytics' built-in val number to RF-DETR's built-in number.

**Depth.** With prediction `d` and ground truth `d*` on valid pixels:
- AbsRel = mean(|d − d*| / d*). Lower is better.
- δ<1.25 = fraction of pixels where max(d/d*, d*/d) < 1.25. Higher is better.

For relative models, align first and report which alignment you used.

**OCR.** Exact match after normalizing (uppercase, strip spaces and punctuation) is the honest metric for room numbers. "214" read as "21" is a wrong room. Add character error rate as a secondary number. Report two cases: OCR on ground-truth crops, and OCR on detector crops (end to end).

**Latency done right.**
- Fix the input size and use batch 1.
- Set threads before creating any model: `torch.set_num_threads(2)`, `torch.set_num_interop_threads(1)`. Also set `OMP_NUM_THREADS=2` in the environment.
- For ONNX Runtime:

```python
so = ort.SessionOptions()
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
sess = ort.InferenceSession("m.onnx", so, providers=["CPUExecutionProvider"])
```

Spinning threads look fast in microbenchmarks but burn CPU the planner needs. Turn spinning off for the budget runs.
- Warm up 20–50 runs, then time 300+ runs with `time.perf_counter()`. Report p50 and p95 (`np.percentile(t, [50, 95])`). p95 is what the robot feels.
- Time preprocessing, inference, and postprocessing (NMS, upsampling) separately and also together.

**Peak memory.** Measure each model in a fresh subprocess. `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` gives peak RSS. Inside the Docker budget container, read `docker stats` or the cgroup file `/sys/fs/cgroup/memory.peak`.

**Gotchas.**
- `ru_maxrss` is in **bytes on macOS** and **kilobytes on Linux**. Your harness runs on both. Handle it.
- macOS has no public API to pin a thread to a core. ORT's affinity setting is not something to rely on there. The M1 has 4 performance and 4 efficiency cores, and the scheduler picks. Run on power, close other apps, repeat each measurement three times, and report the spread.
- `tracemalloc` only sees Python allocations, not ONNX Runtime or PyTorch native memory. Don't use it for this.
- Thermal throttling is mild on an M1 Air but real in long runs. Interleave models (A, B, A, B) rather than running all of A then all of B.

**Try it (15 min).** Write `bench(fn, warmup=30, n=300)` that returns p50 and p95. Run it on the YOLO26n ONNX at 1, 2, and 4 intra-op threads. Plot latency vs threads. Note where it stops helping.

**Read.**
- torchmetrics mAP: https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
- COCO API (pycocotools): https://github.com/cocodataset/cocoapi
- ONNX Runtime threading: https://onnxruntime.ai/docs/performance/tune-performance/threading.html

---

## 6. Edge deployment: ONNX, ONNX Runtime, FP16, INT8, CoreML, Docker caps

**Why it matters for Mini FSD.** E6 compression and the final latency table are Week 12. The Docker budget profiles come from the Nov 10 update, and the Dec 15 demo script (ONNX Runtime CPU, offline) is Week 13.

**Big picture.** ONNX is a file format for a model graph. You export once from PyTorch, then run it with ONNX Runtime anywhere, without PyTorch. Compression makes the model smaller or faster:
- **FP16**: weights and math in 16-bit floats. Half the size. Speedup depends on hardware.
- **INT8**: weights (and maybe activations) in 8-bit integers. About a quarter of the size, usually faster on CPU, some accuracy loss.

CoreML is Apple's runtime. It can place parts of a model on the CPU, the GPU, or the Apple Neural Engine (ANE), a dedicated low-power accelerator.

**In the weeds.**

Export and check:

```python
torch.onnx.export(model.eval(), torch.randn(1, 3, 384, 640), "m.onnx",
                  input_names=["input"], output_names=["out"], opset_version=17)
```

Then compare against PyTorch on 10 real images: `np.testing.assert_allclose(ort_out, torch_out, rtol=1e-3, atol=1e-4)`. Use fixed shapes for the edge models. Dynamic shapes slow down both ORT and CoreML. Recent PyTorch versions use the newer dynamo-based exporter. If export fails on a model, try passing `dynamo=False`.

**Dynamic vs static INT8.**
- Dynamic: weights quantized ahead of time, activation ranges computed at run time. No calibration data. ORT recommends it for transformers.
- Static: activation ranges fixed ahead of time from a calibration set. ORT recommends it for CNNs. Usually faster.

```python
from onnxruntime.quantization import (quantize_static, quantize_dynamic,
    CalibrationDataReader, QuantFormat, QuantType)
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process("m.onnx", "m_pre.onnx")

class Reader(CalibrationDataReader):
    def __init__(self, arrays):
        self.it = iter([{"input": a} for a in arrays])
    def get_next(self):
        return next(self.it, None)

quantize_static("m_pre.onnx", "m_int8.onnx", Reader(calib_arrays),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
```

Calibrate on 100–300 preprocessed frames: a mix of sim and real **val**, never test. If accuracy drops a lot, exclude the first and last layers with `nodes_to_exclude`, or try percentile calibration.

FP16 for ONNX: `onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True)`. Measure it on the M1 CPU. Don't assume it is faster. FP16 pays off mostly on GPU and ANE.

CoreML two ways:
1. ORT with the CoreML execution provider:

```python
sess = ort.InferenceSession("m.onnx", providers=[
    ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram",
     "MLComputeUnits": "CPUAndNeuralEngine", "RequireStaticInputShapes": "1"}),
    "CPUExecutionProvider"])
```

Ops CoreML can't handle fall back to CPU, which can make it slower than pure CPU. Set `"ProfileComputePlan": "1"` to log where each op ran.

2. coremltools directly:

```python
import coremltools as ct
ml = ct.convert(torch.jit.trace(model.eval(), example),
                inputs=[ct.TensorType(name="input", shape=example.shape)],
                convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_AND_NE)
ml.save("m.mlpackage")
```

ML Program models default to FP16 compute.

Docker caps:

```bash
docker run --rm --cpus=4 --memory=4g --memory-swap=4g --cpuset-cpus=0-3 \
  -e OMP_NUM_THREADS=2 minifsd-robot:latest
```

**Gotchas.**
- **Docker on macOS runs inside a Linux VM.** The VM's CPU and memory (Docker Desktop, Settings → Resources) are the ceiling. `--cpus` and `--memory` then limit the container inside that VM. Set the VM to at least the largest profile (6 cores / 8 GB will not fit on an 8 GB Mac; the VM itself needs headroom, so that profile may be impossible locally).
- `--cpus=2` is a time quota. The container still **sees** all VM CPUs, so `os.cpu_count()` and ORT's default thread count ignore the cap and oversubscribe. Always set thread counts explicitly. `--cpuset-cpus` changes what the container sees.
- Set `--memory-swap` equal to `--memory`, or the container can swap past the cap.
- **The ANE is not reachable from inside Docker.** The CoreML path only runs natively on macOS. So the CoreML numbers are outside the Docker budget profiles. Report them as a separate column.
- coremltools 9.0 (Nov 2025) is still the latest stable release and has wheels up to Python 3.13. It is tested against older PyTorch than 2.15. Expect a warning. Pin a torch version it lists as tested if conversion breaks.
- Apple's docs say W8A8 (int8 weights and activations) speeds up the ANE on A17 Pro and M4 and newer. On an **M1**, expect INT8 in CoreML to save size and memory but not compute time. That is a fine finding for E6.
- onnxruntime 1.32 macOS wheels need macOS 14 or newer. Your macOS 26 is fine.

**Try it (20 min).** Take the YOLO26n ONNX. Make dynamic and static INT8 versions (static with 50 of your sim frames). Benchmark all three on 2 threads. Compare file size, p50, and mAP@0.5 on 100 val images.

**Read.**
- ORT quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ORT CoreML execution provider: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- coremltools optimization overview: https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
- Docker resource constraints: https://docs.docker.com/engine/containers/resource_constraints/

---

## 7. OCR for room signs

**Why it matters for Mini FSD.** The sign reader is a Week 12 "if on track" item. It turns a detected room-sign box into a room number for the mission node.

**Big picture.** OCR usually has two stages: detect where text is, then recognize the characters. Your detector already finds the sign, so you mostly need the recognition stage on a crop. Room numbers are short (3–4 characters), printed, mostly digits. That is the easy end of OCR. The hard parts are blur, angle, and small crops.

**In the weeds.**

PaddleOCR 3.7.0 (June 2026) is current. It added **PP-OCRv6** in tiny (1.5M), small (7.7M), and medium (34.5M) sizes, and v6 is now the default. For crops, call the recognizer only:

```python
from paddleocr import TextRecognition
rec = TextRecognition(model_name="PP-OCRv6_tiny_rec")   # or en_PP-OCRv5_mobile_rec
for r in rec.predict(input=crop_bgr, batch_size=1):
    text, score = r["rec_text"], r["rec_score"]
```

If the sign crop holds a whole plate with more than one text line, use the full pipeline and turn the extras off:

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False)
res = ocr.predict(crop)[0]     # res["rec_texts"], res["rec_scores"]
```

Then post-process with your building's room list. Match against valid room numbers (for example edit distance ≤ 1), and reject if the score is low. This fixes many errors cheaply.

Alternatives worth a quick look:
- **RapidOCR**: PaddleOCR models converted to ONNX, run with ONNX Runtime. Same runtime as the rest of your stack, no PaddlePaddle install.
- **Apple Vision** through `ocrmac`: very good on printed text and uses Apple's own accelerators. macOS only, so it won't run in the Docker profile.
- **Tesseract**: easy, but weak on small, angled crops.
- A tiny digit classifier trained on your sim signs. You render the signs, so labels are free.

**Gotchas.**
- `paddlepaddle` 3.3.1 has macOS arm64 wheels. It is a large dependency. Put it in its own pixi feature, or use RapidOCR.
- The first run downloads models. Pre-download them so the Dec 15 demo works offline.
- Upscale tiny crops (to ~48 px text height) before recognition. Pad a little around the box.
- PP-OCRv6's CPU speed claims use OpenVINO on Intel. Measure on the M1.

**Try it (15 min).** Take five phone photos of real room signs at different angles. Crop by hand. Run `PP-OCRv6_tiny_rec` and `en_PP-OCRv5_mobile_rec`. Compare exact-match and time per crop.

**Read.**
- PaddleOCR repo and releases: https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR text recognition module: https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html
- RapidOCR: https://github.com/RapidAI/RapidOCR
- ocrmac (Apple Vision from Python): https://github.com/straussmaximilian/ocrmac

---

## 8. Training infra: Explorer, Colab, Weights & Biases

**Why it matters for Mini FSD.** First E1 jobs go to Explorer in Week 10. E1–E5 training runs there unattended in Weeks 10–12.

**Big picture.** Explorer is Northeastern's shared GPU cluster. You log in to a login node, which is for editing and submitting only. You write a batch script that says what resources you need and what to run, and SLURM (the scheduler) runs it on a GPU node when one frees up. Colab is the backup for quick or interactive work. W&B records metrics, configs, and checkpoints from every run in one web dashboard, so you can compare runs later.

**In the weeds.**

Partition limits from RC's partition page (per user):

| Partition | Default / max time | Running jobs | GPUs per job |
|---|---|---|---|
| gpu-short | 1 h / 2 h | 2 | 1 |
| gpu-interactive | 1 h / 2 h | 2 | 1 |
| gpu | 4 h / 8 h | 4 | 1 |
| courses-gpu | 4 h / 8 h | 1 | 1 (12 GB RAM) |
| multigpu | needs approval | | |

A job array script, one task per seed:

```bash
#!/bin/bash
#SBATCH --job-name=e1-yolo26n
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out

module load anaconda3/2024.06
source activate /projects/<group>/envs/minifsd   # or your /courses student dir
export WANDB_DIR=/scratch/$USER/wandb
python train_e1.py --model yolo26n --seed $SLURM_ARRAY_TASK_ID
```

Commands you will use daily: `sbatch job.sh`, `squeue -u $USER`, `sacct -j <id>`, `scancel <id>`, `sinfo -p gpu` (see free nodes and GPU types), and `srun --partition=gpu-interactive --gres=gpu:1 --time=01:00:00 --pty bash` for a quick interactive shell. Request a specific GPU type with `--gres=gpu:<type>:1`, but expect longer waits.

W&B basics:

```python
import wandb
run = wandb.init(project="minifsd", group="E1", name=f"yolo26n-s{seed}",
                 config=vars(args))
wandb.log({"val/map50": m50, "epoch": ep})
run.finish()
```

Ultralytics logs to W&B after `yolo settings wandb=True`. Run `wandb login` once on the login node. Name runs by experiment and candidate. Use `group` for E1–E6 so the dashboard filters cleanly.

Colab: mount Google Drive and write checkpoints there. Sessions get killed. Use it for the DINOv3 head on cached features or for quick checks, not for 8-hour jobs.

**Gotchas.**
- **8-hour max on `gpu`.** Save checkpoints every epoch and make every training script resumable (`resume=True` in Ultralytics; RF-DETR's `last.ckpt`). Chain jobs with `--dependency=afterany:<jobid>` if needed.
- RC asks you to remove the `conda init` block from `~/.bashrc` and activate inside the job. Keep environments out of `/home` (small quota). Use `/projects` or your `/courses/<course>/students/<user>` folder.
- `courses-gpu` allows **one running job**. If your course has access, it won't run an array in parallel. `gpu` allows 4.
- Pixi vs conda on Explorer: pixi installs per-user without modules and its lockfile reproduces your env. Set `PIXI_CACHE_DIR` to scratch to avoid the home quota. The cluster needs a `linux-64` env with CUDA PyTorch. Your M1 needs `osx-arm64` CPU PyTorch. Use separate pixi features per platform.
- If compute nodes can't reach the internet, use `WANDB_MODE=offline` and run `wandb sync` from the login node. Pre-download weights (YOLO26n, DINOv3, DA3) on the login node.
- Scratch is purged on a schedule. Copy final checkpoints elsewhere.
- Test the script on `gpu-short` with 1 epoch before submitting the 8-hour array. `sbatch --test-only job.sh` checks the request without queuing it.

**Try it (15 min).** Submit a 5-minute `gpu-short` job that runs `nvidia-smi` and `python -c "import torch; print(torch.cuda.get_device_name())"`. Then run it as `--array=0-1` and confirm two log files with different `%a`.

**Read.**
- RC GPU access: https://rc-docs.northeastern.edu/en/latest/gpus/accessinggpus.html
- RC partitions and limits: https://rc.northeastern.edu/partitions
- RC job arrays: https://rc-docs.northeastern.edu/en/explorer-main/runningjobs/slurmarray.html
- RC conda on Explorer: https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/conda.html

---

## 9. Repo tooling and practices

**Why it matters for Mini FSD.** Set up in Week 9–10 and used every week. These are the "engineering practices" signals in the design doc. They also keep the eval harness trustworthy.

**Big picture.** Git tracks every change. Small, clearly labeled commits make history readable. Pixi pins every dependency so Explorer, Colab, and the M1 run the same code. Ruff and mypy catch mistakes before you run anything. Pytest proves the harness computes what you think. Pre-commit runs the checks automatically on each commit. The Google style guide and review guide tell you what "clean" means, so you don't invent it.

**In the weeds.**

**Git + Conventional Commits.** Format: `type(scope): summary`. Types you need: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`. Examples: `feat(eval): add p95 latency to harness CLI`, `fix(datagen): clip boxes to image bounds`. A breaking change gets `!` (`feat(harness)!: ...`). Work on short branches, merge to main often, and tag the commit used for each reported result (`git tag e1-final`).

**Pixi.** Docs moved to pixi.prefix.dev. The manifest table is now `[workspace]` (older examples say `[project]`).

```toml
[workspace]
name = "minifsd"
channels = ["conda-forge", "robostack-jazzy"]
platforms = ["osx-arm64", "linux-64"]

[dependencies]
python = "3.12.*"
numpy = ">=2"

[pypi-dependencies]
minifsd = { path = ".", editable = true }
onnxruntime = "==1.32.0"

[feature.train.target.linux-64.dependencies]
pytorch-gpu = "*"
cuda-version = "12.*"

[tasks]
lint = "ruff check . && ruff format --check ."
types = "mypy src"
test = { cmd = "pytest -q", depends-on = ["lint"] }

[environments]
train = ["train"]
```

`pixi install` writes `pixi.lock`. Commit it. `pixi run test` runs the task. `pixi run -e train python train.py` uses the training env. Keep CUDA packages on the linux-64 target only, or the solve fails on osx-arm64.

**Ruff + mypy** in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "D"]
pydocstyle.convention = "google"
[tool.mypy]
strict = true
ignore_missing_imports = true
```

`ruff check --fix .` lints. `ruff format .` formats. `pydocstyle.convention = "google"` makes ruff enforce Google-style docstrings.

**Pytest.** Three features cover most of what you need:

```python
import numpy as np, pytest
from minifsd.metrics import binary_iou, abs_rel

@pytest.fixture
def square_masks():
    gt = np.zeros((10, 10), bool); gt[:5] = True
    pred = np.zeros((10, 10), bool); pred[:4] = True
    return gt, pred

def test_iou(square_masks):
    gt, pred = square_masks
    assert binary_iou(gt, pred) == pytest.approx(40 / 50)

@pytest.mark.parametrize("scale, expected", [(1.0, 0.0), (1.1, 0.1), (0.5, 0.5)])
def test_abs_rel(scale, expected):
    d = np.full(100, 2.0)
    assert abs_rel(d * scale, d) == pytest.approx(expected)
```

Test the metrics on hand-computed cases first. A wrong metric silently ruins every experiment table. Also test label export (box in sim → box in COCO JSON) and the costmap projection with a pixel you can work out on paper.

**Pre-commit** (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.5
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
```

Run `pre-commit install` once. The hook id is now `ruff-check` (it used to be `ruff`).

**Google Python Style Guide, the rules worth knowing.**
- Import modules, not names: `from minifsd import metrics`, then `metrics.binary_iou(...)`. `from typing import ...` is an allowed exception.
- Naming: `lower_with_under` for modules, functions, and variables. `CapWords` for classes. `CAPS_WITH_UNDER` for constants. `_leading_underscore` for internal.
- Docstrings: one-line summary, then `Args:`, `Returns:` (or `Yields:`), `Raises:`.
- Annotate public functions. Use `X | None`, not implicit Optional.
- No mutable default arguments. Use `None` and create the list inside.
- No bare `except:`. Keep `try` blocks small. Don't use `assert` for input validation in library code.
- Put script code in `main()` under `if __name__ == "__main__":`.
- The guide says 80 columns. Ruff's default is 88. Pick one and set it in config.

**Google engineering practices.** A change (a "CL") should do one thing. About 100 lines is usually reasonable and 1,000 is usually too big. Put refactors in their own change. Include tests with the code they test. As a reviewer, approve once the change makes the code better overall, even if it isn't perfect. Even working solo, open a PR per change and read your own diff before merging. It catches a surprising amount.

**Gotchas.**
- RoboStack's ROS 2 Jazzy builds target a specific Python (3.12 for Jazzy). Don't move the workspace Python past what RoboStack supports.
- mypy on ROS 2 and ML code produces lots of missing-stub errors. Keep `ignore_missing_imports` and run strict mypy on your own `src/` only.
- Don't commit datasets, checkpoints, or W&B folders. Add them to `.gitignore` now. GitHub rejects files over 100 MB.
- Pixi's lock resolves for every listed platform. Adding a Linux-only package to the default feature breaks the M1 install.

**Try it (20 min).** Create the repo skeleton: `pixi init`, add ruff, mypy, pytest, and pre-commit. Write `binary_iou` with a Google docstring and the two tests above. Make a commit `test(metrics): add binary IoU tests` and watch pre-commit run.

**Read.**
- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
- Google eng practices, small CLs: https://google.github.io/eng-practices/review/developer/small-cls.html
- Pixi manifest reference: https://pixi.prefix.dev/latest/reference/pixi_manifest/
- Conventional Commits 1.0.0: https://www.conventionalcommits.org/en/v1.0.0/
