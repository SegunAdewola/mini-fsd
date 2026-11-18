# Mini FSD reading list

All PDFs are in [papers/](../papers/) at the repo root (not committed; 262 MB). Every arXiv ID below was checked against the arXiv
API on 2026-11-17. Section pointers are a starting point; skim the abstract and figures first.

**Read first** (8 papers, marked ★): YOLO26, RF-DETR, SegFormer, DINOv3, Depth Anything V2,
Kendall et al., Tobin et al., Nagel et al. These carry most of the model comparison and the report.

## Contents

1. [E1 Detection](#e1-detection)
2. [E2 Floor segmentation](#e2-floor-segmentation)
3. [E3 Depth](#e3-depth)
4. [E4 Multi-task (HydraNet-lite)](#e4-multi-task-hydranet-lite)
5. [E5 Sim-to-real and data](#e5-sim-to-real-and-data)
6. [E6 Compression](#e6-compression)
7. [Navigation and systems](#navigation-and-systems)
8. [OCR](#ocr)
9. [Notes and corrections](#notes-and-corrections)

---

## E1 Detection

### ★ YOLO26 (official Ultralytics paper)
- G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, M. E. Kalfaoglu. "Ultralytics YOLO26: Unified
  Real-Time End-to-End Vision Models." arXiv preprint, 2026.
- https://arxiv.org/abs/2606.03748 · [2026-yolo26.pdf](../papers/2026-yolo26.pdf)
- Why: this is your E1 baseline, and it states the design choices (NMS-free end-to-end head, simpler
  export) that explain why it is fast on CPU. Cite this, not the third-party YOLO26 reviews on arXiv.
- If short on time, read: the abstract, the architecture section on the end-to-end head, and the
  CPU/ONNX latency tables for the nano model.

### YOLO (original)
- J. Redmon, S. Divvala, R. Girshick, A. Farhadi. "You Only Look Once: Unified, Real-Time Object
  Detection." CVPR 2016.
- https://arxiv.org/abs/1506.02640 · [2016-yolo.pdf](../papers/2016-yolo.pdf)
- Why: the one-stage, grid-based idea that every YOLO since builds on. Good for one paragraph of
  background in the report.
- If short on time, read: Section 2 (Unified Detection) with Figures 1 and 2, and Section 2.4
  (limitations).

### DETR
- N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko. "End-to-End Object
  Detection with Transformers." ECCV 2020.
- https://arxiv.org/abs/2005.12872 · [2020-detr.pdf](../papers/2020-detr.pdf)
- Why: explains set prediction with Hungarian matching, which is how DETR-style models (RF-DETR) drop
  NMS. You need this to explain the E1 matchup.
- If short on time, read: Section 3.1 (set prediction loss) and Figure 2.

### RT-DETR
- Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, J. Chen. "DETRs Beat YOLOs on Real-time
  Object Detection." CVPR 2024.
- https://arxiv.org/abs/2304.08069 · [2024-rt-detr.pdf](../papers/2024-rt-detr.pdf)
- Why: the first DETR that competed with YOLO on speed. Its NMS analysis shows why NMS time must be
  counted in latency, which matters for your harness.
- If short on time, read: Figure 1, the NMS analysis section, and the hybrid encoder section.

### ★ RF-DETR
- I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, N. Peri. "RF-DETR: Neural Architecture Search for
  Real-Time Detection Transformers." ICLR 2026 (arXiv 2025).
- https://arxiv.org/abs/2511.09554 · [2025-rf-detr.pdf](../papers/2025-rf-detr.pdf)
- Why: your second E1 candidate. It uses a DINOv2 backbone and weight-sharing NAS, and it has a
  useful discussion of how inconsistent latency benchmarking is across papers.
- If short on time, read: the method overview figure, the section on latency measurement, and the
  Nano results on COCO and RF100-VL.

### DINOv2
- M. Oquab, T. Darcet, T. Moutakanni, H. Vo, et al. "DINOv2: Learning Robust Visual Features without
  Supervision." TMLR 2024 (arXiv 2023).
- https://arxiv.org/abs/2304.07193 · [2023-dinov2.pdf](../papers/2023-dinov2.pdf)
- Why: the backbone inside RF-DETR and the ancestor of DINOv3. Its frozen-feature results on
  segmentation and depth back up your E2 hypothesis.
- If short on time, read: the results on dense tasks (semantic segmentation and depth with linear
  heads on frozen features) and the qualitative PCA feature figure.

## E2 Floor segmentation

### ★ SegFormer
- E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo. "SegFormer: Simple and Efficient
  Design for Semantic Segmentation with Transformers." NeurIPS 2021.
- https://arxiv.org/abs/2105.15203 · [2021-segformer.pdf](../papers/2021-segformer.pdf)
- Why: SegFormer-B0 is your likely efficient E2 candidate (the study guide prefers it over
  EfficientViT). It also reports robustness to image corruptions, which relates to sim-to-real.
- If short on time, read: Section 3 (Mix Transformer encoder and All-MLP decoder) with Figure 2, and
  the Cityscapes-C robustness table.

### EfficientViT
- H. Cai, J. Li, M. Hu, C. Gan, S. Han. "EfficientViT: Multi-Scale Linear Attention for
  High-Resolution Dense Prediction." ICCV 2023.
- https://arxiv.org/abs/2205.14756 · [2023-efficientvit.pdf](../papers/2023-efficientvit.pdf)
- Why: the other efficient E2 option. The study guide notes the repo is unmaintained, so read it
  mainly to justify why you chose (or dropped) it.
- If short on time, read: the multi-scale linear attention section and the latency comparison figure.

### ★ DINOv3
- O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, et al. "DINOv3." arXiv preprint, 2025.
- https://arxiv.org/abs/2508.10104 · [2025-dinov3.pdf](../papers/2025-dinov3.pdf)
- Why: your frozen ViT-S/16 E2 candidate. The paper's main claim is better dense features, which is
  exactly the property your sim-to-real hypothesis tests. Note the gated license in the report.
- If short on time, read: the Gram anchoring section, the dense-task results with a frozen backbone,
  and the section on distilled small models (ViT-S).

### Probing frozen foundation features (linear probing for dense prediction)
- M. El Banani, A. Raj, K.-K. Maninis, A. Kar, Y. Li, M. Rubinstein, D. Sun, L. Guibas, J. Johnson,
  V. Jampani. "Probing the 3D Awareness of Visual Foundation Models." CVPR 2024.
- https://arxiv.org/abs/2404.08636 · [2024-probe3d.pdf](../papers/2024-probe3d.pdf)
- Why: there is no single "standard" linear-probing paper for dense tasks. This one is the closest: a
  careful protocol for training small heads on frozen DINO/CLIP/SAM features for depth and normals.
  Copy its protocol for your DINOv3 head.
- If short on time, read: the probing setup section and the depth results table. The DINOv2 paper's
  linear-probe results are the other common reference.

## E3 Depth

### Eigen et al. (origin of the depth metrics)
- D. Eigen, C. Puhrsch, R. Fergus. "Depth Map Prediction from a Single Image using a Multi-Scale Deep
  Network." NeurIPS 2014.
- https://arxiv.org/abs/1406.2283 · [2014-eigen-depth.pdf](../papers/2014-eigen-depth.pdf)
- Why: defines the metrics you report (AbsRel, δ<1.25) and the scale-invariant error. It also set up
  the standard NYUv2 test split.
- If short on time, read: the scale-invariant error section and the experiments section where the
  error metrics are listed.

### MiDaS
- R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun. "Towards Robust Monocular Depth
  Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer." TPAMI 2022 (arXiv 2019).
- https://arxiv.org/abs/1907.01341 · [2020-midas.pdf](../papers/2020-midas.pdf)
- Why: explains relative (affine-invariant) depth and how to align scale and shift before scoring.
  You need this because DA3 Small gives relative depth only.
- If short on time, read: the scale- and shift-invariant loss section and the zero-shot evaluation
  protocol.

### Depth Anything (V1)
- L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, H. Zhao. "Depth Anything: Unleashing the Power of
  Large-Scale Unlabeled Data." CVPR 2024.
- https://arxiv.org/abs/2401.10891 · [2024-depth-anything.pdf](../papers/2024-depth-anything.pdf)
- Why: background for V2 and V3. Shows how pseudo-labels on unlabeled images scale depth models.
- If short on time, read: the method overview figure and the zero-shot results table.

### ★ Depth Anything V2
- L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, H. Zhao. "Depth Anything V2." NeurIPS 2024.
- https://arxiv.org/abs/2406.09414 · [2024-depth-anything-v2.pdf](../papers/2024-depth-anything-v2.pdf)
- Why: an E3 candidate, and its recipe (teacher trained on synthetic data, then pseudo-label real
  images, then distill to small students) is close to your own plan for the distilled depth head.
- If short on time, read: the section on why synthetic data helps and where it fails, and the
  teacher-student pipeline figure.

### Depth Anything 3
- H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, B. Kang. "Depth Anything 3:
  Recovering the Visual Space from Any Views." ICLR 2026 (arXiv 2025).
- https://arxiv.org/abs/2511.10647 · [2025-depth-anything-3.pdf](../papers/2025-depth-anything-3.pdf)
- Why: your other E3 candidate and the heavy baseline. It is an any-view model; you use it on single
  images, so focus on the monocular results.
- If short on time, read: the depth-ray representation section and the monocular depth comparison
  against DA2.

## E4 Multi-task (HydraNet-lite)

### ★ Kendall, Gal and Cipolla
- A. Kendall, Y. Gal, R. Cipolla. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
  Geometry and Semantics." CVPR 2018.
- https://arxiv.org/abs/1705.07115 · [2018-kendall-uncertainty-mtl.pdf](../papers/2018-kendall-uncertainty-mtl.pdf)
- Why: the standard way to balance the floor, detection, and depth losses without hand-tuning. Same
  task mix as yours (semantics plus depth).
- If short on time, read: the loss derivation section (the final weighted loss equation) and the
  figure showing how results swing with fixed weights.

### Multi-task learning survey
- S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, L. Van Gool. "Multi-Task
  Learning for Dense Prediction Tasks: A Survey." TPAMI 2021.
- https://arxiv.org/abs/2004.13379 · [2021-mtl-dense-survey.pdf](../papers/2021-mtl-dense-survey.pdf)
- Why: gives you the vocabulary (hard parameter sharing, negative transfer, task balancing) for the
  "does sharing cost accuracy" question. Its experiments use NYUD-v2, like you.
- If short on time, read: the encoder-focused architectures section, the task balancing section, and
  the NYUD-v2 results.

### Tesla HydraNet (no paper; talks only)
- Tesla has never published a HydraNet paper. Best primary sources:
  - Tesla AI Day 2021 (Aug 19, 2021), Karpathy's vision section: https://www.youtube.com/watch?v=j0z4FweCy4M
  - "PyTorch at Tesla," Andrej Karpathy, PyTorch DevCon 2019 (first public HydraNet description):
    https://www.youtube.com/watch?v=oBklltKXtDE
  - Karpathy, CVPR 2021 Workshop on Autonomous Driving keynote: https://www.youtube.com/watch?v=g6bOwQdCJrc
- Why: the shared-backbone, many-heads design your E4 copies. Cite the talks as the source.
- If short on time, watch: the AI Day vision section where the shared backbone with task heads is
  drawn, and the PyTorch DevCon talk (short, and all about HydraNet).

## E5 Sim-to-real and data

### ★ Domain randomization
- J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel. "Domain Randomization for
  Transferring Deep Neural Networks from Simulation to the Real World." IROS 2017.
- https://arxiv.org/abs/1703.06907 · [2017-domain-randomization.pdf](../papers/2017-domain-randomization.pdf)
- Why: the core idea behind your data engine and the E5 "sim + DR" arm.
- If short on time, read: the method section (what they randomize) and the ablation on which
  randomizations matter.

### Structured domain randomization
- A. Prakash, S. Boochoon, M. Brophy, D. Acuna, E. Cameracci, G. State, O. Shapira, S. Birchfield.
  "Structured Domain Randomization: Bridging the Reality Gap by Context-Aware Synthetic Data."
  ICRA 2019.
- https://arxiv.org/abs/1810.10093 · [2019-structured-dr.pdf](../papers/2019-structured-dr.pdf)
- Why: your procedural buildings place doors, signs and boxes in sensible spots, which is structured
  DR rather than pure DR. This paper lets you name and justify that.
- If short on time, read: the method section and the figure comparing DR with SDR images.

### SAM 2
- N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, et al. "SAM 2: Segment Anything in Images
  and Videos." ICLR 2025 (arXiv 2024).
- https://arxiv.org/abs/2408.00714 · [2024-sam2.pdf](../papers/2024-sam2.pdf)
- Why: the tool you use to label floor masks on real photos. Cite it in the dataset section.
- If short on time, read: the abstract and the model overview figure. Skip the video parts.

### NYU Depth V2
- N. Silberman, D. Hoiem, P. Kohli, R. Fergus. "Indoor Segmentation and Support Inference from RGBD
  Images." ECCV 2012.
- DOI: 10.1007/978-3-642-33715-4_54 · PDF from the author's page:
  https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf · [2012-nyu-depth-v2.pdf](../papers/2012-nyu-depth-v2.pdf)
- Why: the source of your public real benchmark (1,449 labeled images). Cite this paper for the
  dataset. Not on arXiv.
- If short on time, read: the dataset section and the figure of example labeled scenes.

## E6 Compression

### Integer quantization
- B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, D. Kalenichenko. "Quantization
  and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
- https://arxiv.org/abs/1712.05877 · [2018-integer-quantization.pdf](../papers/2018-integer-quantization.pdf)
- Why: defines the scale and zero-point scheme that ONNX Runtime INT8 uses.
- If short on time, read: Section 2 (the quantization scheme) and Section 3 (training with simulated
  quantization).

### ★ Quantization white paper
- M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, T. Blankevoort. "A White
  Paper on Neural Network Quantization." arXiv preprint, 2021.
- https://arxiv.org/abs/2106.08295 · [2021-quantization-whitepaper.pdf](../papers/2021-quantization-whitepaper.pdf)
- Why: the practical guide for E6. It covers post-training quantization, calibration, and what to
  check when INT8 accuracy drops.
- If short on time, read: the fundamentals section, the PTQ section, and the standard PTQ pipeline
  and debugging flow at the end of it.

### Knowledge distillation
- G. Hinton, O. Vinyals, J. Dean. "Distilling the Knowledge in a Neural Network." NIPS 2014 Deep
  Learning Workshop (arXiv 2015).
- https://arxiv.org/abs/1503.02531 · [2015-distillation.pdf](../papers/2015-distillation.pdf)
- Why: the reference for your distilled depth head (DA teacher to small student). Short paper.
- If short on time, read: Section 2 (distillation with temperature).

## Navigation and systems

### ROS 2
- S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall. "Robot Operating System 2: Design,
  Architecture, and Uses In The Wild." Science Robotics 7(66), 2022.
- DOI: 10.1126/scirobotics.abm6074 (paywalled) · arXiv preprint: https://arxiv.org/abs/2211.07752 ·
  [2022-ros2.pdf](../papers/2022-ros2.pdf)
- Why: the citation for ROS 2 and a clear explanation of DDS and QoS, which matter for your camera
  topics.
- If short on time, read: the design and architecture sections (DDS, QoS, executors).

### Nav2 ("The Marathon 2")
- S. Macenski, F. Martín, R. White, J. Ginés Clavero. "The Marathon 2: A Navigation System."
  IROS 2020.
- https://arxiv.org/abs/2003.00368 · [2020-nav2-marathon2.pdf](../papers/2020-nav2-marathon2.pdf)
- Why: the Nav2 citation. Explains behavior trees, lifecycle nodes, and the layered costmap you feed.
- If short on time, read: the system design section and the architecture figure.

### Smac planner
- S. Macenski, M. Booker, J. Wallace, T. Fischer. "Open-Source, Cost-Aware Kinematically Feasible
  Planning for Mobile and Surface Robotics." arXiv preprint, 2024 (no peer-reviewed version found).
- https://arxiv.org/abs/2401.13078 · [2024-smac-planner.pdf](../papers/2024-smac-planner.pdf)
- Why: the paper behind Nav2's Smac planners (Hybrid-A*, State Lattice). Cite it if you use Smac.
- If short on time, read: the Hybrid-A* section and the benchmark tables.

### MPPI
- G. Williams, P. Drews, B. Goldfain, J. M. Rehg, E. A. Theodorou. "Information Theoretic Model
  Predictive Control: Theory and Applications to Autonomous Driving." IEEE T-RO 2018 (arXiv 2017).
- https://arxiv.org/abs/1707.02342 · [2018-mppi-it-mpc.pdf](../papers/2018-mppi-it-mpc.pdf)
- Why: the theory behind Nav2's MPPI controller, which sets your max safe speed. This journal version
  covers both earlier ICRA papers.
- If short on time, read: the algorithm section (the MPPI update) and the AutoRally driving results.
- Earlier conference versions, for reference: "Aggressive Driving with Model Predictive Path Integral
  Control" (ICRA 2016, DOI 10.1109/ICRA.2016.7487277) and "Information Theoretic MPC for Model-Based
  Reinforcement Learning" (ICRA 2017, DOI 10.1109/ICRA.2017.7989202). Not downloaded.

### MuJoCo
- E. Todorov, T. Erez, Y. Tassa. "MuJoCo: A physics engine for model-based control." IROS 2012.
- DOI: 10.1109/IROS.2012.6386109 · PDF from the author's lab page:
  https://roboti.us/lab/papers/TodorovIROS12.pdf · [2012-mujoco.pdf](../papers/2012-mujoco.pdf)
- Why: the citation for your simulator. Not on arXiv.
- If short on time, read: the abstract and the overview of the computation pipeline. The modern MuJoCo
  docs are more useful for building.

### Lift-Splat-Shoot (BEV)
- J. Philion, S. Fidler. "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by
  Implicitly Unprojecting to 3D." ECCV 2020.
- https://arxiv.org/abs/2008.05711 · [2020-lift-splat-shoot.pdf](../papers/2020-lift-splat-shoot.pdf)
- Why: the learned version of what your costmap step does by hand (image plus depth to a top-down
  grid). Good for the "small-scale Tesla BEV" paragraph.
- If short on time, read: the Lift and Splat sections and Figure 2.

## OCR

### PP-OCR (original)
- Y. Du, C. Li, R. Guo, X. Yin, W. Liu, J. Zhou, Y. Bai, Z. Yu, et al. "PP-OCR: A Practical Ultra
  Lightweight OCR System." arXiv preprint, 2020.
- https://arxiv.org/abs/2009.09941 · [2020-pp-ocr.pdf](../papers/2020-pp-ocr.pdf)
- Why: explains the three-stage pipeline (text detection, direction classifier, recognition) you run
  on sign crops.
- If short on time, read: the pipeline overview figure and the section listing each stage.

### PaddleOCR 3.0 technical report (covers PP-OCRv5)
- C. Cui, T. Sun, M. Lin, T. Gao, Y. Zhang, J. Liu, X. Wang, Z. Zhang, et al. "PaddleOCR 3.0
  Technical Report." arXiv preprint, 2025.
- https://arxiv.org/abs/2507.05595 · [2025-paddleocr-3.pdf](../papers/2025-paddleocr-3.pdf)
- Why: describes the current toolkit version you will install.
- If short on time, read: the PP-OCRv5 section only.

### PP-OCRv5 (dedicated paper)
- C. Cui, Y. Zhang, T. Sun, X. Wang, H. Liu, M. Lin, Y. Zhang, T. Gao, et al. "PP-OCRv5: A
  Specialized 5M-Parameter Model Rivaling Billion-Parameter Vision-Language Models on OCR Tasks."
  arXiv preprint, 2026.
- https://arxiv.org/abs/2603.24373 · [2026-pp-ocrv5.pdf](../papers/2026-pp-ocrv5.pdf)
- Why: the latest PP-OCR model paper. Cite this for the model you actually run.
- If short on time, read: the abstract and the main results table.

## Notes and corrections

- YOLO26 has an official paper (arXiv 2606.03748, June 2026). Several other arXiv papers with
  "YOLO26" in the title (2509.25164, 2601.12882, 2602.14582) are third-party reviews.
- RT-DETR's paper title is "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024).
- RF-DETR is ICLR 2026; the arXiv version is from Nov 2025.
- Kendall et al. is on arXiv from 2017 but published at CVPR 2018.
- Vandenhende et al. is TPAMI 2021 (arXiv 2020).
- Structured DR (Prakash et al.) is ICRA 2019 (arXiv 2018).
- Hinton et al. is a 2014 workshop paper; arXiv is 2015.
- The MPPI "information theoretic MPC" title belongs to both an ICRA 2017 paper and the T-RO 2018
  journal paper; the journal one is on arXiv and is downloaded here.
- The Smac paper has no peer-reviewed version yet; cite the arXiv preprint.
- Tesla HydraNet has no paper; cite the talks.
