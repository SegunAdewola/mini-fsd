# ADR 0007: Train on Colab's free tier, not Explorer, for now

- Status: Accepted (revisit if a run doesn't fit Colab's limits)
- Date: 2026-11-25
- Plan step: T2

## Context

M3 trains five or so models (SegFormer-B0, a DINO head, YOLO26n, RF-DETR Nano, a
small depth network) between Nov 26 and Dec 2. The M1 has no CUDA and 8 GB of
memory, so training happens elsewhere. The design planned Northeastern's
Explorer cluster: 8 h jobs on the `gpu` partition, SSH keys required since Nov
6, Slurm scripts, scratch space to set up. None of that is set up yet, and the
time to learn it now competes with the report (Dec 8).

Colab's free tier gives a T4 (16 GB) in a browser with nothing to install beyond
`pip install`. Its limits: sessions end after at most about 12 h and after about
90 minutes idle, GPU time is rationed per day without a published number, and
the session disk is wiped each time. The data to train on is about 2.3 GB (v9
1.44 GB, COCO people 0.87 GB).

## Decision

Train on Colab's free tier. Everything that matters lives outside the session:

- Data goes to Google Drive once, as one uncompressed zip per dataset (Drive is
  slow with thousands of small files). Each session copies the zips to the
  session disk and unzips them there.
- Training code stays in `src/minifsd/train/`, tested locally on CPU in the `cv`
  pixi environment. The notebook only mounts Drive, installs the repo from
  GitHub at a pinned commit, and calls those modules. No logic in the notebook.
- Checkpoints go to Drive after every epoch, and every run resumes from the last
  one, so a dropped session costs at most one epoch.
- Packages Colab lacks (ultralytics, rfdetr) are pip-installed at pinned
  versions in the notebook's first cell, recorded here when added.

## Consequences

- No cluster setup now; the first training run can happen the day the adapters
  (T3) are done.
- Runs must fit a T4 and survive disconnects. Long runs need a person to reopen
  the notebook; a run left overnight may stop and wait.
- Results depend on which GPU Colab hands out (usually a T4); the run logs
  record it. Latency is never measured on Colab, only on the M1 through the
  harness, so the edge numbers are unaffected.
- Data on Drive is private to the user's Google account. Nothing new is
  committed.

## Alternatives considered

- **Explorer (the design's plan).** More GPU time and no idle limit, but the
  account, keys, Slurm and scratch quota need setting up first. Kept as the
  fallback if Colab's rationing stops a run.
- **Colab Pro.** Faster GPUs and longer sessions for about $10 a month. Not
  needed until the free tier proves too small.
- **Training on the M1 (MPS).** No setup, but 8 GB shared memory and slow
  training; fine for smoke tests, too slow for real runs.
