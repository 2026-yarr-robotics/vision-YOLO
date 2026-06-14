#!/usr/bin/env python3
"""Run the hand-eye 3-class training NOTEBOOK pipeline locally on the blended set.

The conversion / verification / offline-augmentation stages are exec'd
VERBATIM from the notebook's own cells (yolo26s_m_3class_speedstack_train_
lightaug_redlite.ipynb cells 9, 11, 13, 15, 16) — only the Colab-specific
config cell (Drive mount/zip download) is replaced by local paths, and the
final model.train() call uses the notebook's cell-20 arguments unchanged.
This is the "데이터가 기존 파이프라인을 무수정 통과" proof: the dataset flows
through the notebook's actual code.

Run: python3 train_hand.py [--sizes s] [--data work/hand_mix] [--epochs 250]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from ultralytics import YOLO
import albumentations as A
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "hand-eye-view" / \
    "yolo26s_m_3class_speedstack_train_lightaug_redlite.ipynb"

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT / "work" / "hand_mix",
                help="merged COCO dir (train/valid/test × _annotations.coco.json)")
ap.add_argument("--work", type=Path, default=ROOT / "work" / "hand_train")
ap.add_argument("--sizes", default="s", help="comma list: s,m")
ap.add_argument("--epochs", type=int, default=250)
ap.add_argument("--patience", type=int, default=60)
ap.add_argument("--batch", default="0.60")
args = ap.parse_args()

WORK = args.work
WORK.mkdir(parents=True, exist_ok=True)

# ── notebook cell-4 config, Colab paths → local (everything else verbatim) ─
IMG_SIZE = 1280
EPOCHS = args.epochs
PATIENCE = args.patience
MODEL_SIZES = [s.strip() for s in args.sizes.split(",")]
MODEL_NAME_MAP = {"n": "YOLOv26n-seg", "s": "YOLOv26s-seg",
                  "m": "YOLOv26m-seg", "l": "YOLOv26l-seg"}
EXPECTED_CLASSES = ["fallen-cup", "upright-cup", "mouth-up-cup"]

USE_OFFLINE_AUGMENTATION = True
AUG_COPIES_PER_IMAGE = 1
AUGMENT_NEGATIVE_IMAGES = True
USE_RED_RECOLOR_AUG = True
RED_RECOLOR_PROB = 0.25
RED_COPIES_PER_SELECTED_IMAGE = 1
RED_HUE_OPENCV = 0
RED_HUE_JITTER = 6
RED_APPLY_PHOTO_AUG = True
REBUILD_AUGMENTED_DATASET = True
SAVE_AUGMENTED_DATASET_TO_DRIVE = False
AUG_SEED = 42
random.seed(AUG_SEED)
np.random.seed(AUG_SEED)

TRAIN_BATCH = float(args.batch) if "." in str(args.batch) else int(args.batch)
EVAL_BATCH = 8
CACHE_MODE = os.environ.get("YOLO_CACHE") or False   # 'ram'/'disk' for fast loading
PRED_CONF = 0.25
PRED_IOU = 0.70

LOCAL_COCO_DIR = args.data.resolve()
YOLO_DATASET_DIR = WORK / f"speedstack_3class_yolo_seg_{IMG_SIZE}"
DATA_YAML = YOLO_DATASET_DIR / "data.yaml"
AUG_TAG = f"geom{AUG_COPIES_PER_IMAGE}_redp{int(RED_RECOLOR_PROB * 100)}"
AUGMENTED_YOLO_DATASET_DIR = WORK / f"speedstack_3class_yolo_seg_{IMG_SIZE}_{AUG_TAG}"
AUGMENTED_DATA_YAML = AUGMENTED_YOLO_DATASET_DIR / "data.yaml"
RUN_PROJECT = WORK / "runs" / "segment"
EVAL_PROJECT = WORK / "runs" / "segment_eval"
EXPERIMENT_TAG = f"3class_lightaug_{AUG_TAG}_simmix"
DRIVE_AUG_DATASET_DIR = WORK / "unused_drive_aug"

# YOLO_DEVICE='0,1,2,3' → multi-GPU DDP (near-linear speedup on H200 node)
DEVICE = (os.environ.get("YOLO_DEVICE", "0") if torch.cuda.is_available() else "cpu")
POSSIBLE_SPLITS = ["train", "valid", "val", "test"]

# ── exec the notebook's own cells on this namespace ───────────────────────
nb = json.loads(NOTEBOOK.read_text())
CELLS = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])
         if c["cell_type"] == "code"}
ns = globals()

for idx in (9, 11, 13, 15, 16):       # split scan, convert, verify, aug utils, aug build
    print(f"\n===== notebook cell {idx} =====")
    exec(compile(CELLS[idx], f"<nb-cell-{idx}>", "exec"), ns)

# cell 16 reassigned DATA_YAML to the augmented dataset (notebook behaviour)
print("\nDATA_YAML for training:", ns["DATA_YAML"])

# ── train (cell-20 model.train arguments, verbatim) ────────────────────────
results = {}
for size in MODEL_SIZES:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_name = f"speedstack3class_yolo26{size}_seg_{IMG_SIZE}_epoch{EPOCHS}_{EXPERIMENT_TAG}"
    print(f"\n========== TRAIN {MODEL_NAME_MAP[size]} ({run_name}) ==========")
    model = YOLO(f"yolo26{size}-seg.pt")
    model.train(
        data=str(ns["DATA_YAML"]),
        task="segment",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=TRAIN_BATCH,
        patience=PATIENCE,
        optimizer="auto",
        cos_lr=True,
        warmup_epochs=5.0,
        weight_decay=0.0005,
        overlap_mask=True,
        mask_ratio=2,
        hsv_h=0.03, hsv_s=0.25, hsv_v=0.15,
        degrees=3.0, translate=0.03, scale=0.15, shear=0.0,
        perspective=0.0001, flipud=0.0, fliplr=0.5,
        mosaic=0.15, close_mosaic=20, mixup=0.0, copy_paste=0.0,
        device=DEVICE,
        workers=8,
        amp=True,
        cache=CACHE_MODE,
        plots=True,
        save=True,
        save_period=10,
        project=str(RUN_PROJECT),
        name=run_name,
        exist_ok=True,
    )
    best = RUN_PROJECT / run_name / "weights" / "best.pt"
    assert best.exists(), f"best.pt missing: {best}"
    results[size] = str(best)
    model = YOLO(str(best))
    for split in ("val", "test"):
        m = model.val(data=str(ns["DATA_YAML"]), task="segment", imgsz=IMG_SIZE,
                      batch=EVAL_BATCH, device=DEVICE, split=split,
                      project=str(EVAL_PROJECT), name=f"{run_name}_{split}",
                      exist_ok=True, verbose=False)
        print(f"[{size}:{split}] box mAP50={m.results_dict['metrics/mAP50(B)']:.4f} "
              f"mask mAP50={m.results_dict['metrics/mAP50(M)']:.4f} "
              f"mask mAP50-95={m.results_dict['metrics/mAP50-95(M)']:.4f}")

print("\n========== DONE ==========")
for size, best in results.items():
    print(f"{size}: {best}")
