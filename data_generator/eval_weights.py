#!/usr/bin/env python3
"""Acceptance metrics for the finetuned weights (temp_task 검증 기준 1·2).

1. sim holdout  : dataset/sim/{hand,exo} test split —
                  fallen recall ≥ 0.9, upright→fallen confusion ≈ 0 @ conf 0.5
2. real 회귀     : dataset/real/* test split — new weights must match the
                  deployed weights' single-class cup detection (the real GT is
                  class-agnostic 'cup', so we compare class-agnostic mask
                  detection rates old vs new on identical GT geometry).

Run: python3 eval_weights.py --hand-new <pt> --exo-new <pt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INTEG = ROOT.parent / "cup-stack-integration"
YOLO_DIR = INTEG / "ros2-depth-point-cloude" / "vision" / "yolo"
CLASS_NAMES = ["fallen-cup", "mouth-up-cup", "upright-cup"]

ap = argparse.ArgumentParser()
ap.add_argument("--hand-new", type=Path, default=None)
ap.add_argument("--exo-new", type=Path, default=None)
ap.add_argument("--hand-old", type=Path, default=YOLO_DIR /
                "speedstack3class_yolo26s_seg_1280_epoch250_3class_lightaug_geom1_redp25_sm_a100_best.pt")
ap.add_argument("--exo-old", type=Path, default=YOLO_DIR / "0609_exo_best.pt")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--iou", type=float, default=0.5)
args = ap.parse_args()


def predict(model, img_path, conf):
    res = model.predict(str(img_path), imgsz=1280, conf=conf, verbose=False)[0]
    out = []
    if res.masks is None:
        return out
    h, w = res.orig_shape
    for i in range(len(res.boxes)):
        m = res.masks.data[i].cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        out.append((model.names[int(res.boxes.cls[i])],
                    float(res.boxes.conf[i]), m))
    return out


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    return float(inter) / max(float(np.logical_or(a, b).sum()), 1.0)


def norm_cls(name: str) -> str:
    """Map any historical class name onto the 3-class scheme."""
    n = name.lower()
    if "fallen" in n:
        return "fallen-cup"
    if "mouth" in n:
        return "mouth-up-cup"
    return "upright-cup"


# ── 1) sim holdout: GT from meta.jsonl test-split records ─────────────────
def eval_sim(view: str, weights: Path):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    meta = ROOT / "dataset" / "sim" / "meta.jsonl"
    img_root = ROOT / "dataset" / "sim" / view
    conf_mat = {g: {p: 0 for p in CLASS_NAMES + ["miss"]} for g in CLASS_NAMES}
    n_img = 0
    with meta.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["view"] != view or rec["split"] != "test":
                continue
            img = (img_root / "test" / rec["file"] if view == "hand"
                   else img_root / "test" / "images" / rec["file"])
            if not img.exists():
                continue
            n_img += 1
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            preds = predict(model, img, args.conf)
            for lab in rec["labels"]:
                gt_mask = np.zeros((h, w), np.uint8)
                for poly in lab["polys"]:
                    pts = np.asarray(poly, np.float32).reshape(-1, 2)
                    cv2.fillPoly(gt_mask, [pts.astype(np.int32)], 1)
                gt_mask = gt_mask.astype(bool)
                best = max(((mask_iou(gt_mask, pm), cn) for cn, cf, pm in preds),
                           default=(0, None))
                pred = (norm_cls(best[1]) if best[0] >= args.iou and best[1]
                        else "miss")
                conf_mat[lab["cls"]][pred] += 1

    print(f"\n=== sim holdout [{view}] {weights.name} ({n_img} imgs, conf {args.conf}) ===")
    for g in CLASS_NAMES:
        row = conf_mat[g]
        tot = sum(row.values())
        if not tot:
            continue
        rec_ = row[g] / tot
        print(f"  GT {g:13s} n={tot:4d} recall={rec_:.3f}  " +
              " ".join(f"{p}:{row[p]}" for p in CLASS_NAMES + ['miss']))
    up = conf_mat["upright-cup"]
    up_tot = max(sum(up.values()), 1)
    print(f"  → fallen recall = {conf_mat['fallen-cup']['fallen-cup'] / max(sum(conf_mat['fallen-cup'].values()), 1):.3f} (기준 ≥0.9)")
    print(f"  → upright→fallen = {up['fallen-cup']}/{up_tot} (기준 ≈0)")
    return conf_mat


# ── 2) real regression: class-agnostic detection vs the GT cup masks ──────
def eval_real(view: str, weights_old: Path, weights_new: Path):
    from ultralytics import YOLO
    import pycocotools.mask as maskUtils
    models = {"old": YOLO(str(weights_old)), "new": YOLO(str(weights_new))}
    gt_items = []          # (img_path, [gt_mask])
    if view == "hand":
        split_dir = ROOT / "dataset" / "real" / "hand" / "test"
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            masks = []
            for a in by_img.get(im["id"], []):
                seg = a["segmentation"]
                rle = (maskUtils.frPyObjects(seg, im["height"], im["width"])
                       if isinstance(seg.get("counts"), list) else seg)
                m = maskUtils.decode(rle)
                masks.append((m if m.ndim == 2 else np.any(m, 2)).astype(bool))
            gt_items.append((split_dir / im["file_name"], masks))
    else:
        base = ROOT / "dataset" / "real" / "exo" / "YOLO_YARR-2" / "test"
        for img in sorted((base / "images").glob("*.jpg")):
            lbl = base / "labels" / f"{img.stem}.txt"
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            masks = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    pts = (np.asarray([float(v) for v in parts[1:]], np.float32)
                           .reshape(-1, 2) * [w, h]).astype(np.int32)
                    m = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(m, [pts], 1)
                    masks.append(m.astype(bool))
            gt_items.append((img, masks))

    print(f"\n=== real regression [{view}] (class-agnostic, conf {args.conf}, iou {args.iou}) ===")
    for tag, model in models.items():
        tp = fn = fp = 0
        for img_path, masks in gt_items:
            preds = predict(model, img_path, args.conf)
            used = set()
            for gm in masks:
                best_i, best_v = -1, 0.0
                for i, (_, _, pm) in enumerate(preds):
                    if i in used:
                        continue
                    v = mask_iou(gm, pm)
                    if v > best_v:
                        best_i, best_v = i, v
                if best_v >= args.iou:
                    tp += 1
                    used.add(best_i)
                else:
                    fn += 1
            fp += len(preds) - len(used)
        n = tp + fn
        print(f"  [{tag:3s}] {Path(model.ckpt_path).name[:46]:46s} "
              f"recall={tp / max(n, 1):.3f} ({tp}/{n})  FP={fp}")


if __name__ == "__main__":
    if args.hand_new:
        eval_sim("hand", args.hand_new)
        eval_real("hand", args.hand_old, args.hand_new)
    if args.exo_new:
        eval_sim("exo", args.exo_new)
        eval_real("exo", args.exo_old, args.exo_new)
