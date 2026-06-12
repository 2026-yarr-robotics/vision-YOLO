#!/usr/bin/env python3
"""Relabel the single-class real datasets into the 3-class scheme.

The deployed models were trained on multi-class variants that are not in this
repo (hand: 4-cat roboflow v2, exo: 'YOLO_YARR-2-class'), while dataset/real
holds single-class exports (hand v3: 'cup', exo: 'CUP'). To blend real+sim for
3-class training we re-derive per-annotation classes by matching each GT mask
against the deployed model's predictions (mask IoU), keeping the GT geometry:

  hand: speedstack3class_yolo26s_*  (fallen/mouth-up/upright on real images)
  exo : 0609_exo_best               (fallen/upright on real images)

Unmatched or low-IoU annotations default to upright-cup (the overwhelmingly
dominant pose in these captures) and every non-upright or suspicious sample is
dumped into review montages for an eyeball pass. Manual corrections go into
<out>/overrides.json: {"<split>/<image file>": {"<ann index>": "<class>"}}
and are applied on a re-run.

Run with the system python3 (ultralytics):  python3 relabel_real.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # vision-YOLO
INTEG = ROOT.parent / "cup-stack-integration"
YOLO_DIR = INTEG / "ros2-depth-point-cloude" / "vision" / "yolo"
CLASS_NAMES = ["fallen-cup", "mouth-up-cup", "upright-cup"]
SPLITS = ("train", "valid", "test")

ap = argparse.ArgumentParser()
ap.add_argument("--hand-model", default=str(
    YOLO_DIR / "speedstack3class_yolo26s_seg_1280_epoch250_3class_lightaug_geom1_redp25_sm_a100_best.pt"))
ap.add_argument("--exo-model", default=str(YOLO_DIR / "0609_exo_best.pt"))
ap.add_argument("--conf", type=float, default=0.20)
ap.add_argument("--iou-match", type=float, default=0.45)
ap.add_argument("--views", default="hand,exo")
args = ap.parse_args()


def pred_masks(model, img_path: Path, imgsz=1280):
    """[(class_name, conf, bool mask HxW)] in image resolution."""
    res = model.predict(str(img_path), imgsz=imgsz, conf=args.conf,
                        verbose=False)[0]
    out = []
    if res.masks is None:
        return out
    h, w = res.orig_shape
    for i in range(len(res.boxes)):
        cname = model.names[int(res.boxes.cls[i])]
        m = res.masks.data[i].cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        out.append((cname, float(res.boxes.conf[i]), m))
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(float(union), 1.0)


def crop(img, bbox, margin=0.25, size=170):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * (1 + margin)
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        c = np.zeros((size, size, 3), np.uint8)
    return cv2.resize(c, (size, size))


class Montage:
    def __init__(self, out_dir: Path, cols=6, rows=5, cell=170):
        self.dir, self.cols, self.rows, self.cell = out_dir, cols, rows, cell
        self.tiles, self.page = [], 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, tile_img, text: str):
        t = tile_img.copy()
        for j, line in enumerate(text.split("\n")):
            cv2.putText(t, line, (3, 14 + 13 * j), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 255, 255), 1)
        self.tiles.append(t)
        if len(self.tiles) >= self.cols * self.rows:
            self.flush()

    def flush(self):
        if not self.tiles:
            return
        n = len(self.tiles)
        rows = (n + self.cols - 1) // self.cols
        sheet = np.zeros((rows * self.cell, self.cols * self.cell, 3), np.uint8)
        for i, t in enumerate(self.tiles):
            r, c = divmod(i, self.cols)
            sheet[r * self.cell:(r + 1) * self.cell,
                  c * self.cell:(c + 1) * self.cell] = t
        cv2.imwrite(str(self.dir / f"page_{self.page:03d}.png"), sheet)
        self.page += 1
        self.tiles = []


def load_overrides(out_root: Path) -> dict:
    p = out_root / "overrides.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ── hand: COCO 'cup' → 4-cat 3-class COCO ─────────────────────────────────
def relabel_hand():
    from pycocotools import mask as maskUtils
    from ultralytics import YOLO
    model = YOLO(args.hand_model)
    src = ROOT / "dataset" / "real" / "hand"
    dst = ROOT / "dataset" / "real" / "hand3class"
    overrides = load_overrides(dst)
    review = Montage(dst / "review")
    stats = {"matched": 0, "default_upright": 0, "overridden": 0}
    cls_count = {c: 0 for c in CLASS_NAMES}

    for split in SPLITS:
        coco = json.loads((src / split / "_annotations.coco.json").read_text())
        (dst / split).mkdir(parents=True, exist_ok=True)
        cats = [{"id": 0, "name": "hand-eye-view-speed-stack-cup",
                 "supercategory": "none"}] + [
            {"id": i + 1, "name": n,
             "supercategory": "hand-eye-view-speed-stack-cup"}
            for i, n in enumerate(CLASS_NAMES)]
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        new_anns = []
        for im in coco["images"]:
            img_path = src / split / im["file_name"]
            shutil.copy2(img_path, dst / split / im["file_name"])
            anns = by_img.get(im["id"], [])
            if not anns:
                continue
            preds = pred_masks(model, img_path)
            img = cv2.imread(str(img_path))
            for k, a in enumerate(sorted(anns, key=lambda x: x["id"])):
                seg = a["segmentation"]
                if isinstance(seg, dict):
                    rle = (maskUtils.frPyObjects(seg, im["height"], im["width"])
                           if isinstance(seg.get("counts"), list) else seg)
                    gt_mask = maskUtils.decode(rle)
                    if gt_mask.ndim == 3:
                        gt_mask = np.any(gt_mask, axis=2)
                    gt_mask = gt_mask.astype(bool)
                else:
                    gt_mask = np.zeros((im["height"], im["width"]), bool)
                    for poly in seg:
                        pts = np.asarray(poly, np.float32).reshape(-1, 2)
                        cv2.fillPoly(gt_mask.view(np.uint8).reshape(gt_mask.shape),
                                     [pts.astype(np.int32)], 1)
                best = max(((mask_iou(gt_mask, pm), cn, cf)
                            for cn, cf, pm in preds), default=(0, None, 0))
                key = f"{split}/{im['file_name']}"
                if key in overrides and str(k) in overrides[key]:
                    cls = overrides[key][str(k)]
                    prov = "override"
                    stats["overridden"] += 1
                elif best[0] >= args.iou_match and best[1] in CLASS_NAMES:
                    cls = best[1]
                    prov = f"iou{best[0]:.2f} c{best[2]:.2f}"
                    stats["matched"] += 1
                else:
                    cls = "upright-cup"
                    prov = f"DEFAULT iou{best[0]:.2f}"
                    stats["default_upright"] += 1
                cls_count[cls] += 1
                if cls != "upright-cup" or best[0] < args.iou_match or \
                        (best[1] and best[1] != cls):
                    review.add(crop(img, a["bbox"]),
                               f"{cls}\n{prov}\n{split} #{k}\n{im['file_name'][:18]}")
                new_anns.append({**a, "category_id": 1 + CLASS_NAMES.index(cls)})
        out = {**coco, "categories": cats, "annotations": new_anns}
        (dst / split / "_annotations.coco.json").write_text(json.dumps(out))
        print(f"[hand:{split}] images={len(coco['images'])} anns={len(new_anns)}")
    review.flush()
    print(f"[hand] {stats} classes={cls_count} → {dst}")


# ── exo: YOLO 'CUP' txt → 3-class txt ─────────────────────────────────────
def relabel_exo():
    from ultralytics import YOLO
    model = YOLO(args.exo_model)
    src = ROOT / "dataset" / "real" / "exo" / "YOLO_YARR-2"
    dst = ROOT / "dataset" / "real" / "exo" / "YOLO_YARR-3class"
    overrides = load_overrides(dst)
    review = Montage(dst / "review")
    stats = {"matched": 0, "default_upright": 0, "overridden": 0}
    cls_count = {c: 0 for c in CLASS_NAMES}

    for split in SPLITS:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in sorted(img_dir.glob("*.jpg")):
            shutil.copy2(img_path, dst / split / "images" / img_path.name)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            lines_out = []
            gt_polys = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    pts = np.asarray([float(v) for v in parts[1:]],
                                     np.float32).reshape(-1, 2)
                    gt_polys.append(pts)
            preds = pred_masks(model, img_path) if gt_polys else []
            for k, pts in enumerate(gt_polys):
                gt_mask = np.zeros((h, w), np.uint8)
                px = (pts * [w, h]).astype(np.int32)
                cv2.fillPoly(gt_mask, [px], 1)
                gt_mask = gt_mask.astype(bool)
                best = max(((mask_iou(gt_mask, pm), cn, cf)
                            for cn, cf, pm in preds), default=(0, None, 0))
                key = f"{split}/{img_path.name}"
                if key in overrides and str(k) in overrides[key]:
                    cls = overrides[key][str(k)]
                    prov = "override"
                    stats["overridden"] += 1
                elif best[0] >= args.iou_match and best[1] in CLASS_NAMES:
                    cls = best[1]
                    prov = f"iou{best[0]:.2f} c{best[2]:.2f}"
                    stats["matched"] += 1
                else:
                    cls = "upright-cup"
                    prov = f"DEFAULT iou{best[0]:.2f}"
                    stats["default_upright"] += 1
                cls_count[cls] += 1
                x0, y0 = px[:, 0].min(), px[:, 1].min()
                bw, bh = px[:, 0].max() - x0, px[:, 1].max() - y0
                if cls != "upright-cup" or best[0] < args.iou_match or best[2] < 0.6:
                    review.add(crop(img, (x0, y0, bw, bh)),
                               f"{cls}\n{prov}\n{split} #{k}\n{img_path.name[:18]}")
                coords = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
                lines_out.append(f"{CLASS_NAMES.index(cls)} {coords}")
            (dst / split / "labels" / f"{img_path.stem}.txt").write_text(
                "\n".join(lines_out))
        n = len(list((dst / split / "images").glob("*.jpg")))
        print(f"[exo:{split}] images={n}")
    (dst / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
        f"\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    review.flush()
    print(f"[exo] {stats} classes={cls_count} → {dst}")


if __name__ == "__main__":
    views = args.views.split(",")
    if "hand" in views:
        relabel_hand()
    if "exo" in views:
        relabel_exo()
