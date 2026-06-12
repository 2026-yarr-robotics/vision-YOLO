#!/usr/bin/env python3
"""Assemble the real+sim blended training sets.

hand → work/hand_mix/{train,valid,test}/{_annotations.coco.json + *.jpg}
       (the exact roboflow-COCO layout the lightaug/redlite notebook ingests;
       also zipped as hand-eye-view-speed-stack-cup.v3-sim-mix.coco-segmentation.zip
       so the Colab notebook can consume it unmodified)
exo  → work/exo_mix/YOLO_YARR-2-class/{train,valid,test}/{images,labels} + data.yaml
       (the directory name train_segmentation_v4.py's DATA_YAML points at)

Blend: all sim + real oversampled ×k so that real:sim ≈ 1:2 in train (the
temp_task starting ratio; valid/test stay 1×). Oversampled copies are real
file duplicates with a _dupN suffix — the notebook treats them as ordinary
images, no pipeline change.

Run: python3 build_trainsets.py [--real-mult-hand N] [--real-mult-exo N]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ["fallen-cup", "mouth-up-cup", "upright-cup"]

ap = argparse.ArgumentParser()
ap.add_argument("--real-mult-hand", type=int, default=0, help="0 = auto (≈1:2)")
ap.add_argument("--real-mult-exo", type=int, default=0, help="0 = auto (≈1:2)")
ap.add_argument("--zip", action="store_true", help="also write the Colab zip")
args = ap.parse_args()


def build_hand():
    real = ROOT / "dataset" / "real" / "hand3class"
    sim = ROOT / "dataset" / "sim" / "hand"
    out = WORK / "hand_mix"
    if out.exists():
        shutil.rmtree(out)

    n_real = len(json.loads((real / "train" / "_annotations.coco.json")
                            .read_text())["images"])
    n_sim = len(json.loads((sim / "train" / "_annotations.coco.json")
                           .read_text())["images"])
    mult = args.real_mult_hand or max(1, round(n_sim / (2 * n_real)))
    print(f"[hand] real train {n_real} ×{mult} + sim train {n_sim}")

    for split in SPLITS:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        m = mult if split == "train" else 1
        merged = {"info": {"description": "real+sim blend (build_trainsets.py)"},
                  "licenses": [{"id": 1, "name": "CC BY 4.0", "url": ""}],
                  "categories": None, "images": [], "annotations": []}
        next_img, next_ann = 1, 1
        for src, tag, copies in ((real, "real", m), (sim, "sim", 1)):
            coco = json.loads((src / split / "_annotations.coco.json").read_text())
            cats = sorted(coco["categories"], key=lambda c: c["id"])
            names = [c["name"] for c in cats]
            if merged["categories"] is None:
                merged["categories"] = cats
            else:
                assert [c["name"] for c in merged["categories"]] == names, \
                    f"category mismatch: {names}"
            by_img = {}
            for a in coco["annotations"]:
                by_img.setdefault(a["image_id"], []).append(a)
            for im in coco["images"]:
                for k in range(copies):
                    suffix = "" if k == 0 else f"_dup{k}"
                    stem, ext = Path(im["file_name"]).stem, Path(im["file_name"]).suffix
                    fname = f"{stem}{suffix}{ext}"
                    if k == 0:
                        shutil.copy2(src / split / im["file_name"], d / fname)
                    else:
                        shutil.copy2(d / im["file_name"], d / fname)
                    merged["images"].append({**im, "id": next_img, "file_name": fname})
                    for a in by_img.get(im["id"], []):
                        merged["annotations"].append(
                            {**a, "id": next_ann, "image_id": next_img})
                        next_ann += 1
                    next_img += 1
        (d / "_annotations.coco.json").write_text(json.dumps(merged))
        print(f"[hand:{split}] images={len(merged['images'])} "
              f"anns={len(merged['annotations'])}")

    if args.zip:
        z = WORK / "hand-eye-view-speed-stack-cup.v3-sim-mix.coco-segmentation"
        shutil.make_archive(str(z), "zip", root_dir=out)
        print(f"[hand] zip → {z}.zip")


def build_exo():
    real = ROOT / "dataset" / "real" / "exo" / "YOLO_YARR-3class"
    sim = ROOT / "dataset" / "sim" / "exo"
    out = WORK / "exo_mix" / "YOLO_YARR-2-class"
    if out.parent.exists():
        shutil.rmtree(out.parent)

    n_real = len(list((real / "train" / "images").glob("*.jpg")))
    n_sim = len(list((sim / "train" / "images").glob("*.jpg")))
    mult = args.real_mult_exo or max(1, round(n_sim / (2 * n_real)))
    print(f"[exo] real train {n_real} ×{mult} + sim train {n_sim}")

    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        m = mult if split == "train" else 1
        for src, copies in ((real, m), (sim, 1)):
            for img in sorted((src / split / "images").glob("*.jpg")):
                lbl = src / split / "labels" / f"{img.stem}.txt"
                for k in range(copies):
                    suffix = "" if k == 0 else f"_dup{k}"
                    shutil.copy2(img, out / split / "images" / f"{img.stem}{suffix}.jpg")
                    if lbl.exists():
                        shutil.copy2(lbl, out / split / "labels" / f"{img.stem}{suffix}.txt")
        n = len(list((out / split / "images").glob("*.jpg")))
        print(f"[exo:{split}] images={n}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\n"
        f"test: test/images\n\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    print(f"[exo] → {out}")


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    build_hand()
    build_exo()
