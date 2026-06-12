#!/usr/bin/env python3
"""Run the exo train_segmentation_v4.py pipeline UNMODIFIED on the blended set.

The original script lives in exo-view/finetune-medium/ and expects
'YOLO_YARR-2-class/data.yaml' relative to the CWD plus DEVICE='1,2' (the lab
training box). We chdir into work/exo_mix (where build_trainsets.py created
that directory) and import the script's functions as-is — the only runtime
adaptation is DEVICE, which is a property of this machine (one RTX 5080),
not of the pipeline. ONNX export is kept but non-fatal.

Run: python3 train_exo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXO_MIX = ROOT / "work" / "exo_mix"
assert (EXO_MIX / "YOLO_YARR-2-class" / "data.yaml").exists(), \
    "run build_trainsets.py first"

os.chdir(EXO_MIX)                  # DATA_YAML resolves exactly like the lab setup
sys.path.insert(0, str(ROOT / "exo-view" / "finetune-medium"))
import train_segmentation_v4 as t  # noqa: E402  (pipeline code, unmodified)

t.DEVICE = "0"                     # env adaptation: single local GPU

best = t.train_two_stage()
t.evaluate(best)
try:
    t.export_onnx(best)
except Exception as exc:           # onnx is an optional artifact here
    print(f"[train_exo] ONNX export skipped: {exc}")
print("BEST:", Path(best).resolve())
