#!/usr/bin/env python3
"""Headless Isaac Sim YOLO-seg dataset generator for the cup-stack digital twin.

Standalone SimulationApp script (probe-tool pattern, no ROS). Rebuilds the
EXACT runtime scene (scene_builder board/ArUco/lighting + M0609+RG2 arm) and
authors a fresh randomized cup layout every frame with the kinematic
frozen-author pattern (cup_reset.py) — no physics settling, so thousands of
frames render fast. Both cameras (hand on link_6, exo) render per frame and
labels come from Replicator instance segmentation + GT cup orientation:

    tilt(body +Z vs world Z) < 15 deg  → upright-cup   (mouth-down stack pose)
    tilt > 165 deg (axis inverted)     → mouth-up-cup  (입구 위)
    75..105 deg (lying)                → fallen-cup
    anything between                   → never authored (transition poses are
                                         class-ambiguous; frame re-rolled)

Output mirrors the real training data conventions exactly so the existing
pipelines consume it unmodified:
  hand → <out>/hand/{train,valid,test}/{_annotations.coco.json + *.jpg}
         (roboflow COCO-seg layout: categories = supercategory id0 + classes)
  exo  → <out>/exo/{train,valid,test}/{images,labels}/* + data.yaml (YOLO-seg)

Split is SCENE-level (same scene's hand+exo land in the same split) with the
8/1/1 ratio derived deterministically from the frame index, so interrupted /
resumed runs are stable. Per-frame GT (poses, classes, visibility) appends to
<out>/meta.jsonl; the COCO jsons are rebuilt from it on every flush, which is
also what makes --resume cheap.

Run (Isaac python, system ROS NOT sourced):
  ~/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      gen_yolo_dataset.py --frames 3000 [--preview 20] [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

# ── CLI (before SimulationApp so --help is instant) ───────────────────────
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--frames", type=int, default=3000, help="scenes to generate (per view)")
ap.add_argument("--out", type=Path, default=None,
                help="output root (default: <vision-YOLO>/dataset/sim)")
ap.add_argument("--playground", type=Path, default=None,
                help="yarr-isaac-playground root (default: auto-detect)")
ap.add_argument("--views", default="hand,exo", help="comma list: hand,exo")
ap.add_argument("--class-names", default="fallen-cup,mouth-up-cup,upright-cup",
                help="YOLO class order (exo txt ids = this order)")
ap.add_argument("--coco-super", default="hand-eye-view-speed-stack-cup",
                help="COCO supercategory name at id 0 (roboflow convention)")
ap.add_argument("--seed", type=int, default=20260613)
ap.add_argument("--preview", type=int, default=0,
                help="also dump N overlay PNGs to <out>/preview for eyeballing")
ap.add_argument("--settle-renders", type=int, default=3,
                help="render steps after authoring before capture")
ap.add_argument("--resume", action="store_true",
                help="skip frame indices already present in meta.jsonl")
ap.add_argument("--min-pixels", type=int, default=300,
                help="drop instances with fewer visible mask pixels")
ap.add_argument("--min-visibility", type=float, default=0.10,
                help="drop instances with visible/expected area below this")
args = ap.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_playground() -> Path:
    cands = []
    if args.playground:
        cands.append(args.playground)
    if env := os.environ.get("PLAYGROUND_ROOT"):
        cands.append(Path(env))
    # committed location: yarr-isaac-playground/tools/gen_yolo_dataset.py
    cands.append(SCRIPT_DIR.parent)
    # dev location: vision-YOLO/data_generator/ next to cup-stack-integration
    cands.append(SCRIPT_DIR.parents[1] / "cup-stack-integration" / "yarr-isaac-playground")
    for c in cands:
        if (c / "scripts" / "scene" / "scene_builder.py").exists():
            return c.resolve()
    raise SystemExit("yarr-isaac-playground not found — pass --playground")


PLAYGROUND = _find_playground()
OUT = (args.out or SCRIPT_DIR.parent / "dataset" / "sim").resolve()
VIEWS = [v.strip() for v in args.views.split(",") if v.strip()]
CLASS_NAMES = [c.strip() for c in args.class_names.split(",") if c.strip()]

# ── Isaac bootstrap ────────────────────────────────────────────────────────
from isaacsim import SimulationApp  # noqa: E402

# FXAA (AA_MODE=2) = the production main.py default; DLSS shifts edges.
app = SimulationApp({"headless": True, "anti_aliasing": 2})

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import cv2  # noqa: E402  (bundled with Isaac python — verified 4.11)
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

_ext = omni.kit.app.get_app().get_extension_manager()
_ext.set_extension_enabled_immediate("isaacsim.robot_setup.assembler", True)
app.update()

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

sys.path.insert(0, str(PLAYGROUND / "scripts"))
from scene import cameras as cameras_mod  # noqa: E402
from scene import robot_loader, scene_builder  # noqa: E402

# ── constants from the runtime config ─────────────────────────────────────
CFG = yaml.safe_load((PLAYGROUND / "config" / "sim_params.yaml").read_text())
INTR = yaml.safe_load((PLAYGROUND / "config" / "d435i.yaml").read_text())

CUP_R_BOT = float(CFG["cups"]["radius_m"])        # 0.039 (wide mouth rim)
CUP_R_TOP = 0.027                                  # narrow closed end (scene_builder)
CUP_H = float(CFG["cups"]["height_m"])            # 0.095
LAYER_H = float(CFG["cups"]["layer_height_m"])    # 0.093 (server grid)
SPACING = float(CFG["pyramid"]["spacing_m"])      # 0.078
MARKER_XY = np.array(CFG["marker"]["offset_xyz"][:2], dtype=float)
EXO_POS = np.array(CFG["exo_camera"]["position"], dtype=float)
EXO_LOOK = np.array(CFG["exo_camera"]["look_at"], dtype=float)
RES = (INTR["image_width"], INTR["image_height"])
FX, FY = float(INTR["fx"]), float(INTR["fy"])

N_CUPS = 14
COLORS = ["red", "blue", "green", "purple"]
# unused cups are made render-INVISIBLE (user finding: under-board parking
# peeked out below the board silhouette at low exo angles); the park pose
# only keeps PhysX happy while hidden
PARK = [(-0.18 + 0.09 * i, 0.0, -0.45) for i in range(N_CUPS)]
# scatter workspace (board x∈[-0.3,1.3] y±0.6; stay in the camera-relevant zone)
WS_X, WS_Y = (0.06, 0.85), (-0.45, 0.45)
LYING_ELEV = math.atan2(CUP_R_BOT - CUP_R_TOP, CUP_H)   # frustum slant ≈ 7.2°
# projected silhouette areas (m²) for the visibility estimate
A_SIDE = (CUP_R_BOT + CUP_R_TOP) * CUP_H
A_TOP = math.pi * CUP_R_BOT ** 2

UPRIGHT_COS = math.cos(math.radians(15.0))
FALLEN_COS = math.cos(math.radians(75.0))


# ── small quaternion helpers (w,x,y,z) ────────────────────────────────────
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qaxis(axis, deg):
    h = math.radians(deg) / 2.0
    s = math.sin(h)
    return np.array([math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s])


def q_body_z(q):
    """World direction of the body +Z axis (mouth → closed bottom)."""
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)])


def q_axes(q):
    """World directions of the body X, Y, Z axes (rotation-matrix columns)."""
    w, x, y, z = (float(v) for v in q)
    ux = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    uy = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    uz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return ux, uy, uz


def settle_z(ori, top_r: float = 0.027, floor_eps: float = 0.001) -> float:
    """DETERMINISTIC floor placement (사용자 2026-06-14 — replaces the flaky
    bbox drop that left fallen cups floating). The cup is a frustum with a
    bottom rim (CUP_R_BOT @ body-z 0) and a top rim (top_r @ body-z CUP_H); a
    convex frustum's lowest contact is always on one of those rim circles. For
    a rim at body height `bz` with radius `r`, its lowest WORLD-z relative to
    the cup origin is bz*uz_z − r*√(ux_z²+uy_z²). Return the origin z that puts
    the lowest rim point on the board (z=floor_eps) — exact for any orientation
    (upright, mouth-up, fallen at any tilt)."""
    ux, uy, uz = q_axes(ori)
    rim_zext = math.sqrt(ux[2] ** 2 + uy[2] ** 2)        # circle's z half-extent factor
    min_rel = min(0.0 * uz[2] - CUP_R_BOT * rim_zext,    # bottom rim (bz=0)
                  CUP_H * uz[2] - top_r * rim_zext)       # top rim (bz=CUP_H)
    return floor_eps - min_rel


# ── scene build (runtime parity) ──────────────────────────────────────────
cfg = dict(CFG)
cfg["cups"] = dict(CFG["cups"])
cfg["cups"]["positions"] = [
    {"xy": [PARK[i][0], PARK[i][1]], "color": COLORS[i % len(COLORS)]}
    for i in range(N_CUPS)
]

world = World(stage_units_in_meters=1.0, physics_dt=float(CFG["physics"]["dt"]))
scene_objects = scene_builder.build(world, cfg)
assembly = robot_loader.load_m0609_with_rg6(world, cfg, app)
cam_cfg = dict(exo_camera=dict(CFG["exo_camera"], enabled="exo" in VIEWS),
               hand_camera=dict(CFG["hand_camera"], enabled="hand" in VIEWS))
cams = cameras_mod.create_cameras(cam_cfg, INTR, assembly["ee_path"])

world.reset()
robot = assembly["robot"]
robot.initialize()
cameras_mod.finalize_cameras(cams, INTR)

stage = omni.usd.get_context().get_stage()
cups = scene_objects["cups"]


def set_kinematic(prim_path: str, enabled: bool) -> None:
    rb = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(prim_path))
    if rb:
        rb.CreateKinematicEnabledAttr().Set(enabled)


def write_usd_pose(cup, pos, ori) -> None:
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(cup.prim_path))
    for op in xf.GetOrderedXformOps():
        name = op.GetOpName()
        if name == "xformOp:translate":
            op.Set(Gf.Vec3d(*(float(v) for v in pos)))
        elif name == "xformOp:orient":
            w, x, y, z = (float(v) for v in ori)
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
            else:
                op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def author_cup(cup, pos, ori, visible: bool = True, settle: bool = True) -> None:
    """Kinematic frozen-author (cup_reset pattern): kinematic FIRST, then the
    USD xform is authoritative for renderer + PhysX kinematic target.

    settle=True (ground cups): the z is RECOMPUTED from the orientation via
    settle_z so the cup rests exactly on the board — deterministic, fixes the
    floating fallen cups (사용자 2026-06-14). settle=False (pyramid tiers):
    keep the authored stacked z so the stack survives."""
    set_kinematic(cup.prim_path, True)
    pos = np.asarray(pos, float)
    if visible and settle:
        pos = np.array([pos[0], pos[1], settle_z(ori)])
    write_usd_pose(cup, pos, ori)
    cup.set_world_pose(position=pos, orientation=np.asarray(ori, float))
    img = UsdGeom.Imageable(stage.GetPrimAtPath(cup.prim_path))
    img.MakeVisible() if visible else img.MakeInvisible()


# semantics: one distinct label per cup so instance ids map back to GT poses
def apply_semantics(prim_path: str, label: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, semantic_label=label, type_label="class")
        return
    except Exception:
        pass
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
        return
    except Exception:
        pass
    from pxr import Semantics
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(label)


for i, cup in enumerate(cups):
    apply_semantics(cup.prim_path, f"cup_{i}")

# instance segmentation + rgb annotators per camera render product
annotators: dict[str, dict] = {}
for name, entry in cams.items():
    camera = entry["camera"]
    rp_path = camera.get_render_product_path()
    iseg = None
    for ann_name in ("instance_segmentation_fast", "instance_segmentation"):
        try:
            iseg = rep.AnnotatorRegistry.get_annotator(
                ann_name, init_params={"colorize": False})
            iseg.attach(rp_path)
            break
        except Exception:
            iseg = None
    if iseg is None:
        raise SystemExit("no instance segmentation annotator available")
    annotators[name] = {"camera": camera, "iseg": iseg}


def render_until_converged(max_steps: int = 60, thresh: float = 0.45,
                           min_steps: int = 6) -> int:
    """Render the scene like the LIVE stack does — until the RTX temporal
    denoiser CONVERGES (사용자 2026-06-14). gen_yolo teleports all cups to a
    fresh scene each frame; with only ~3 render steps the denoiser still carries
    the PREVIOUS scene as translucent GHOSTS and the shadows stay blotchy
    (sim_000046). The live stack renders continuously so its frames are clean.
    Step until the exo frame stops changing (mean abs Δ < thresh) → ghost-free,
    converged frames that match the live render."""
    ref = annotators.get("exo", annotators[next(iter(annotators))])["camera"]
    prev = None
    for s in range(max_steps):
        world.step(render=True)
        if s < min_steps:
            continue
        rgba = ref.get_rgba()
        if hasattr(rgba, "numpy"):
            rgba = rgba.numpy()
        rgba = np.asarray(rgba)
        if rgba.ndim != 3:
            continue
        cur = rgba[:, :, :3].astype(np.int16)
        if prev is not None:
            if float(np.mean(np.abs(cur - prev))) < thresh:
                return s + 1
        prev = cur
    return max_steps

# ── lighting randomizer ────────────────────────────────────────────────────
sun_prim = stage.GetPrimAtPath("/World/sun")
sun = UsdLux.DistantLight(sun_prim)
sun_rot_op = UsdGeom.Xformable(sun_prim).GetOrderedXformOps()[0]
dome = UsdLux.DomeLight.Define(stage, "/World/dome_fill")
dome.CreateIntensityAttr(0.0)   # off by default — live render has no dome fill
board_color_attr = stage.GetPrimAtPath("/World/board").GetAttribute(
    "primvars:displayColor")
BOARD_RGB = np.array([0.55, 0.42, 0.26])


def kelvin_rgb(k: float) -> Gf.Vec3f:
    """Crude blackbody → RGB (good enough for tint jitter)."""
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1.0, 1.292936 * (t - 60) ** -0.1332047)
    g = (min(1.0, 0.3900816 * math.log(t) - 0.6318414) if t <= 66
         else min(1.0, 1.129891 * (t - 60) ** -0.0755148))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else
                             min(1.0, 0.5432068 * math.log(t - 10) - 1.19625))
    return Gf.Vec3f(float(r), float(g), float(b))


def randomize_lights(rng: random.Random) -> None:
    # MATCH THE LIVE main.py RENDER (사용자 지시 2026-06-14). The previous
    # pass added a dome fill + soft wide-angle sun + board tint that skewed the
    # training distribution BRIGHT (μ≈142) and SOFT, away from the live digital
    # twin which uses scene_builder's bare default — DistantLight 3000, sharp
    # (default 0.53° angle), NO dome (black void), untinted board (live μ≈108).
    # The finetuned model then underfit the live-like darker/sharper regime and
    # regressed on the actual deployment target. Center tightly on the live
    # config; photometric robustness comes from the TRAINING augmentation
    # (lightaug/redlite hand, AUG exo), not the renderer.
    sun.GetIntensityAttr().Set(rng.uniform(2700.0, 3300.0))   # live 3000 ±10%
    sun.CreateColorAttr(kelvin_rgb(rng.uniform(5500, 6800)))  # near-neutral
    sun.CreateAngleAttr(rng.uniform(0.4, 1.2))                # sharp, like live
    sun_rot_op.Set(Gf.Vec3f(rng.uniform(-42, -28), rng.uniform(12, 28),
                            rng.uniform(-15, 15)))            # live default ±jitter
    dome.GetIntensityAttr().Set(0.0)                          # no fill — black void
    # untinted board (live is fixed BOARD_RGB)
    board_color_attr.Set([Gf.Vec3f(*BOARD_RGB.tolist())])


# ── arm pose bank (hand-camera viewpoints from valid joint configs) ───────
ARM_DOFS = 6
dof_count = len(robot.dof_names)
xf_cache = UsdGeom.XformCache()


def set_arm(q6) -> None:
    full = np.zeros(dof_count, dtype=float)
    full[:ARM_DOFS] = q6
    robot.set_joint_positions(full)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=full))


def cam_world(prim_path: str):
    """Position + ROS-optical forward/down axes of a USD camera prim."""
    xf_cache.Clear()
    m = xf_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path))
    pos = np.array(m.ExtractTranslation())
    fwd = np.array(m.TransformDir(Gf.Vec3d(0, 0, -1)))   # USD cam looks -Z
    return pos, fwd / np.linalg.norm(fwd)


HAND_CAM_PATH = f"{assembly['ee_path']}/hand_cam" if "hand" in VIEWS else None
J_RANGES = [(-95, 95), (-15, 70), (25, 120), (-50, 50), (35, 145), (-180, 180)]


# strict top-down: the hand cam optical axis must point essentially straight
# DOWN (사용자 지시 2026-06-14 — "지면을 수직으로 내려다봄"). The live recovery/
# pick approach is top-down, so training the hand model only on vertical views
# matches the deployment. -cos(15°)=-0.966 → within ~15° of vertical.
TOPDOWN_COS = -0.966


def build_pose_bank(rng: random.Random, want: int = 800, tries: int = 80000):
    """Random valid joint configs whose hand camera looks ~straight down.
    EE (x,y) and camera height z vary across the bank (사용자 지시: EE pos random,
    방향은 항상 수직). Rejection-sampled; joint_5 (wrist pitch) biased toward the
    range that orients the tool downward to keep the yield workable."""
    bank = []
    for _ in range(tries):
        if len(bank) >= want:
            break
        q = np.radians([rng.uniform(*r) for r in J_RANGES])
        set_arm(q)
        world.step(render=False)
        pos, fwd = cam_world(HAND_CAM_PATH)
        if not (0.25 <= pos[2] <= 0.95) or fwd[2] > TOPDOWN_COS:
            continue                                  # reject non-vertical
        t = -pos[2] / fwd[2]
        hit = pos + t * fwd                           # table intersection (EE x,y)
        if not (0.18 <= t <= 1.10 and 0.0 <= hit[0] <= 0.95 and abs(hit[1]) <= 0.50):
            continue
        xf_cache.Clear()
        ee = np.array(xf_cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(assembly["ee_path"])).ExtractTranslation())
        if ee[2] < 0.20:
            continue
        bank.append({"q": q, "cam": pos, "hit": hit, "ee": ee})
    return bank


# ── scene randomizer ───────────────────────────────────────────────────────
def pyramid_slots(center, n_layers, deg, rng):
    th = math.radians(deg)
    ux, uy = math.cos(th), math.sin(th)
    slots = []
    for layer in range(n_layers):
        n = n_layers - layer
        z = 0.001 + layer * LAYER_H
        for k in range(n):
            off = (k - (n - 1) / 2.0) * SPACING
            slots.append(np.array([center[0] + off * ux,
                                   center[1] + off * uy, z]))
    return slots


def far_enough(xy, placed_xy, d=0.105):
    return all(np.hypot(xy[0] - p[0], xy[1] - p[1]) >= d for p in placed_xy)


def sample_xy(rng, placed_xy, near=None, avoid_marker=True, tries=60):
    for _ in range(tries):
        if near is not None:
            r, a = rng.uniform(0.13, 0.38), rng.uniform(0, 2 * math.pi)
            xy = (near[0] + r * math.cos(a), near[1] + r * math.sin(a))
        else:
            xy = (rng.uniform(*WS_X), rng.uniform(*WS_Y))
        if not (WS_X[0] <= xy[0] <= WS_X[1] and WS_Y[0] <= xy[1] <= WS_Y[1]):
            continue
        if np.hypot(*xy) < 0.20:                      # robot base keep-out
            continue
        if avoid_marker and np.hypot(xy[0] - MARKER_XY[0], xy[1] - MARKER_XY[1]) < 0.13:
            continue
        if far_enough(xy, placed_xy):
            return xy
    return None


def upright_pose(xy, rng):
    return (np.array([xy[0], xy[1], 0.001]),
            qaxis((0, 0, 1), rng.uniform(0, 360)))


def mouthup_pose(xy, rng):
    q = qmul(qaxis((0, 0, 1), rng.uniform(0, 360)), qaxis((1, 0, 0), 180.0))
    return np.array([xy[0], xy[1], CUP_H + 0.001]), q


def fallen_pose(xy, rng):
    """Rest on rim + slant: axis pitched down by the frustum half-angle.

    The cup USD origin sits at ONE END (bottom); a lying cup extends a full
    CUP_H along its axis from there. Author the origin shifted back by half the
    height so the cup BODY is CENTRED on `xy` — then the isotropic min_sep
    clearance check is valid (사용자 2026-06-14: fallen cups were still
    overlapping because the end-origin body reached into neighbours).

    Tilt sign (사용자 2026-06-14): the cup is a frustum — the WIDE end (open
    mouth, r=0.039) and the NARROW end (closed bottom, r=0.027). Lying on its
    side it rests on both rims, so the axis tilts DOWN toward the narrow closed
    bottom (axis_z: wide 0.039 → narrow 0.027). body +Z points wide→narrow, so
    it must pitch DOWN by LYING_ELEV (+, not −) — a − tilted the closed bottom
    UP, the wrong way."""
    elev = math.degrees(LYING_ELEV) + rng.uniform(-2.0, 2.0)
    q = qmul(qaxis((0, 0, 1), rng.uniform(0, 360)),
             qmul(qaxis((0, 1, 0), 90.0 + elev), qaxis((0, 0, 1), rng.uniform(0, 360))))
    z = CUP_R_BOT * math.cos(LYING_ELEV) + 0.002
    ax = q_body_z(q)                              # world dir the body extends along
    cx = xy[0] - 0.5 * CUP_H * float(ax[0])
    cy = xy[1] - 0.5 * CUP_H * float(ax[1])
    return np.array([cx, cy, z]), q


def pose_for(kind: str, xy, rng):
    return {"fallen": fallen_pose, "mouthup": mouthup_pose,
            "upright": upright_pose}[kind](xy, rng)


# Minimum centre-to-centre separation so cups never interpenetrate
# (사용자 2026-06-14): an UPRIGHT/mouth-up footprint is the cup DIAMETER, but a
# FALLEN cup lies on its side so its footprint is the cup HEIGHT (0.095) — use
# that as the clearance for any pair involving a fallen cup.
UPRIGHT_SEP = CUP_R_BOT * 2 + 0.014          # diameter 0.078 + clear gap (≈0.092)
# fallen cup is now CENTRED on its xy (fallen_pose) so half-length+half-length
# along the worst-case axis = CUP_H; +margin for the rounded ends.
FALLEN_SEP = CUP_H + 0.014                     # ≈ 0.109


def min_sep(a: str, b: str) -> float:
    return FALLEN_SEP if (a == "fallen" or b == "fallen") else UPRIGHT_SEP


def place_near(rng, center, kind, placed, r_max=0.20, tries=50):
    """Random XY within r_max of center with kind-aware clearance to every
    already-placed (xy, kind) — packs tight (live staging/recovery layouts)
    WITHOUT interpenetration."""
    for _ in range(tries):
        r, a = rng.uniform(0.0, r_max), rng.uniform(0, 2 * math.pi)
        xy = (center[0] + r * math.cos(a), center[1] + r * math.sin(a))
        if not (WS_X[0] <= xy[0] <= WS_X[1] and WS_Y[0] <= xy[1] <= WS_Y[1]):
            continue
        if np.hypot(*xy) < 0.20:
            continue
        if np.hypot(xy[0] - MARKER_XY[0], xy[1] - MARKER_XY[1]) < 0.10:
            continue
        if all(np.hypot(xy[0] - p[0][0], xy[1] - p[0][1]) >= min_sep(kind, p[1])
               for p in placed):
            return xy
    return None


def valid_center(rng, x_range=(0.18, 0.66), y_range=(-0.30, 0.30)):
    for _ in range(40):
        c = (rng.uniform(*x_range), rng.uniform(*y_range))
        if np.hypot(*c) > 0.24 and np.hypot(c[0] - MARKER_XY[0], c[1] - MARKER_XY[1]) > 0.10:
            return c
    return (0.42, 0.0)


def randomize_scene(rng: random.Random):
    """Author all 14 cups; returns [(cup_idx, pos, quat, class_name)].

    Scene archetypes cover the LIVE OPERATING distribution (사용자: 라이브 캡처),
    not just clean scatter — the v1/v2 finetunes worked on scattered holdouts but
    FAILED on the live staging/recovery scenes (tight clusters of uprights with a
    fallen cup mixed in). Archetypes: dense cluster + fallen_clear(직립열+전도) +
    pyramid + scatter + background."""
    order = list(range(N_CUPS))
    rng.shuffle(order)
    placements = []          # (pos, ori)
    placed_xy = []

    arch = rng.choices(
        ["cluster", "fallen_clear", "pyramid", "scatter", "background"],
        weights=[0.24, 0.16, 0.28, 0.27, 0.05])[0]

    # ── dense cluster: K cups packed tight (live staging/dense table) ──────
    if arch == "cluster":
        center = valid_center(rng)
        n = rng.randint(5, 12)
        r_max = rng.uniform(0.12, 0.22)
        placed = []                                   # (xy, kind) — clearance track
        for _ in range(n):
            kind = rng.choices(["upright", "fallen", "mouthup"],
                               weights=[0.62, 0.26, 0.12])[0]
            xy = place_near(rng, center, kind, placed, r_max=r_max)
            if xy:
                placements.append(pose_for(kind, xy, rng))
                placed.append((xy, kind))
        return _finish_scene(rng, order, placements, placed)

    # ── fallen_clear-style: tight cluster of uprights + 1-2 fallen nearby ──
    if arch == "fallen_clear":
        center = valid_center(rng, x_range=(0.20, 0.42))
        placed = []
        for _ in range(rng.randint(3, 7)):
            xy = place_near(rng, center, "upright", placed, r_max=0.16)
            if xy:
                placements.append(pose_for("upright", xy, rng))
                placed.append((xy, "upright"))
        for _ in range(rng.randint(1, 2)):            # fallen cup(s) — height clearance
            xy = place_near(rng, center, "fallen", placed, r_max=0.26)
            if xy:
                placements.append(pose_for("fallen", xy, rng))
                placed.append((xy, "fallen"))
        if rng.random() < 0.3:
            xy = place_near(rng, center, "mouthup", placed, r_max=0.22)
            if xy:
                placements.append(pose_for("mouthup", xy, rng))
                placed.append((xy, "mouthup"))
        return _finish_scene(rng, order, placements, placed)

    if arch == "background":                          # empty board
        return _finish_scene(rng, order, placements, placed_xy)

    # ── pyramid: 1-4 tier stack, complete/building/collapsed ──────────────
    if arch == "pyramid":
        n_layers = rng.choices([1, 2, 3, 4], weights=[0.12, 0.20, 0.30, 0.38])[0]
        slots, near = [], None
        for _ in range(40):
            c = valid_center(rng)
            s = pyramid_slots(c, n_layers, rng.uniform(0, 360), rng)
            if all(WS_X[0] <= p[0] <= WS_X[1] and WS_Y[0] <= p[1] <= WS_Y[1]
                   and np.hypot(p[0] - MARKER_XY[0], p[1] - MARKER_XY[1]) > 0.13
                   and np.hypot(p[0], p[1]) > 0.22 for p in s):
                slots, near = s, c
                break
        mode = rng.choices(["complete", "building", "collapsed"],
                           weights=[0.45, 0.30, 0.25])[0]
        extra_fallen = 0
        if slots and mode == "building" and len(slots) > 1:
            slots = slots[: rng.randint(1, len(slots) - 1)]
        elif slots and mode == "collapsed" and len(slots) > 1:
            extra_fallen = rng.randint(1, min(4, len(slots) - 1))
            slots = slots[: len(slots) - extra_fallen]
        for p in slots:
            # pyramid tiers are STACKED — keep their authored z (settle=False)
            placements.append((np.array([p[0], p[1], p[2]]),
                               qaxis((0, 0, 1), rng.uniform(0, 360)), False))
            placed_xy.append((p[0], p[1]))
        for _ in range(extra_fallen):
            xy = sample_xy(rng, placed_xy, near=near)
            if xy:
                placements.append(fallen_pose(xy, rng))
                placed_xy.append(xy)
        # a few scattered extras around the stack
        for kind, n in (("fallen", rng.randint(0, 2)), ("upright", rng.randint(0, 2))):
            for _ in range(n):
                xy = sample_xy(rng, placed_xy)
                if xy:
                    placements.append(pose_for(kind, xy, rng))
                    placed_xy.append(xy)
        return _finish_scene(rng, order, placements, placed_xy)

    # ── scatter: spread cups across the board (general robustness) ────────
    n_fallen = rng.choices([0, 1, 2, 3], weights=[0.35, 0.30, 0.20, 0.15])[0]
    n_mouthup = rng.choices([0, 1, 2, 3], weights=[0.22, 0.33, 0.27, 0.18])[0]
    n_upright = rng.randint(0, 5)
    for kind, n in (("fallen", n_fallen), ("mouthup", n_mouthup),
                    ("upright", n_upright)):
        for _ in range(n):
            xy = sample_xy(rng, placed_xy,
                           avoid_marker=not (kind != "upright" and rng.random() < 0.06))
            if xy:
                placements.append(pose_for(kind, xy, rng))
                placed_xy.append(xy)
    if rng.random() < 0.12 and len(placements) < N_CUPS:    # fallen on the marker
        xy = (MARKER_XY[0] + rng.uniform(-0.03, 0.03),
              MARKER_XY[1] + rng.uniform(-0.03, 0.03))
        if far_enough(xy, placed_xy):
            placements.append(fallen_pose(xy, rng))
            placed_xy.append(xy)
    return _finish_scene(rng, order, placements, placed_xy)


def _finish_scene(rng, order, placements, placed_xy):
    """Author cups, derive class from GT orientation, park the rest hidden.
    Placement item = (pos, ori) [ground → settle to board] or (pos, ori, False)
    [pyramid tier → keep stacked z]."""
    gt = []
    for slot, item in enumerate(placements):
        pos, ori = item[0], item[1]
        settle = item[2] if len(item) > 2 else True
        idx = order[slot]
        author_cup(cups[idx], pos, ori, settle=settle)
        cz = float(q_body_z(ori)[2])
        if cz >= UPRIGHT_COS:
            cls = "upright-cup"
        elif cz <= -UPRIGHT_COS:
            cls = "mouth-up-cup"
        elif abs(cz) <= FALLEN_COS:
            cls = "fallen-cup"
        else:
            return None                                # ambiguous — re-roll frame
        gt.append((idx, pos, ori, cls))
    for slot in range(len(placements), N_CUPS):
        idx = order[slot]
        author_cup(cups[idx], np.array(PARK[idx]), np.array([1.0, 0, 0, 0]),
                   visible=False)
    return gt


# ── label extraction ───────────────────────────────────────────────────────
def parse_id_to_cup(info: dict) -> dict[int, int]:
    """Annotator idToLabels → {instance_id: cup_index} (format-tolerant)."""
    out = {}
    for key, val in (info or {}).items():
        m = re.search(r"cup_(\d+)", json.dumps(val) + str(key))
        if m:
            try:
                out[int(key)] = int(m.group(1))
            except ValueError:
                continue
    return out


def expected_pixels(pos, ori, cam_pos) -> float:
    d = float(np.linalg.norm(np.asarray(pos) - np.asarray(cam_pos)))
    if d < 1e-3:
        return 1.0
    view = (np.asarray(pos) - np.asarray(cam_pos)) / d
    c = abs(float(np.dot(q_body_z(ori), view)))        # axis · view ray
    area_m2 = c * A_TOP + (1.0 - c) * A_SIDE
    return area_m2 * FX * FY / (d * d)


def mask_polygons(mask: np.ndarray, min_area: float = 60.0):
    m8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(pts) >= 3:
            polys.append(pts.astype(float))
    polys.sort(key=lambda p: cv2.contourArea(p.astype(np.int32)), reverse=True)
    return polys


def extract_instances(view: str, gt, cam_pos):
    """Returns (labels, drop_reason). labels = list of dicts per visible cup."""
    ann = annotators[view]
    rgba = ann["camera"].get_rgba()
    if hasattr(rgba, "numpy"):
        rgba = rgba.numpy()
    rgb = np.asarray(rgba)[:, :, :3]
    data = ann["iseg"].get_data()
    if isinstance(data, dict):
        seg = np.asarray(data["data"]).squeeze()
        info = data.get("info", {}).get("idToLabels", {})
    else:
        seg = np.asarray(data).squeeze()
        info = {}
    id2cup = parse_id_to_cup(info)
    labels = []
    for idx, pos, ori, cls in gt:
        ids = [iid for iid, ci in id2cup.items() if ci == idx]
        if not ids:
            continue
        mask = np.isin(seg, ids)
        vis_px = int(mask.sum())
        if vis_px < args.min_pixels:
            continue
        ratio = vis_px / max(expected_pixels(pos, ori, cam_pos), 1.0)
        if ratio < args.min_visibility:
            continue
        polys = mask_polygons(mask)
        if not polys:
            continue
        ys, xs = np.nonzero(mask)
        bbox = [float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
        labels.append({"cup": idx, "cls": cls, "pixels": vis_px,
                       "visibility": round(min(ratio, 2.0), 3),
                       "bbox": bbox, "polys": [p.reshape(-1).tolist() for p in polys]})
    return rgb, labels


# ── writers ────────────────────────────────────────────────────────────────
SPLITS = ("train", "valid", "test")


def split_of(idx: int) -> str:
    h = (idx * 2654435761) % 10
    return "train" if h < 8 else ("valid" if h == 8 else "test")


def out_dirs():
    for v in VIEWS:
        if v == "hand":
            for s in SPLITS:
                (OUT / "hand" / s).mkdir(parents=True, exist_ok=True)
        else:
            for s in SPLITS:
                (OUT / "exo" / s / "images").mkdir(parents=True, exist_ok=True)
                (OUT / "exo" / s / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    if "exo" in VIEWS:
        (OUT / "exo" / "data.yaml").write_text(
            "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
            f"\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")


def write_exo(idx: int, split: str, rgb, labels) -> str:
    name = f"sim_{idx:06d}_exo"
    h, w = rgb.shape[:2]
    cv2.imwrite(str(OUT / "exo" / split / "images" / f"{name}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = []
    for lab in labels:
        poly = np.asarray(lab["polys"][0], float).reshape(-1, 2)   # largest
        poly[:, 0] = np.clip(poly[:, 0] / w, 0, 1)
        poly[:, 1] = np.clip(poly[:, 1] / h, 0, 1)
        if len(poly) < 3:
            continue
        coords = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
        lines.append(f"{CLASS_NAMES.index(lab['cls'])} {coords}")
    (OUT / "exo" / split / "labels" / f"{name}.txt").write_text("\n".join(lines))
    return f"{name}.jpg"


def write_hand(idx: int, split: str, rgb) -> str:
    name = f"sim_{idx:06d}_hand.jpg"
    cv2.imwrite(str(OUT / "hand" / split / name),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return name


def rebuild_coco(meta_path: Path) -> None:
    cats = [{"id": 0, "name": args.coco_super, "supercategory": "none"}] + [
        {"id": i + 1, "name": n, "supercategory": args.coco_super}
        for i, n in enumerate(CLASS_NAMES)
    ]
    records: dict[int, dict] = {}
    with meta_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("view") == "hand":
                records[rec["idx"]] = rec       # dedup: last write wins
    per_split = {s: {"images": [], "annotations": []} for s in SPLITS}
    ann_id = {s: 1 for s in SPLITS}
    for rec in records.values():
        s = rec["split"]
        per_split[s]["images"].append(
            {"id": rec["idx"], "file_name": rec["file"], "width": RES[0],
             "height": RES[1], "license": 1,
             "date_captured": "2026-06-13T00:00:00+00:00"})
        for lab in rec["labels"]:
            per_split[s]["annotations"].append({
                "id": ann_id[s], "image_id": rec["idx"],
                "category_id": 1 + CLASS_NAMES.index(lab["cls"]),
                "bbox": lab["bbox"], "area": lab["pixels"],
                "segmentation": lab["polys"], "iscrowd": 0})
            ann_id[s] += 1
    for s in SPLITS:
        body = {"info": {"description": "isaac sim generated (gen_yolo_dataset.py)"},
                "licenses": [{"id": 1, "name": "CC BY 4.0", "url": ""}],
                "categories": cats, **per_split[s]}
        (OUT / "hand" / s / "_annotations.coco.json").write_text(json.dumps(body))


def draw_preview(idx: int, view: str, rgb, labels) -> None:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    palette = {"fallen-cup": (0, 0, 255), "mouth-up-cup": (0, 255, 255),
               "upright-cup": (0, 200, 0)}
    for lab in labels:
        col = palette[lab["cls"]]
        for poly in lab["polys"]:
            pts = np.asarray(poly, float).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, col, 2)
        x, y = int(lab["bbox"][0]), int(lab["bbox"][1])
        cv2.putText(img, f"{lab['cls']} v{lab['visibility']:.2f}",
                    (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.imwrite(str(OUT / "preview" / f"{idx:06d}_{view}.png"), img)


# ── main loop ─────────────────────────────────────────────────────────────
def main() -> None:
    out_dirs()
    meta_path = OUT / "meta.jsonl"
    done = set()
    if args.resume and meta_path.exists():
        with meta_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    continue
    elif meta_path.exists() and not args.resume:
        meta_path.unlink()

    boot_rng = random.Random(args.seed)
    bank = []
    if "hand" in VIEWS:
        print("[gen] building hand-camera pose bank …", flush=True)
        bank = build_pose_bank(boot_rng)
        print(f"[gen] pose bank: {len(bank)} valid configs", flush=True)
        if len(bank) < 50:
            raise SystemExit("pose bank too small — joint ranges need tuning")

    # park everything once so warmup frames are clean
    for i, cup in enumerate(cups):
        author_cup(cup, np.array(PARK[i]), np.array([1.0, 0, 0, 0]),
                   visible=False)
    for _ in range(20):                                # shader/SDG warmup
        world.step(render=True)

    stats = {c: 0 for c in CLASS_NAMES}
    written = 0
    meta_f = meta_path.open("a")
    for idx in range(args.frames):
        if idx in done:
            continue
        rng = random.Random((args.seed << 20) ^ idx)
        gt = None
        for _ in range(5):
            gt = randomize_scene(rng)
            if gt is not None:
                break
        if gt is None:
            continue
        randomize_lights(rng)

        hand_entry = None
        if "hand" in VIEWS:
            rng.shuffle(bank)
            want_cups = bool(gt) and rng.random() < 0.85
            fallback = None
            for cand in bank:
                tip = cand["ee"][2] - 0.18
                clear = all(tip > p[2] + CUP_H + 0.02
                            or np.hypot(cand["ee"][0] - p[0],
                                        cand["ee"][1] - p[1]) > 0.13
                            for _, p, _, _ in gt)
                if not clear:
                    continue
                fallback = fallback or cand
                # bias the view onto the cups: empty top-down board frames
                # dominated the first smoke batch otherwise
                if not want_cups or any(
                        np.hypot(cand["hit"][0] - p[0], cand["hit"][1] - p[1]) < 0.28
                        for _, p, _, _ in gt):
                    hand_entry = cand
                    break
            set_arm((hand_entry or fallback or bank[0])["q"])
        if "exo" in VIEWS:
            if rng.random() < 0.65:
                pos = EXO_POS + np.array([rng.gauss(0, 0.10), rng.gauss(0, 0.10),
                                          rng.gauss(0, 0.06)])
                look = EXO_LOOK + np.array([rng.gauss(0, 0.12), rng.gauss(0, 0.12),
                                            rng.gauss(0, 0.04)])
            else:
                r, a = rng.uniform(0.9, 1.7), math.radians(rng.uniform(-150, -10))
                pos = np.array([0.45 + r * math.cos(a), r * math.sin(a),
                                rng.uniform(0.30, 0.90)])
                look = np.array([rng.uniform(0.25, 0.6), rng.uniform(-0.2, 0.2), 0.0])
            pos[2] = max(pos[2], 0.25)
            cams["exo"]["camera"].set_world_pose(
                pos, scene_builder.look_at_quat_wxyz(pos, look), camera_axes="ros")

        conv_steps = render_until_converged()       # like live: render till clean

        split = split_of(idx)
        gt_json = [{"cup": i, "pos": [round(float(v), 4) for v in p],
                    "quat": [round(float(v), 5) for v in o], "cls": c}
                   for i, p, o, c in gt]
        for view in VIEWS:
            cam_pos, _ = cam_world(HAND_CAM_PATH if view == "hand"
                                   else "/World/exo_cam")
            rgb, labels = extract_instances(view, gt, cam_pos)
            if view == "hand":
                fname = write_hand(idx, split, rgb)
            else:
                fname = write_exo(idx, split, rgb, labels)
            for lab in labels:
                stats[lab["cls"]] += 1
            meta_f.write(json.dumps({
                "idx": idx, "view": view, "split": split, "file": fname,
                "labels": labels, "gt": gt_json}) + "\n")
            if idx < args.preview:
                draw_preview(idx, view, rgb, labels)
        meta_f.flush()
        written += 1
        if written % 50 == 0 or idx == args.frames - 1:
            rebuild_coco(meta_path)
            print(f"[gen] {idx + 1}/{args.frames} scenes | instances {stats}",
                  flush=True)

    meta_f.close()
    rebuild_coco(meta_path)
    print(f"[gen] DONE — {written} new scenes, instance totals {stats}", flush=True)
    print(f"[gen] output: {OUT}", flush=True)


try:
    main()
finally:
    app.close()
