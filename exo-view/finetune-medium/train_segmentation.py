"""
YOLO26m-seg Fine-tuning Pipeline
- Single-Stage 또는 2-Stage 파인튜닝 선택 가능
- 학습 완료 후 검증/테스트 평가 및 ONNX 변환 자동 수행
"""

from ultralytics import YOLO
import torch

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
DATA_YAML   = 'YOLO_YARR/data.yaml'
BASE_MODEL  = 'yolo26m-seg.pt'
DEVICE      = 0
IMGSZ       = 1280
BATCH       = 16

# 데이터 증강
AUG = dict(
    degrees=45,      # 회전 ±45°
    hsv_h=0.015,     # Hue
    hsv_s=0.7,       # Saturation
    hsv_v=0.4,       # Value (밝기)
)

USE_TWO_STAGE = True #True로 바꾸면 2-Stage 파인튜닝 실행


# ────────────────────────────────────────────────
# Single-Stage 파인튜닝
# ────────────────────────────────────────────────
def train_single_stage():
    print("▶ Single-Stage 파인튜닝 시작")
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer='SGD',
        lr0=0.001,
        patience=20,
        device=DEVICE,
        project='runs/seg',
        name='single_stage',
        **AUG,
    )
    print(f"  Box  mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"  Mask mAP50: {results.results_dict['metrics/mAP50(M)']:.4f}")
    return str(results.save_dir / 'weights' / 'best.pt')


# ────────────────────────────────────────────────
# 2-Stage 파인튜닝
#   Stage 1: Backbone 동결 → Head만 학습 (높은 lr)
#   Stage 2: 전체 해제 → 낮은 lr로 미세조정
# ────────────────────────────────────────────────
def train_two_stage():
    # Stage 1
    print("▶ 2-Stage Stage 1: Backbone 동결, Head 학습")
    model = YOLO(BASE_MODEL)
    for name, param in model.model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False

    r1 = model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer='SGD',
        lr0=0.01,
        patience=10,
        device=DEVICE,
        project='runs/seg',
        name='two_stage_s1',
        **AUG,
    )
    print(f"  [Stage1] Box mAP50: {r1.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"  [Stage1] Mask mAP50: {r1.results_dict['metrics/mAP50(M)']:.4f}")

    # Stage 2
    print("▶ 2-Stage Stage 2: 전체 레이어 미세조정")
    model2 = YOLO(str(r1.save_dir / 'weights' / 'best.pt'))
    for param in model2.model.parameters():
        param.requires_grad = True

    r2 = model2.train(
        data=DATA_YAML,
        epochs=70,
        imgsz=IMGSZ,
        batch=8,
        optimizer='SGD',
        lr0=0.001,
        patience=20,
        device=DEVICE,
        project='runs/seg',
        name='two_stage_s2',
        **AUG,
    )
    print(f"  [Stage2] Box mAP50: {r2.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"  [Stage2] Mask mAP50: {r2.results_dict['metrics/mAP50(M)']:.4f}")
    return str(r2.save_dir / 'weights' / 'best.pt')


# ────────────────────────────────────────────────
# 검증 / 테스트 평가
# ────────────────────────────────────────────────
def evaluate(weights_path):
    model = YOLO(weights_path)

    print("\n▶ Validation 평가")
    rv = model.val(data=DATA_YAML, split='val', imgsz=IMGSZ, device=DEVICE, verbose=False)
    print(f"  Box  mAP50={rv.results_dict['metrics/mAP50(B)']:.4f}  mAP50-95={rv.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"  Mask mAP50={rv.results_dict['metrics/mAP50(M)']:.4f}  mAP50-95={rv.results_dict['metrics/mAP50-95(M)']:.4f}")

    print("\n▶ Test 평가")
    rt = model.val(data=DATA_YAML, split='test', imgsz=IMGSZ, device=DEVICE, verbose=False)
    print(f"  Box  mAP50={rt.results_dict['metrics/mAP50(B)']:.4f}  mAP50-95={rt.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"  Mask mAP50={rt.results_dict['metrics/mAP50(M)']:.4f}  mAP50-95={rt.results_dict['metrics/mAP50-95(M)']:.4f}")

    return rv, rt


# ────────────────────────────────────────────────
# ONNX 변환
# ────────────────────────────────────────────────
def export_onnx(weights_path):
    print(f"\n▶ ONNX 변환: {weights_path}")
    model = YOLO(weights_path)
    model.export(format='onnx', imgsz=IMGSZ, dynamic=True)
    print("  ONNX 저장 완료")


# ────────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────────
if __name__ == '__main__':
    if USE_TWO_STAGE:
        best_weights = train_two_stage()
    else:
        best_weights = train_single_stage()

    evaluate(best_weights)
    export_onnx(best_weights)
