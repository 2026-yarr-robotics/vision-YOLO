"""
YOLO26m-seg Fine-tuning Pipeline
- Single-Stage 또는 2-Stage 파인튜닝 선택 가능
- 학습 완료 후 검증/테스트 평가 및 ONNX 변환 자동 수행
"""

from ultralytics import YOLO
import torch
import os
import glob

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
DATA_YAML   = 'YOLO_YARR-2-class/data.yaml'
BASE_MODEL  = 'yolo26m-seg.pt'
DEVICE      = '0,1'
IMGSZ       = 1280
BATCH       = 8  # mask_ratio 제거로 메모리가 넉넉하므로 BATCH=8 상향 (각 GPU당 8개 복제)

# 데이터 증강 (컵의 방향성을 보존하는 안전한 증강 기법 적용)
AUG = dict(
    degrees=0.0,     # 회전 차단 (서 있는/넘어진 상태 구분 보전)
    scale=0.5,       # 스케일 증강 (±50%ㄴ 크기 변화)
    translate=0.1,   # Affine - 이동 (가로/세로 10% 이동)
    shear=2.0,       # Affine - 전단 (미세한 기울임 허용, 2도 이내)
    fliplr=0.5,      # 좌우 반전 허용 (서 있는/누운 성질이 유지되므로 안전)
    flipud=0.0,      # 상하 반전 차단 (뒤집힘 방지)
    hsv_h=0.5,     # Color - Hue
    hsv_s=0.7,       # Color - Saturation
    hsv_v=0.4,       # Color - Value (밝기)
)

# ────────────────────────────────────────────────
# 2-Stage 파인튜닝
#   Stage 1: Backbone 동결 → Head만 학습 (높은 lr)
#   Stage 2: 전체 해제 → 낮은 lr로 미세조정
# ────────────────────────────────────────────────
def train_two_stage():
    # Stage 1: Backbone 동결, Head 학습
    print("▶ 2-Stage Stage 1: Backbone 동결 (freeze=11), Head 학습")
    model = YOLO(BASE_MODEL)

    r1 = model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer='AdamW',  # AdamW로 교체
        lr0=0.001,          # 적절한 미세조정용 학습률
        cos_lr=True,        # 코사인 스케줄러 활성화
        cls=2.5,            # 분류 로스 강화 (m 모델 성능 한계 돌파)
        box=8.5,            # 바운딩 박스 타이트 피팅
        freeze=11,          # 0~10번 레이어(Backbone 전체) 완벽 동결
        patience=10,
        device=DEVICE,
        project='runs/seg',
        name='two_stage_s1',
        **AUG,
    )
    # multi-GPU DDP 환경에서는 r1이 None으로 반환되므로 None 체크
    if r1 is not None and hasattr(r1, 'results_dict') and r1.results_dict:
        print(f"  [Stage1] Box mAP50: {r1.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"  [Stage1] Mask mAP50: {r1.results_dict['metrics/mAP50(M)']:.4f}")
    else:
        print("  [Stage1] mAP 결과를 가져올 수 없습니다 (multi-GPU DDP 환경).")

    # Stage 1 save_dir을 직접 glob으로 탐색 (YOLO 실제 저장 경로 기준)
    s1_dirs = sorted(
        glob.glob('runs/segment/runs/seg/two_stage_s1*/') +
        glob.glob('runs/seg/two_stage_s1*/'),
        key=os.path.getmtime
    )
    if not s1_dirs:
        raise FileNotFoundError("Stage1 학습 결과 디렉토리를 찾을 수 없습니다.")
    s1_best = os.path.join(s1_dirs[-1], 'weights', 'best.pt')
    print(f"  [Stage1] best.pt 경로: {s1_best}")

    # Stage 2: 전체 레이어 미세조정
    print("▶ 2-Stage Stage 2: 전체 레이어 미세조정")
    model2 = YOLO(s1_best)

    r2 = model2.train(
        data=DATA_YAML,
        epochs=70,
        imgsz=IMGSZ,
        batch=BATCH,        # 동일하게 4 유지
        optimizer='AdamW',  # AdamW 적용
        lr0=0.0005,         # 전체 튜닝이므로 더 정교하고 낮게 설정
        cos_lr=True,
        cls=2.5,            # 분류 로스 강화 (m 모델 성능 한계 돌파)
        box=8.5,            # 바운딩 박스 타이트 피팅
        patience=20,
        device=DEVICE,
        project='runs/seg',
        name='two_stage_s2',
        **AUG,
    )
    # multi-GPU DDP 환경에서는 r2도 None으로 반환될 수 있음
    if r2 is not None and hasattr(r2, 'results_dict') and r2.results_dict:
        print(f"  [Stage2] Box mAP50: {r2.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"  [Stage2] Mask mAP50: {r2.results_dict['metrics/mAP50(M)']:.4f}")
    else:
        print("  [Stage2] mAP 결과를 가져올 수 없습니다 (multi-GPU DDP 환경).")

    # Stage 2 save_dir을 직접 glob으로 탐색 (YOLO 실제 저장 경로 기준)
    s2_dirs = sorted(
        glob.glob('runs/segment/runs/seg/two_stage_s2*/') +
        glob.glob('runs/seg/two_stage_s2*/'),
        key=os.path.getmtime
    )
    if not s2_dirs:
        raise FileNotFoundError("Stage2 학습 결과 디렉토리를 찾을 수 없습니다.")
    s2_best = os.path.join(s2_dirs[-1], 'weights', 'best.pt')
    print(f"  [Stage2] best.pt 경로: {s2_best}")
    return s2_best


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
    best_weights = train_two_stage()

    evaluate(best_weights)
    export_onnx(best_weights)

    # ────────────────────────────────────────────────
    # 테스트 이미지 추론 및 시각화 저장
    # ────────────────────────────────────────────────
    print("\n▶ 테스트 이미지 추론 및 시각화 저장")
    model = YOLO(best_weights)
    test_images = ['IMG_9503.jpg', 'IMG_9231.jpg']
    for img_path in test_images:
        if os.path.exists(img_path):
            results = model(img_path, imgsz=IMGSZ, device=DEVICE)
            for r in results:
                save_path = img_path.replace('.jpg', '_two_stage.jpg')
                r.save(filename=save_path)
                print(f"  추론 결과 저장 완료: {save_path}")
        else:
            print(f"  경고: {img_path} 이미지가 존재하지 않습니다.")
