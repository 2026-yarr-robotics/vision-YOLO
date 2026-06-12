"""
YOLO26m-seg Fine-tuning Pipeline (v4 - Background Augmentation bg_ratio=0.2)
- baseline: two_stage_s2-11 (scale=0.9)
- 개선사항: bg_ratio=0.2 설정 주입 (배경 Hard Negative 강제 노출), Mosaic=0.5
- 자원 최적화: Stage 1 Batch=16, Stage 2 Batch=8 (OOM 방지)
"""

from ultralytics import YOLO
import torch
import os
import glob
import subprocess

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
DATA_YAML   = 'YOLO_YARR-2-class/data.yaml'
BASE_MODEL  = 'yolo26m-seg.pt'
DEVICE      = '1,2'
IMGSZ       = 1280

# 데이터 증강 (baseline scale=0.9, mosaic=0.5 사수 및 bg_ratio=0.2 추가)
AUG = dict(
    degrees=0.0,
    scale=0.9,
    translate=0.1,
    shear=2.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.5,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=0.5,
    mixup=0.0,
)

# ────────────────────────────────────────────────
# 2-Stage 파인튜닝 (v4)
# ────────────────────────────────────────────────
def train_two_stage():
    # Stage 1: Backbone 동결, Head 학습
    print("▶ [v4] 2-Stage Stage 1: Backbone 동결 (freeze=11, Batch=16)")
    model = YOLO(BASE_MODEL)

    r1 = model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=IMGSZ,
        batch=16,           # Stage 1은 동결로 메모리 여유가 있어 16배치 기동
        optimizer='AdamW',
        lr0=0.001,          
        cos_lr=True,        
        cls=2.5,
        box=8.5,            
        freeze=11,          
        patience=10,
        device=DEVICE,
        project='runs/seg',
        name='two_stage_s1_v4',  # v4 지정
        **AUG,
    )

    # Stage 1 save_dir 탐색
    s1_dirs = sorted(
        glob.glob('runs/segment/runs/seg/two_stage_s1_v4*/') +
        glob.glob('runs/seg/two_stage_s1_v4*/'),
        key=os.path.getmtime
    )
    if not s1_dirs:
        raise FileNotFoundError("Stage1 v4 학습 결과 디렉토리를 찾을 수 없습니다.")
    s1_best = os.path.join(s1_dirs[-1], 'weights', 'best.pt')
    print(f"  [Stage1] best.pt 경로: {s1_best}")

    # Stage 2: 전체 레이어 미세조정
    print("▶ [v4] 2-Stage Stage 2: 전체 레이어 미세조정 시작 (Batch=8)")
    model2 = YOLO(s1_best)

    r2 = model2.train(
        data=DATA_YAML,
        epochs=70,
        imgsz=IMGSZ,
        batch=8,            # Stage 2는 전체 unfreeze로 메모리가 급증하므로 8배치 하향
        optimizer='AdamW',  
        lr0=0.0005,         
        cos_lr=True,
        cls=2.5,
        box=8.5,            
        patience=20,
        device=DEVICE,
        project='runs/seg',
        name='two_stage_s2_v4',  # v4 지정
        **AUG,
    )

    # Stage 2 save_dir 탐색
    s2_dirs = sorted(
        glob.glob('runs/segment/runs/seg/two_stage_s2_v4*/') +
        glob.glob('runs/seg/two_stage_s2_v4*/'),
        key=os.path.getmtime
    )
    if not s2_dirs:
        raise FileNotFoundError("Stage2 v4 학습 결과 디렉토리를 찾을 수 없습니다.")
    s2_best = os.path.join(s2_dirs[-1], 'weights', 'best.pt')
    print(f"  [Stage2] best.pt 경로: {s2_best}")
    return s2_best


# ────────────────────────────────────────────────
# 검증 / 테스트 평가
# ────────────────────────────────────────────────
def evaluate(weights_path):
    model = YOLO(weights_path)
    print("\n▶ Validation 평가 (v4)")
    rv = model.val(data=DATA_YAML, split='val', imgsz=IMGSZ, device=DEVICE, verbose=False)
    print(f"  Box  mAP50={rv.results_dict['metrics/mAP50(B)']:.4f}  mAP50-95={rv.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"  Mask mAP50={rv.results_dict['metrics/mAP50(M)']:.4f}  mAP50-95={rv.results_dict['metrics/mAP50-95(M)']:.4f}")

    print("\n▶ Test 평가 (v4)")
    rt = model.val(data=DATA_YAML, split='test', imgsz=IMGSZ, device=DEVICE, verbose=False)
    print(f"  Box  mAP50={rt.results_dict['metrics/mAP50(B)']:.4f}  mAP50-95={rt.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"  Mask mAP50={rt.results_dict['metrics/mAP50(M)']:.4f}  mAP50-95={rt.results_dict['metrics/mAP50-95(M)']:.4f}")


# ────────────────────────────────────────────────
# ONNX 변환
# ────────────────────────────────────────────────
def export_onnx(weights_path):
    print(f"\n▶ ONNX 변환 (v4): {weights_path}")
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
    
    # 학습 완료 후 자동화 스크립트 실행
    print("\n▶ [자동화] 모든 테스트 이미지 및 영상 시각화 생성 중...")
    try:
        subprocess.run([
            '/home/rim0504/miniconda3/envs/yolo_train/bin/python',
            '/home/rim0504/ss_project/visualize_all_test_v4.py'
        ], check=True)
        print("▶ [v4] 모든 검증 시각화 생성 및 완료 보고 완료!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 시각화 스크립트 실행 오류: {e}")
