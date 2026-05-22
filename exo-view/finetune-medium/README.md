# YOLO26m-seg 파인튜닝 파이프라인

`train_segmentation.py`는 새로운 데이터셋으로 YOLO26m 인스턴스 세그멘테이션 모델을 파인튜닝하고, 검증/테스트 평가 및 ONNX 변환까지 전 과정을 자동화한 스크립트입니다.

---

## 전체 진행 순서

**1. 환경 준비**
- conda 가상환경(`yolo_env`) 생성 및 `ultralytics`, `torch` 설치
- CUDA 12.x 호환 PyTorch 필요 (cu124 권장)

**2. 데이터셋 준비**
- YOLO Segmentation 포맷 라벨 필요 (polygon 좌표)
- `YOLO_YARR-2-class/data.yaml`에 train/val/test 절대경로 및 클래스 정보 설정
- 2-class: 서있는 컵(standing) + 넘어진 컵(knocked-over)

**3. 학습 실행**
```bash
conda activate yolo_env
python train_segmentation.py
```

**4. 평가 및 ONNX 변환**
- 학습 완료 후 Validation / Test mAP 자동 출력
- `best.pt` → `best.onnx` 자동 변환

**5. 추론 시각화**
- 학습 완료 후 `IMG_9503.jpg`, `IMG_9231.jpg`에 대한 추론 결과를 자동 저장
- 저장 경로: `IMG_9503_two_stage.jpg`, `IMG_9231_two_stage.jpg`

---

## 주요 설정값

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DATA_YAML` | `YOLO_YARR-2-class/data.yaml` | 데이터셋 설정 파일 경로 |
| `BASE_MODEL` | `yolo26m-seg.pt` | 사전학습 모델 |
| `DEVICE` | `'0,1'` | 멀티-GPU (단일 GPU는 `0`) |
| `IMGSZ` | `1280` | 입력 이미지 크기 |
| `BATCH` | `8` | 배치 사이즈 (GPU당) |

### 데이터 증강

컵의 서있는/넘어진 방향성을 보존하는 증강 기법만 적용합니다.

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `degrees` | `0.0` | 회전 차단 (방향성 보존) |
| `scale` | `0.5` | 스케일 ±50% |
| `translate` | `0.1` | 이동 10% |
| `shear` | `2.0` | 전단 2도 이내 |
| `fliplr` | `0.5` | 좌우 반전 허용 |
| `flipud` | `0.0` | 상하 반전 차단 |
| `hsv_h` | `0.5` | Hue 변화 |
| `hsv_s` | `0.7` | Saturation 변화 |
| `hsv_v` | `0.4` | Value(밝기) 변화 |

---

## 학습 방식 상세

### 2-Stage Fine-Tuning (항상 적용)

- **Stage 1**: Backbone 동결(`freeze=11`) → Head만 학습
  - optimizer: AdamW, lr=0.001, cos_lr, epochs=30, patience=10
  - cls=2.5, box=8.5 (분류/박스 로스 강화)
- **Stage 2**: 전체 레이어 미세조정
  - optimizer: AdamW, lr=0.0005, cos_lr, epochs=70, patience=20
  - Stage 1의 `best.pt`를 초기 가중치로 사용

> Multi-GPU DDP 환경에서는 학습 결과 경로를 glob으로 자동 탐색합니다.

---

## 실험 결과

| 모델 | Val mAP50 (Box/Mask) | Test mAP50 (Box/Mask) |
|------|----------------------|-----------------------|
| Single-Stage (1-class, lr=0.001) | 0.9937 / 0.9937 | 0.9898 / 0.9898 |
| **2-Stage (1-class)** | **0.9923 / 0.9923** | **0.9913 / 0.9913** |

> 학습 시간: RTX 4090 기준 약 20분 (2-Stage 합산)

---

## 출력 파일 구조

```
runs/seg/
├── two_stage_s1/        ← Stage 1 (Backbone 동결)
│   └── weights/
│       └── best.pt
└── two_stage_s2/        ← Stage 2 (전체 미세조정)
    └── weights/
        ├── best.pt
        └── best.onnx
```

---

## 팁

- GPU 메모리 부족 시 `BATCH`를 줄이거나 `IMGSZ`를 낮추세요
- `data.yaml`의 경로는 절대경로로 설정해야 오류가 없습니다
- 단일 GPU 사용 시 `DEVICE = 0`으로 변경하세요
