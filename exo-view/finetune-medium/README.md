# 🤖 YOLO26m-seg 파인튜닝 파이프라인

`train_segmentation.py`는 새로운 데이터셋으로 YOLO26m 인스턴스 세그멘테이션 모델을 파인튜닝하고, 검증/테스트 평가 및 ONNX 변환까지 전 과정을 자동화한 스크립트입니다.

---

## 📌 전체 진행 순서

**1. 환경 준비**
- conda 가상환경(`yolo_env`) 생성 및 `ultralytics`, `torch` 설치
- CUDA 12.x 호환 PyTorch 필요 (cu124 권장)

**2. 데이터셋 준비**
- YOLO Segmentation 포맷 라벨 필요 (polygon 좌표)
- `data.yaml`에 train/val/test 절대경로 및 클래스 정보 설정
- 서있는 컵 + 넘어진 컵 데이터 포함 (v4, 총 240장)

**3. 학습 방식 선택 (설정 변수)**
- `USE_TWO_STAGE = False` → Single-Stage 학습 (빠르고 간편)
- `USE_TWO_STAGE = True` → 2-Stage 파인튜닝 (소규모 데이터에서 효과적)

**4. 학습 실행**
```bash
conda activate yolo_env
python train_segmentation.py
```

**5. 평가 및 ONNX 변환**
- 학습 완료 후 Validation / Test mAP 자동 출력
- `best.pt` → `best.onnx` 자동 변환

---

## ⚙️ 주요 설정값

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DATA_YAML` | `YOLO_YARR/data.yaml` | 데이터셋 설정 파일 경로 |
| `BASE_MODEL` | `yolo26m-seg.pt` | 사전학습 모델 |
| `DEVICE` | `0` | GPU 번호 (CPU는 `'cpu'`) |
| `IMGSZ` | `1280` | 입력 이미지 크기 |
| `BATCH` | `16` | 배치 사이즈 |
| `USE_TWO_STAGE` | `True` | 2-Stage 파인튜닝 사용 여부 |

### 데이터 증강

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `degrees` | `45` | 회전 ±45° |
| `hsv_h` | `0.015` | Hue 변화 |
| `hsv_s` | `0.7` | Saturation 변화 |
| `hsv_v` | `0.4` | Value(밝기) 변화 |

---

## 🔬 학습 방식 상세

### Single-Stage
- 전체 모델 가중치를 한 번에 학습
- lr=0.001, SGD, epochs=100, patience=20

### 2-Stage Fine-Tuning
- **Stage 1**: Backbone 동결 → Head만 학습 (lr=0.01, 30 epochs)  
  소규모 데이터에서 사전학습 특징을 보존하면서 검출 헤드를 먼저 적응시킴
- **Stage 2**: 전체 레이어 해제 → 낮은 lr로 미세조정 (lr=0.001, 70 epochs, batch=8)  
  Stage 1 가중치를 초기값으로 전체 모델을 세밀하게 조정

> ⚠️ Stage 2는 메모리 사용량이 높으므로 batch=8로 고정

---

## 📊 실험 결과 (CUP 1-class, 240장 학습 / 서있는 컵 + 넘어진 컵)

| 모델 | Val mAP50 (Box/Mask) | Test mAP50 (Box/Mask) |
|------|----------------------|-----------------------|
| Single-Stage (lr=0.001) | 0.9937 / 0.9937 | 0.9898 / 0.9898 |
| **2-Stage** | **0.9923 / 0.9923** | **0.9913 / 0.9913** |

> 학습 시간: RTX 4090 기준 약 20분 (2-Stage 합산)

---

## 📁 출력 파일 구조

```
runs/seg/
├── single_stage/
│   └── weights/
│       ├── best.pt      ← 최고 성능 모델
│       ├── last.pt      ← 마지막 epoch 모델
│       └── best.onnx    ← ONNX 변환 파일
├── two_stage_s1/        ← 2-Stage Stage 1
│   └── weights/
│       └── best.pt
└── two_stage_s2/        ← 2-Stage Stage 2
    └── weights/
        ├── best.pt
        └── best.onnx
```

---

## 💡 팁

- GPU 메모리 부족 시 `BATCH`를 줄이거나 `IMGSZ`를 낮추세요
- `data.yaml`의 경로는 절대경로로 설정해야 오류가 없습니다
- `USE_TWO_STAGE = True`로 두 방식을 모두 실행해 성능을 비교해 보세요
