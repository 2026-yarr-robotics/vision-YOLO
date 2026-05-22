# Hand-eye View YOLOv26m-seg Fine-tuning Pipeline (Roboflow 0518, Medium Only)

이 디렉토리는 **hand-eye view 이미지에서 speed stack cup을 instance segmentation**하기 위해 YOLOv26m-seg medium 모델만 파인튜닝하는 Colab 노트북을 정리한 공간입니다.
Roboflow에서 병합 및 augmentation까지 완료된 COCO segmentation dataset을 사용한다고 가정하며, YOLO segmentation format 변환 → medium 모델 학습 → validation/test 평가 → test 전체 이미지 정성 결과 저장 → 보고서용 표/플롯 생성까지 한 번에 수행할 수 있도록 구성되어 있습니다.

---

## 1. Notebook Overview

사용 노트북:

```text
hand_eye_view_yolo26m_seg_medium_only_0518_coco.ipynb
```

전체 흐름은 다음과 같습니다.

```text
Environment setup
→ Global configuration
→ Google Drive dataset copy
→ COCO dataset check
→ COCO segmentation to YOLO segmentation conversion
→ YOLO label verification
→ (optional) Offline augmentation
→ YOLOv26m-seg medium training
→ best.pt validation / test evaluation
→ Predict all test images (instance segmentation + bbox)
→ Build validation / test result tables (CSV, XLSX)
→ (optional) Compare with baseline (original dataset) experiment
→ Trade-off plots and report-ready summary
→ Display qualitative prediction images
```

---

## 2. Purpose

이 파이프라인의 목적은 다음과 같습니다.

- Hand-eye camera view에서 cup 영역을 segmentation하기 위한 **YOLOv26m-seg medium 모델** 학습
- Roboflow에서 augmentation까지 완료된 COCO segmentation dataset을 YOLO segmentation 학습 형식으로 변환
- `best.pt`, validation/test 지표 표, test 전체 이미지의 정성적 segmentation 결과를 모두 Google Drive에 자동 저장
- 보고서 작성에 바로 사용할 수 있는 CSV/XLSX 표와 trade-off 플롯, 자동 해석 요약 생성
- 이후 ROS2 perception node 또는 cup pose estimation node에서 사용할 수 있는 모델 산출

핵심 원칙:

- Roboflow에서 이미 augmentation을 적용한 dataset version을 사용한다고 가정합니다.
- 따라서 이 노트북의 추가 offline augmentation은 **기본적으로 비활성화**되어 있습니다 (`USE_OFFLINE_AUGMENTATION = False`).
- medium 모델만 학습하기 위해 `MODEL_SIZES = ['m']`로 고정되어 있습니다.

---

## 3. Expected Dataset Structure

Google Drive에는 Roboflow의 **COCO Segmentation** export가 아래와 같이 준비되어 있다고 가정합니다.

```text
/content/drive/MyDrive/hand-eye-view-speed-stack-cup.0518.coco-segmentation/
├── train/
│   ├── _annotations.coco.json
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── valid/                 # 또는 val/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

노트북의 기본 dataset path는 다음과 같습니다.

```python
DRIVE_DATASET = Path('/content/drive/MyDrive/hand-eye-view-speed-stack-cup.0518.coco-segmentation')
```

데이터셋 위치가 다르면 `2. Global configuration` 셀에서 이 경로만 수정하면 됩니다.

---

## 4. Main Configuration

노트북의 핵심 설정값은 `2. Global configuration` 셀에 모아두었습니다.

```python
IMG_SIZE = 1280
EPOCHS = 250
PATIENCE = 60

MODEL_SIZES = ['m']            # medium만 학습
MODEL_NAME_MAP = {
    'n': 'YOLOv26n-seg',
    's': 'YOLOv26s-seg',
    'm': 'YOLOv26m-seg',
    'l': 'YOLOv26l-seg',
}

USE_OFFLINE_AUGMENTATION = False
AUG_COPIES_PER_IMAGE = 0
AUG_FACTOR = 1 + AUG_COPIES_PER_IMAGE
REBUILD_AUGMENTED_DATASET = True
SAVE_AUGMENTED_DATASET_TO_DRIVE = False
AUG_SEED = 42

TRAIN_BATCH = 0.70
EVAL_BATCH = 8
PRED_CONF = 0.25
PRED_IOU = 0.70

SKIP_TRAIN_IF_DRIVE_BEST_EXISTS = False
SAVE_RUN_ZIP = True
```

EXPERIMENT_TAG는 자동으로 다음 규칙으로 결정됩니다.

```python
if USE_OFFLINE_AUGMENTATION:
    EXPERIMENT_TAG = f'local_augx{AUG_FACTOR}'
else:
    EXPERIMENT_TAG = 'roboflow_0518_medium'
```

학습 run name 및 가중치 파일명은 다음 패턴으로 생성됩니다.

```text
hand_eye_view_yolo26{size}_seg_{IMG_SIZE}_epoch{EPOCHS}_{EXPERIMENT_TAG}_a100
hand_eye_view_yolo26m_seg_1280_epoch250_roboflow_0518_medium_a100_best.pt
hand_eye_view_yolo26m_seg_1280_epoch250_roboflow_0518_medium_a100_last.pt
```

A100 환경에서 `yolo26m-seg.pt`를 baseline weight로 사용합니다. OOM이 발생하면 `TRAIN_BATCH`를 정수 batch로 낮추세요 (예: `8`, `6`, `4`).

---

## 5. Pipeline Steps

### Step 1. Environment Setup

Colab 환경에서 GPU 상태를 확인하고 필요한 패키지를 설치합니다.

주요 패키지:

- `ultralytics`
- `opencv-python-headless`
- `pycocotools`
- `pyyaml`
- `matplotlib`
- `openpyxl` (CSV/XLSX 저장)
- `albumentations` (optional offline augmentation)

Colab 기본 환경과 충돌을 피하기 위해 `pandas==2.2.2`로 고정합니다.

```python
!pip install -q -U ultralytics opencv-python-headless pycocotools pyyaml matplotlib openpyxl albumentations
!pip install -q "pandas==2.2.2"
```

정상적으로 설정되면 다음과 같이 GPU가 인식됩니다.

```text
torch cuda available: True
device count: 1
GPU: NVIDIA A100-SXM4-40GB
```

---

### Step 2. Dataset Copy

Google Drive에 있는 Roboflow COCO segmentation dataset을 Colab local path로 복사합니다.

```text
/content/drive/MyDrive/hand-eye-view-speed-stack-cup.0518.coco-segmentation
→ /content/coco_dataset_1280
```

Colab에서 직접 학습하면 Drive I/O가 느릴 수 있으므로, 학습 전에 local storage로 복사하는 구조입니다.

---

### Step 3. COCO Dataset Check

다음 split을 자동으로 탐색합니다.

```python
POSSIBLE_SPLITS = ['train', 'valid', 'val', 'test']
```

필수 조건은 다음과 같습니다.

- `train/_annotations.coco.json` 존재
- `valid/_annotations.coco.json` 또는 `val/_annotations.coco.json` 존재

COCO json 내부의 `categories`를 읽어서 YOLO class name으로 사용합니다.

---

### Step 4. COCO Segmentation → YOLO Segmentation Conversion

COCO annotation(polygon 또는 RLE)을 YOLO segmentation label format으로 변환합니다.

변환 후 생성되는 구조는 다음과 같습니다.

```text
/content/hand_eye_view_yolo26_seg_1280/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   ├── valid/
│   └── test/
└── data.yaml
```

YOLO segmentation label은 다음 형식을 가집니다.

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

좌표는 모두 `0~1` 범위로 정규화됩니다.
RLE format은 `pycocotools`로 binary mask를 만든 뒤 contour를 polygon으로 변환합니다.

---

### Step 5. Label Verification

학습 전에 label 파일이 정상적으로 생성되었는지 확인합니다.

확인 항목:

- label `.txt` 파일 개수
- 비어 있지 않은 label 파일 개수
- 첫 번째 label line
- 한 줄에 포함된 값 개수

Segmentation label이라면 보통 값 개수가 5개보다 많아야 합니다.
값 개수가 5개라면 detection bbox format일 가능성이 높습니다.

train/valid/test 각각에 대해 검증을 수행합니다.

---

### Step 6. Optional Offline Augmentation

기본값은 `USE_OFFLINE_AUGMENTATION = False`입니다.
Roboflow에서 이미 augmentation된 dataset version을 사용한다고 가정하기 때문입니다.

추가로 Albumentations 기반 offline augmentation을 한 번 더 적용하고 싶을 때만 `USE_OFFLINE_AUGMENTATION = True`로 변경하세요.

활성화하면 다음과 같이 동작합니다.

- train split만 복제 및 증강한 새 dataset directory를 생성합니다.
- 원본 1장당 `AUG_COPIES_PER_IMAGE`장의 새 augmentation 이미지를 추가합니다.
- 결과 dataset 경로:

```text
/content/hand_eye_view_yolo26_seg_1280_augx{AUG_FACTOR}/
```

학습 시 사용하는 `data.yaml`은 자동으로 augmented version으로 전환됩니다.

---

### Step 7. YOLOv26m-seg Medium Training

medium 모델만 학습합니다 (`MODEL_SIZES = ['m']`). 기본 학습 설정은 다음과 같습니다.

```python
model.train(
    data=str(DATA_YAML),
    task='segment',

    epochs=250,
    imgsz=1280,
    batch=0.70,
    patience=60,

    optimizer='auto',
    cos_lr=True,
    warmup_epochs=5.0,
    weight_decay=0.0005,

    overlap_mask=True,
    mask_ratio=2,

    hsv_h=0.015, hsv_s=0.45, hsv_v=0.30,
    degrees=5.0, translate=0.05, scale=0.30,
    shear=0.0, perspective=0.0003,
    flipud=0.0, fliplr=0.5,

    mosaic=0.30, close_mosaic=20,
    mixup=0.0, copy_paste=0.0,

    device=DEVICE,
    workers=8,
    amp=True,
    cache='ram',
    plots=True,
    save=True,
    save_period=10,

    project=str(RUN_PROJECT),
    name=run_name,
    exist_ok=True,
)
```

A100 환경에서는 `batch=0.70` 자동 batch를 사용합니다.
OOM이 발생하면 다음처럼 정수 batch로 낮추면 됩니다.

```python
TRAIN_BATCH = 8
# or
TRAIN_BATCH = 6
# or
TRAIN_BATCH = 4
```

학습 결과는 다음 경로에 저장됩니다.

```text
/content/runs/segment/hand_eye_view_yolo26m_seg_1280_epoch250_roboflow_0518_medium_a100/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
└── ...
```

학습이 끝나면 `best.pt`와 `last.pt`가 Drive로 자동 복사되고, 필요 시 `SAVE_RUN_ZIP = True`이면 run folder 전체가 zip으로 백업됩니다.

`SKIP_TRAIN_IF_DRIVE_BEST_EXISTS = True`로 설정하면 Drive에 이미 best.pt가 있는 경우 학습을 건너뛰고 평가만 다시 수행할 수 있습니다.

---

### Step 8. Validation and Test Evaluation

학습이 끝난 뒤 `best.pt`를 기준으로 validation과 test를 모두 평가합니다.

수집하는 주요 metric은 다음과 같습니다.

| Metric | Meaning |
|---|---|
| `box_P`, `box_R` | Box precision, recall |
| `box_mAP50`, `box_mAP50_95` | Box mAP50, mAP50-95 |
| `mask_P`, `mask_R` | Mask precision, recall |
| `mask_mAP50`, `mask_mAP50_95` | Mask mAP50, mAP50-95 |
| `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms` | 1장당 처리 시간 |
| `fps_total`, `fps_inference_only` | 처리량 |
| `params_M`, `GFLOPs` | 모델 파라미터/연산량 |

평가 결과는 `val` / `test` split별로 한 행씩 누적되어, 최종적으로 DataFrame과 CSV/XLSX 표로 저장됩니다.

---

### Step 9. Predict All Test Images

`best.pt`를 사용해 **test set 전체 이미지**에 대해 instance segmentation + bbox를 예측합니다.

```python
model.predict(
    source=str(test_img_dir),
    task='segment',
    imgsz=IMG_SIZE,
    conf=PRED_CONF,
    iou=PRED_IOU,
    save=True,
    save_txt=True,
    save_conf=True,
    retina_masks=True,
    ...
)
```

저장 위치:

```text
/content/runs/segment_predict/hand_eye_view_yolo26m_seg_1280_epoch250_roboflow_0518_medium_a100_test_predict_all/
```

이 폴더는 자동으로 Google Drive (`DRIVE_PRED_DIR`) 아래에도 복사되어 보고서 자료로 사용할 수 있습니다.

---

### Step 10. Build Result Tables

전체 결과를 정렬해서 보고서용 표로 정리합니다.

저장되는 표:

```text
tables/all_metrics_val_test.csv / .xlsx
tables/validation_metrics.csv / .xlsx
tables/test_metrics.csv / .xlsx
tables/validation_metrics_compact.csv / .xlsx
tables/test_metrics_compact.csv / .xlsx
tables/all_metrics_intermediate.csv / .xlsx
```

compact 표는 다음 column만 포함합니다.

```text
model, split, params_M, GFLOPs,
box_P, box_R, box_mAP50, box_mAP50_95,
mask_P, mask_R, mask_mAP50, mask_mAP50_95,
inference_ms, total_ms, fps_total
```

---

### Step 11. Optional Baseline Comparison

기존 원본 데이터셋 실험 결과(`test_metrics_compact.csv`)가 Drive에 남아 있다면, 이 셀에서 자동으로 **원본 vs 현재 실험** 비교 표와 mask mAP50-95 변화량 bar plot을 생성합니다.

비교 결과는 다음 경로에 저장됩니다.

```text
tables/original_vs_augmented_val_compare.csv / .xlsx
tables/original_vs_augmented_test_compare.csv / .xlsx
```

baseline csv가 없으면 자동으로 SKIP됩니다.

---

### Step 12. Trade-off Plots and Report Summary

다음 시각화를 생성하고 Drive에 저장합니다.

```text
tables/test_map50_95_by_model.png
tables/test_accuracy_speed_tradeoff.png
tables/test_accuracy_model_size_tradeoff.png
```

또한 보고서 작성용 자동 해석 텍스트를 출력합니다.

- Best mask mAP50-95 모델
- Best box mAP50-95 모델
- 가장 빠른 모델 (FPS 기준)
- 가장 작은 모델 (params 기준)
- medium 모델에 대한 상세 해석 (Box/Mask P, R, mAP50, mAP50-95, 추론 속도)
- 정확도/속도 trade-off 해석

---

### Step 13. Display Qualitative Prediction Images

`DRIVE_PRED_DIR` 아래의 test prediction 이미지 중 일부 (`MAX_DISPLAY_PER_MODEL = 5`)를 Colab에 직접 표시합니다.
전체 이미지를 보려면 이 값을 늘리면 됩니다.

---

## 6. Output Summary

| Output | Path | Description |
|---|---|---|
| YOLO dataset | `/content/hand_eye_view_yolo26_seg_1280` | COCO에서 변환된 YOLO-seg dataset |
| data yaml | `/content/hand_eye_view_yolo26_seg_1280/data.yaml` | YOLO 학습용 dataset yaml |
| local best | `/content/runs/segment/.../weights/best.pt` | validation 기준 최적 모델 |
| local last | `/content/runs/segment/.../weights/last.pt` | 마지막 epoch 모델 |
| drive best | `weights/hand_eye_view_yolo26m_seg_1280_epoch250_roboflow_0518_medium_a100_best.pt` | Drive 백업된 best 모델 |
| drive last | `weights/hand_eye_view_yolo26m_seg_1280_epoch250_roboflow_0518_medium_a100_last.pt` | Drive 백업된 last 모델 |
| run zip | `runs_zip/...zip` | 전체 run folder zip 백업 |
| validation/test metrics | `tables/*.csv / *.xlsx` | 보고서용 평가 지표 표 |
| trade-off plots | `tables/*.png` | mAP50-95 / 속도 / 모델 크기 plot |
| test predict images | `predict_images/{run}_test_predict_all/` | test 전체 이미지 정성적 결과 |

Drive 저장 root는 다음과 같습니다.

```text
/content/drive/MyDrive/hand_eye_view_yolo26m_seg_result_epoch250_roboflow_0518_medium/
├── weights/
├── runs_zip/
├── predict_images/
└── tables/
```

---

## 7. Common Issues

### 1. `label file count: 0`

원인:

- `YOLO_DATASET_DIR` 경로가 잘못됨
- COCO 변환 셀이 실행되지 않음
- 원본 dataset에 `_annotations.coco.json`이 없음
- `train`, `valid`, `test` 폴더 구조가 예상과 다름

해결:

- `4. Check COCO dataset structure` 셀에서 split이 정상 탐색되는지 확인
- `5. Convert COCO segmentation to YOLO segmentation` 셀을 다시 실행
- `YOLO_DATASET_DIR / labels / train` 경로에 `.txt` 파일이 생성됐는지 확인

---

### 2. `pandas` dependency conflict

Colab에서 다음과 같은 경고가 나올 수 있습니다.

```text
google-colab requires pandas==2.2.2, but you have pandas 3.x.x
```

해결:

```python
!pip install -q "pandas==2.2.2"
```

그 후 runtime을 restart하고 첫 번째 셀부터 다시 실행하는 것이 안전합니다.

---

### 3. CUDA OOM

A100이 아닌 GPU나 Colab 환경에 따라 memory 부족이 발생할 수 있습니다.

해결:

```python
TRAIN_BATCH = 8
```

그래도 부족하면 다음 순서로 낮춥니다.

```python
TRAIN_BATCH = 6
TRAIN_BATCH = 4
TRAIN_BATCH = 2
```

`EVAL_BATCH`도 동일하게 낮출 수 있습니다.

---

### 4. `.pt` file looks like a zip file

PyTorch `.pt` checkpoint는 내부적으로 zip 기반 구조를 사용할 수 있습니다.
따라서 Google Drive나 파일 탐색기에서 압축 파일처럼 보여도 정상입니다.

해결:

- 압축을 풀지 않습니다.
- 아래처럼 `.pt` 파일 경로를 그대로 사용합니다.

```python
model = YOLO('/content/drive/MyDrive/hand_eye_view_yolo26m_seg_result_epoch250_roboflow_0518_medium/weights/hand_eye_view_yolo26m_seg_1280_epoch250_roboflow_0518_medium_a100_best.pt')
```

---

### 5. baseline csv가 없어 비교가 SKIP되는 경우

`Step 11`의 비교 셀은 이전 원본 데이터셋 실험의 `test_metrics_compact.csv`가 있다고 가정합니다.
존재하지 않으면 자동으로 `[SKIP]`이 출력되고 비교는 생략됩니다. 정상 동작입니다.

---

## 8. Quick Usage

Colab에서 노트북을 열고 아래 순서대로 실행합니다.

```text
1. Environment setup
2. Global configuration
3. Copy dataset from Drive to Colab local
4. Check COCO dataset structure
5. Convert COCO segmentation to YOLO segmentation
6. Verify YOLO labels safely
7. (optional) Offline augmentation
8. Train / validate / test / predict for YOLOv26m-seg medium only
9. Build final validation/test tables
10. (optional) Compare with baseline original-dataset experiment
11. Trade-off plots and report-ready interpretation
12. Display qualitative prediction images
```

기본 설정은 medium 모델 + Roboflow augmentation만 사용한 학습입니다.
추가로 offline augmentation을 더 적용해보고 싶을 때만 `USE_OFFLINE_AUGMENTATION = True`로 바꾸세요.

---

## 9. Notes for ROS2 Integration

학습된 모델은 ROS2 perception node에서 다음 목적으로 사용할 수 있습니다.

- cup instance segmentation
- cup bounding box extraction
- cup mask center estimation
- fallen cup direction estimation
- cup pose estimation preprocessing

`predict_images/`의 정성적 결과와 `tables/*_compact.csv`의 정량 지표는 보고서 작성과 모델 채택 의사결정에 바로 사용할 수 있습니다.

---

## 10. 보고서 작성 시 해석 포인트

이 노트북의 결과는 기존 원본 데이터셋 실험과 비교하여 다음을 확인하기 위한 것입니다.

- Roboflow에서 augmentation까지 적용한 dataset version을 사용했을 때 medium 모델의 test Mask mAP50-95가 개선되는가?
- medium 모델 단독 학습에서 box mAP와 mask mAP, recall 사이의 균형이 cup-stacking task 요구사항을 만족하는가?
- validation/test set은 원본 그대로이므로, 성능 향상은 단순히 쉬운 augmented image를 맞힌 것이 아니라 원본 분포에 대한 일반화 성능 향상으로 해석할 수 있습니다.

로봇팔 컵쌓기 task에서는 cup 중심점과 외곽 mask를 사용하므로 Box mAP보다 **Mask mAP50-95**와 **Mask Recall**을 더 중요하게 해석하는 것이 적절합니다.
