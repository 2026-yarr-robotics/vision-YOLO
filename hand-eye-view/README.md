# Hand-eye View YOLO26-seg Fine-tuning Pipeline

이 디렉토리는 **hand-eye view 이미지에서 speed stack cup을 instance segmentation**하기 위해 YOLO26-seg 모델을 파인튜닝하는 Colab 노트북을 정리한 공간입니다.  
노트북은 COCO 형식의 segmentation annotation을 YOLO segmentation format으로 변환하고, YOLO26-seg 모델 학습부터 검증, 예측, 결과 백업, mask/center 정보 추출까지 한 번에 수행할 수 있도록 구성되어 있습니다.

---

## 1. Notebook Overview

사용 노트북:

```text
hand_eye_view_yolo26_seg_repo_style.ipynb
```

전체 흐름은 다음과 같습니다.

```text
Environment setup
→ Global configuration
→ Google Drive dataset copy
→ COCO dataset check
→ COCO segmentation to YOLO segmentation conversion
→ YOLO label verification
→ YOLO26-seg training
→ best.pt validation
→ sample prediction
→ model/result backup
→ mask, bbox, confidence, center extraction
```

---

## 2. Purpose

이 파이프라인의 목적은 다음과 같습니다.

- Hand-eye camera view에서 cup 영역을 segmentation하기 위한 YOLO26-seg 모델 학습
- Roboflow/COCO 스타일 annotation을 YOLO segmentation 학습 형식으로 변환
- 학습된 `best.pt` 모델을 Google Drive에 안전하게 저장
- 추론 결과에서 mask, bounding box, confidence, object center를 추출
- 이후 ROS2 perception node 또는 cup pose estimation node에서 사용할 수 있는 모델 생성

---

## 3. Expected Dataset Structure

Google Drive에는 COCO 형식 데이터셋이 아래와 같이 준비되어 있다고 가정합니다.

```text
/content/drive/MyDrive/hand_eye_view_train_1280/
├── train/
│   ├── _annotations.coco.json
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── valid/                 # 또는 val/
│   ├── _annotations.coco.json
│   └── ...
└── test/                  # optional
    ├── _annotations.coco.json
    └── ...
```

노트북의 기본 dataset path는 다음과 같습니다.

```python
DRIVE_DATASET = Path('/content/drive/MyDrive/hand_eye_view_train_1280')
```

데이터셋 위치가 다르면 `2. Global configuration` 셀에서 이 경로만 수정하면 됩니다.

---

## 4. Main Configuration

노트북의 핵심 설정값은 `2. Global configuration` 셀에 모아두었습니다.

```python
IMG_SIZE = 1280
MODEL_SIZE = 'm'
MODEL_WEIGHTS = f'yolo26{MODEL_SIZE}-seg.pt'
EXPERIMENT_NAME = f'hand_eye_view_yolo26{MODEL_SIZE}_seg_{IMG_SIZE}_a100'
```

`MODEL_SIZE`는 아래처럼 변경할 수 있습니다.

| 값 | 모델 |
|---|---|
| `n` | YOLO26 nano segmentation |
| `s` | YOLO26 small segmentation |
| `m` | YOLO26 medium segmentation |
| `l` | YOLO26 large segmentation |
| `x` | YOLO26 xlarge segmentation |

현재 기본값은 A100 환경에서 `yolo26m-seg.pt`를 사용하는 설정입니다.  
Large 모델로 학습하려면 다음처럼 바꾸면 됩니다.

```python
MODEL_SIZE = 'l'
```

---

## 5. Pipeline Steps

### Step 1. Environment Setup

Colab 환경에서 GPU 상태를 확인하고 필요한 패키지를 설치합니다.

주요 패키지:

- `ultralytics`
- `opencv-python-headless`
- `pycocotools`
- `pyyaml`
- `pandas`
- `matplotlib`

Colab 기본 환경과 충돌을 피하기 위해 `pandas==2.2.2`로 고정합니다.

```python
!pip install -q -U ultralytics opencv-python-headless pycocotools pyyaml matplotlib
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

Google Drive에 있는 원본 COCO dataset을 Colab local path로 복사합니다.

```text
/content/drive/MyDrive/hand_eye_view_train_1280
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

COCO annotation을 YOLO segmentation label format으로 변환합니다.

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

---

### Step 5. Label Verification

학습 전에 label 파일이 정상적으로 생성되었는지 확인합니다.

확인 항목:

- label `.txt` 파일 개수
- 비어 있지 않은 label 파일 개수
- 첫 번째 label line
- 한 줄에 포함된 값 개수

Segmentation label이라면 보통 값 개수가 5개보다 많아야 합니다.  
만약 값 개수가 5개라면 detection bbox format일 가능성이 높습니다.

---

### Step 6. YOLO26-seg Training

기본 학습 설정은 다음과 같습니다.

```python
epochs=150
imgsz=1280
batch=0.70
patience=60
optimizer='auto'
cos_lr=True
warmup_epochs=5.0
overlap_mask=True
mask_ratio=2
amp=True
cache='ram'
```

A100 환경에서는 `batch=0.70`을 사용하여 자동 batch 설정을 사용합니다.  
OOM이 발생하면 다음처럼 정수 batch로 낮추면 됩니다.

```python
batch=8
# or
batch=6
# or
batch=4
```

학습 결과는 다음 경로에 저장됩니다.

```text
/content/runs/segment/hand_eye_view_yolo26m_seg_1280_a100/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
└── ...
```

---

### Step 7. Validation

학습이 끝난 뒤 `best.pt`를 기준으로 validation을 수행합니다.

출력되는 주요 metric은 다음과 같습니다.

| Metric | Meaning |
|---|---|
| `metrics.box.map` | Box mAP50-95 |
| `metrics.box.map50` | Box mAP50 |
| `metrics.seg.map` | Mask mAP50-95 |
| `metrics.seg.map50` | Mask mAP50 |

실험 결과는 실행 후 아래 표에 기록하면 됩니다.

| Model | Image Size | Validation Box mAP50 | Validation Mask mAP50 | Note |
|---|---:|---:|---:|---|
| YOLO26m-seg | 1280 | - | - | Default setting |
| YOLO26l-seg | 1280 | - | - | Optional large model |

---

### Step 8. Sample Prediction

학습된 `best.pt`를 사용하여 test, valid, train 이미지 중 하나를 자동으로 선택하고 segmentation 예측을 수행합니다.

예측 결과는 다음 경로에 저장됩니다.

```text
/content/preds/hand_eye_view_yolo26m_seg_1280_a100_single/
```

저장되는 결과에는 예측 mask와 bounding box가 overlay된 이미지가 포함됩니다.

---

### Step 9. Model and Run Backup

Colab runtime이 종료되어도 모델이 사라지지 않도록 Google Drive에 결과를 복사합니다.

저장 위치:

```text
/content/drive/MyDrive/hand_eye_view_yolo_seg_result/
```

저장 파일 예시:

```text
hand_eye_view_yolo26m_seg_1280_a100_best.pt
hand_eye_view_yolo26m_seg_1280_a100_last.pt
hand_eye_view_yolo26m_seg_1280_a100.zip
```

주의할 점: `.pt` 파일은 내부적으로 zip archive처럼 보일 수 있습니다.  
Google Drive에서 압축 파일처럼 보여도 압축을 풀지 말고 `.pt` 파일 자체를 모델 가중치로 사용해야 합니다.

---

### Step 10. Mask, BBox, Confidence, Center Extraction

마지막 셀에서는 단일 이미지 추론 결과에서 객체별 정보를 추출합니다.

추출 정보:

- class id
- class name
- confidence
- bounding box 좌표
- mask 중심점 `(center_x, center_y)`
- mask area pixel 수
- object mask image path

결과는 CSV로 저장됩니다.

```text
/content/hand_eye_view_yolo26m_seg_1280_a100_masks/segmentation_objects.csv
```

CSV 예시 column:

```text
id, class_id, class_name, confidence,
bbox_x1, bbox_y1, bbox_x2, bbox_y2,
center_x, center_y, mask_area_px, mask_path
```

이 출력은 이후 fallen cup direction estimation, cup pose estimation, ROS2 perception node와 연결할 때 사용할 수 있습니다.

---

## 6. Output Summary

| Output | Path | Description |
|---|---|---|
| YOLO dataset | `/content/hand_eye_view_yolo26_seg_1280` | COCO에서 변환된 YOLO-seg dataset |
| data yaml | `/content/hand_eye_view_yolo26_seg_1280/data.yaml` | YOLO 학습용 dataset yaml |
| best model | `/content/runs/segment/.../weights/best.pt` | validation 기준 최적 모델 |
| last model | `/content/runs/segment/.../weights/last.pt` | 마지막 epoch 모델 |
| prediction image | `/content/preds/...` | segmentation 예측 시각화 결과 |
| backup models | `/content/drive/MyDrive/hand_eye_view_yolo_seg_result` | Drive에 저장된 모델 및 zip |
| object csv | `/content/..._masks/segmentation_objects.csv` | mask, bbox, confidence, center 정보 |

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
batch=8
```

그래도 부족하면 다음 순서로 낮춥니다.

```python
batch=6
batch=4
batch=2
```

---

### 4. `.pt` file looks like a zip file

PyTorch `.pt` checkpoint는 내부적으로 zip 기반 구조를 사용할 수 있습니다.  
따라서 Google Drive나 파일 탐색기에서 압축 파일처럼 보여도 정상입니다.

해결:

- 압축을 풀지 않습니다.
- 아래처럼 `.pt` 파일 경로를 그대로 사용합니다.

```python
model = YOLO('/content/drive/MyDrive/hand_eye_view_yolo_seg_result/hand_eye_view_yolo26m_seg_1280_a100_best.pt')
```

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
7. Train YOLO26 segmentation model
8. Validate best.pt
9. Predict sample images
10. Save trained model and run folder to Google Drive
11. Single-image mask, bbox, confidence, center extraction
```

처음 실행할 때는 기본 설정인 `MODEL_SIZE='m'`으로 학습한 뒤, 성능이 부족하면 `MODEL_SIZE='l'`로 변경하여 large 모델을 추가 실험하는 것을 권장합니다.

---

## 9. Notes for ROS2 Integration

학습된 모델은 ROS2 perception node에서 다음 목적으로 사용할 수 있습니다.

- cup instance segmentation
- cup bounding box extraction
- cup mask center estimation
- fallen cup direction estimation
- cup pose estimation preprocessing

특히 마지막 셀에서 추출하는 `center_x`, `center_y`, `mask_area_px`, `bbox` 정보는 이후 depth image 또는 camera intrinsic과 결합하여 3D position estimation에 사용할 수 있습니다.
