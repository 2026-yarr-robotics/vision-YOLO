# data_generator — Isaac 자동 YOLO-seg 데이터 생성 + sim 파인튜닝 파이프라인

Isaac 디지털 트윈이 textured 컵 USD로 교체된 뒤 가용 가중치 전부가 시뮬
렌더의 전도(fallen)/뒤집힘(mouth-up) 컵을 분류하지 못하는 문제(temp_task
2026-06-13)를 해소하기 위한 파이프라인. **통제된 Isaac 씬에서 대규모
라벨링 데이터를 자동 생성**하고, 기존 hand/exo 학습 파이프라인을
**무수정으로 통과**시켜 새 가중치를 만든다.

## 구성

| 스크립트 | 실행 환경 | 역할 |
|---|---|---|
| `gen_yolo_dataset.py` | **Isaac python** (`~/dev_ws/isaac_sim/.../python.sh`) | 헤드리스 SimulationApp. 런타임 씬(scene_builder 보드/ArUco/조명 + M0609+RG2 암) 재구성, 프레임마다 kinematic frozen-author 로 컵 14개 랜덤 배치(피라미드 0~4단·부분/붕괴 상태, 산개 직립/전도/mouth-up, 마커 위 전도), hand(link_6)/exo 양 카메라 동시 렌더, Replicator instance segmentation + GT 기울기로 클래스 산출 → `dataset/sim/{hand,exo}` |
| `relabel_real.py` | 시스템 python3 | 단일 클래스 실데이터(hand v3 'cup' COCO, exo 'CUP' YOLO)를 배포 모델 예측 IoU 매칭으로 3클래스 재라벨 → `dataset/real/{hand3class, exo/YOLO_YARR-3class}` + 의심 표본 review 몽타주 + `overrides.json` 수동 교정 |
| `build_trainsets.py` | 시스템 python3 | real+sim 블렌드(train 에서 real ×k 오버샘플 ≈ 1:2) → `work/hand_mix`(roboflow COCO zip 레이아웃), `work/exo_mix/YOLO_YARR-2-class`(YOLO-seg) |
| `train_hand.py` | 시스템 python3 | **노트북 셀(9/11/13/15/16) 원본 코드를 exec** — COCO→YOLO 변환·검증·lightaug/redlite 오프라인 증강을 노트북 코드 그대로 수행 후, cell-20 의 model.train 인자 그대로 학습 |
| `train_exo.py` | 시스템 python3 | `exo-view/finetune-medium/train_segmentation_v4.py` 를 **무수정 import** 해 2-stage 파인튜닝 실행 (CWD=`work/exo_mix`, DEVICE 만 로컬 GPU 로 적응) |
| `eval_weights.py` | 시스템 python3 | 검증 기준 1·2 — sim 홀드아웃(fallen recall ≥0.9, upright→fallen ≈0 @conf0.5) + 실데이터 test 회귀(클래스 무관 검출률 old vs new) |

검증 기준 4(exo 클래스 플리커)는
`yarr-isaac-playground/tools/measure_label_flicker.py` (통합 세션에서
`/digital_twin/boxes` 샘플링), 기준 3(E2E)은
`yarr-isaac-playground/tools/verify_recovery.py --perception`.

## 클래스 체계 (3클래스, 이름 기반 소비 확인됨)

`fallen-cup` / `mouth-up-cup`(입구 위) / `upright-cup`(mouth-down 스태킹 자세).
GT 기울기 분류: 축 tilt <15° → upright(축 정방향)·mouth-up(축 역방향),
75~105° → fallen. 중간 기울기는 작성 단계에서 배제(전이 상태는 클래스 모호).

- hand COCO categories: roboflow 관행대로 id0 슈퍼카테고리
  `hand-eye-view-speed-stack-cup` + 1 fallen / 2 mouth-up / 3 upright
  (배포 speedstack3class 모델의 names 와 동일 매핑).
- exo data.yaml: `names: ['fallen-cup','mouth-up-cup','upright-cup']`
  (fallen=0 은 0609_exo 와 동일; 소비자는 전부 이름 문자열 기반 —
  `fallen_cup_pose_node.target_class_name`, `cup_fusion.cup_class_names`).

## 실행 순서

```bash
# 1) 데이터 생성 (Isaac python, 시스템 ROS 소스 금지; 3000씬 ≈ 30~50분)
~/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
    gen_yolo_dataset.py --frames 3000 --preview 24
# 산출: dataset/sim/{hand,exo} (씬 단위 8/1/1 분할), meta.jsonl(GT/가시성),
#       preview/*.png (마스크·클래스 오버레이 육안 검수용)

# 2) 실데이터 3클래스 재라벨 (review 몽타주 검수 → overrides.json → 재실행)
python3 relabel_real.py

# 3) 블렌드 학습셋 조립 (+--zip 으로 Colab 노트북용 zip)
python3 build_trainsets.py --zip

# 4) 학습
python3 train_hand.py --sizes s        # → work/hand_train/runs/segment/*/weights/best.pt
python3 train_exo.py                   # → work/exo_mix/runs/seg/two_stage_s2_v4*/weights/best.pt

# 5) 오프라인 검증 (기준 1·2)
python3 eval_weights.py --hand-new <hand_best.pt> --exo-new <exo_best.pt>
```

## 생성 데이터 1차 실측 (seed 20260613, 3000씬)

- 뷰당 3000장 (train 2400 / valid 300 / test 300, 씬 단위 분할 — 두 뷰가
  같은 씬에서 갈리지 않음), 총 461MB
- 인스턴스: fallen 5,100 / mouth-up 4,288 / upright 15,891 (mouth-up ≈
  fallen 동급 비중 — 기존 모델의 mouth-up↔fallen 혼동 차단 목적)
- 가시성 필터: 마스크 <300px 또는 visible/expected <0.10 제외 (roboflow
  실데이터 어노테이션 관행과 일치); 컵 0개(배경) 프레임 ~5% 포함
- 색×클래스 직교: 4색(red/blue/green/purple) 풀에 역할을 색과 무관하게
  배정 — 0609_exo 의 red↔fallen 상관 오검 차단

## 주의

- 조명은 런타임 sun(DistantLight 3000) 주변 랜덤(1700~4200, 각도/색온도) +
  돔 필 (40% 프레임은 돔 OFF — 런타임의 검은 보이드 배경 도메인 유지).
- 그리퍼가 hand 프레임 하단에 실기처럼 등장하도록 로봇 전체를 로드하고
  관절 뱅크(유효 FK 자세 700개)에서 프레임마다 선택 — 85% 는 컵 근처를
  보도록 편향, 나머지는 hard-negative 빈 보드 뷰.
- `gen_yolo_dataset.py` 는 yarr-isaac-playground 의 scene 모듈을 직접
  import 한다 (`--playground` 또는 자동 탐지). 완료 후 playground
  `tools/` 로 커밋해 관리한다 (temp_task 반영 항목).
