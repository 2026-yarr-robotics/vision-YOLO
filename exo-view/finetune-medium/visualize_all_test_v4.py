import os
import glob
import cv2
import subprocess
from ultralytics import YOLO

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
TEST_IMAGES_DIR = '/home/rim0504/ss_project/YOLO_YARR-2-class/test/images/'
VIDEO_PATH = '/home/rim0504/ss_project/KakaoTalk_Video_2026-06-09-18-50-06.mp4'

# v4 학습 결과 폴더 탐색
MODEL_V4_DIRS = sorted(
    glob.glob('/home/rim0504/ss_project/runs/segment/runs/seg/two_stage_s2_v4*/') +
    glob.glob('/home/rim0504/ss_project/runs/seg/two_stage_s2_v4*/'),
    key=os.path.getmtime
)

OUTPUT_IMG_DIR = '/home/rim0504/ss_project/runs/segment/predict_all_test_v4/'

def visualize_all():
    # 학습 후 최신 경로 동적 갱신
    dirs = sorted(
        glob.glob('/home/rim0504/ss_project/runs/segment/runs/seg/two_stage_s2_v4*/') +
        glob.glob('/home/rim0504/ss_project/runs/seg/two_stage_s2_v4*/'),
        key=os.path.getmtime
    )
    if not dirs:
        print("⚠️ 에러: v4 학습 완료 폴더를 찾을 수 없습니다.")
        return
        
    best_weights = os.path.join(dirs[-1], 'weights', 'best.pt')
    print(f"▶ 로드할 v4 모델 가중치: {best_weights}")
    
    if not os.path.exists(best_weights):
        print(f"⚠️ 에러: {best_weights} 파일이 없습니다.")
        return

    # 1. 테스트 이미지 전체 시각화
    model = YOLO(best_weights)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    img_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, '*')))
    print(f"▶ 발견된 테스트 이미지 개수: {len(img_paths)}장")
    
    for idx, img_path in enumerate(img_paths):
        base_name = os.path.basename(img_path)
        if not base_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
            
        print(f"[{idx+1}/{len(img_paths)}] 이미지 추론 중: {base_name}")
        res = model(img_path, imgsz=1280, device=0)[0]
        
        plotted_img = res.plot()
        save_path = os.path.join(OUTPUT_IMG_DIR, base_name)
        cv2.imwrite(save_path, plotted_img)
        print(f"  -> 저장 완료: {save_path}")

    # 2. 테스트 동영상 시각화
    if os.path.exists(VIDEO_PATH):
        print(f"▶ 비디오 추론 시작: {VIDEO_PATH}")
        # CLI 호출 형태로 yolo predict 명령 수행 (yolo_train 가상환경 python 진입 유도)
        video_cmd = [
            '/home/rim0504/miniconda3/envs/yolo_train/bin/yolo',
            'task=segment', 'mode=predict',
            f'model={best_weights}',
            f'source={VIDEO_PATH}',
            'imgsz=1280', 'device=1', 'save=True',
            'project=runs/segment/predict_video_v4',
            'name=v4_video_result'
        ]
        try:
            # yolo CLI 실행
            subprocess.run(video_cmd, check=True)
            print("▶ 비디오 추론 시각화 완료!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 비디오 추론 실패: {e}")
    else:
        print(f"⚠️ 경고: 비디오 파일이 존재하지 않아 비디오 시각화는 생략합니다: {VIDEO_PATH}")

if __name__ == '__main__':
    visualize_all()
