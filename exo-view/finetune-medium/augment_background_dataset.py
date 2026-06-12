import os
import glob
import cv2
import numpy as np

# ────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────
TRAIN_IMG_DIR = '/home/rim0504/ss_project/YOLO_YARR-2-class/train/images/'
TRAIN_LBL_DIR = '/home/rim0504/ss_project/YOLO_YARR-2-class/train/labels/'

def augment_backgrounds():
    # 1. 훈련 데이터셋 내 모든 이미지 탐색
    img_paths = glob.glob(os.path.join(TRAIN_IMG_DIR, '*'))
    bg_images = []

    for img_path in img_paths:
        base_name = os.path.basename(img_path)
        name_wo_ext, ext = os.path.splitext(base_name)
        if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
            
        txt_path = os.path.join(TRAIN_LBL_DIR, name_wo_ext + '.txt')
        
        # 라벨 파일이 없거나 파일 크기가 0 bytes 이면 배경 이미지로 판단
        if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
            bg_images.append(img_path)

    print(f"▶ 발견된 학습용 배경 이미지: {len(bg_images)}장")
    if len(bg_images) == 0:
        print("⚠️ 배경 이미지가 존재하지 않아 증강을 건너뜁니다.")
        return

    # 2. 증강 기법 정의 및 복제 수행
    count = 0
    for img_path in bg_images:
        base_name = os.path.basename(img_path)
        name_wo_ext, ext = os.path.splitext(base_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 변환 1: Horizontal Flip (가로 뒤집기)
        img_flip = cv2.flip(img, 1)
        
        # 변환 2: 밝게 조절 (+35)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 35)
        img_bright = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        
        # 변환 3: 어둡게 조절 (-35)
        hsv_d = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hd, sd, vd = cv2.split(hsv_d)
        vd = cv2.subtract(vd, 35)
        img_dark = cv2.cvtColor(cv2.merge((hd, sd, vd)), cv2.COLOR_HSV2BGR)
        
        # 변환 4: 가우시안 노이즈 주입
        noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
        img_noise = cv2.add(img, noise)
        
        # 변환 5: 가벼운 가우시안 블러
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        # 각 변환 이미지 저장 및 빈 라벨 파일 생성
        transforms = {
            'flip': img_flip,
            'bright': img_bright,
            'dark': img_dark,
            'noise': img_noise,
            'blur': img_blur
        }
        
        for suffix, trans_img in transforms.items():
            new_img_name = f"{name_wo_ext}_bgaug_{suffix}{ext}"
            new_img_path = os.path.join(TRAIN_IMG_DIR, new_img_name)
            new_txt_path = os.path.join(TRAIN_LBL_DIR, f"{name_wo_ext}_bgaug_{suffix}.txt")
            
            # 이미지 저장
            cv2.imwrite(new_img_path, trans_img)
            # 빈 라벨 파일 생성
            with open(new_txt_path, 'w') as f:
                pass
            count += 1

    print(f"▶ 성공적으로 {count}장의 배경 증강 이미지 및 빈 라벨 파일을 주입 완료했습니다.")

if __name__ == '__main__':
    augment_backgrounds()
