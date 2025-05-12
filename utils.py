import cv2 
import numpy as np
import math 
import random 
import torch 
import torch.nn as nn
from torchvision import models 
from PIL import Image 
from torch.utils.data import Dataset
import os 

def noise_image(number_coords, colors, hour, minute, image, image_size, 
                font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1,
                minute_length=60, hour_length=40, center=(113, 113)):

    # 중심 좌표 계산 (숫자 좌표 평균)
    n = len(number_coords)
    center_dot = (
        sum(x for x, _ in number_coords.values()) // n,
        sum(y for _, y in number_coords.values()) // n
    )

    # 이미지 복사 (static 버전)
    image_original = image.copy()

    # 노이즈 생성
    noise_level = np.random.choice([1, 64, 128, 192, 256], p=[0.1, 0.2, 0.4, 0.2, 0.1])
    noise = np.random.randint(0, noise_level, image.shape, dtype=np.uint8)
    print('noise level:', noise_level)

    # 노이즈 추가
    noise_img = cv2.add(image.copy(), noise)

    # 바늘 각도 계산
    minute_angle = (minute * 6 + 270) % 360
    hour_angle = (hour * 30 + minute * 0.5 + 270) % 360
    print(f"hour_angle: {hour_angle}°, minute_angle: {minute_angle}°")

    # 라디안 변환
    minute_rad = math.radians(minute_angle)
    hour_rad = math.radians(hour_angle)

    # 시계 바늘 위치 계산
    minute_end = (
        round(center[0] + minute_length * math.cos(minute_rad)),
        round(center[1] + minute_length * math.sin(minute_rad))
    )
    hour_end = (
        round(center[0] + hour_length * math.cos(hour_rad)),
        round(center[1] + hour_length * math.sin(hour_rad))
    )

    # 바늘 그리기 (노이즈 이미지 위에만)
    cv2.line(noise_img, center, minute_end, colors['black'], 5)
    cv2.line(noise_img, center, hour_end, colors['black'], 10)

    # 중심 원
    cv2.circle(noise_img, center_dot, 10, colors['dark_gray'], -1)

    # 시각화 (선택)
    cv2.imshow("clock", noise_img)

    return noise_img, image_original

        

def resize_clock(image, clock_box, bg_color, colors, scale_range=(0.3, 0.9)):
    """
    Resize the clock region and paste it back centered at original clock center.
    image: numpy.ndarray (OpenCV BGR)
    clock_box: ((x_min, y_min), (x_max, y_max))
    """
    # Image Size 설정
    img_h, img_w = image.shape[:2]
    (x_min, y_min), (x_max, y_max) = clock_box
    clock_crop = image[y_min:y_max, x_min:x_max].copy()

    clock_w, clock_h = x_max - x_min, y_max - y_min
    center_x, center_y = x_min + clock_w // 2, y_min + clock_h // 2

    # 리사이즈 크기 (정사각형)
    target_size = int(random.uniform(*scale_range) * min(img_w, img_h))
    resized_clock = cv2.resize(clock_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 새로운 좌표 계산
    new_x_min = np.clip(center_x - target_size // 2, 0, img_w - target_size)
    new_y_min = np.clip(center_y - target_size // 2, 0, img_h - target_size)
    new_x_max = new_x_min + target_size
    new_y_max = new_y_min + target_size

    # 원래 위치 제거 (선택적)
    image[y_min:y_max, x_min:x_max] = colors[bg_color]

    # 새로운 위치에 삽입
    image[new_y_min:new_y_max, new_x_min:new_x_max] = resized_clock

    return image

# Now, we draw the "dynamic" information:

def draw_clock(hour, minute, number_coords, colors, image_size, rectangle_size, center=None):
    # 
    hour_length = (193-33) * 0.25
    minute_length = (193-33) * 0.425

    # 1. 캔버스 생성
    image = np.zeros(image_size, dtype=np.uint8)
    color = random.choice([k for k, v in colors.items() if k != 'black'])
    image[:] = colors[color]

    # 2. 시계 배경
    bg_color = random.choice([v for k, v in colors.items() if k not in ['black',color]])
    cv2.rectangle(image, rectangle_size[0], rectangle_size[1], bg_color, -1)

    # 3. 중심 계산
    if center is None:
        n = len(number_coords)
        cx = sum(x for x, y in number_coords.values()) // n
        cy = sum(y for x, y in number_coords.values()) // n
        center = (cx, cy)

    # 4. 숫자 표시
    for i in range(1, 13):
        cv2.putText(image, f"{i}", number_coords[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['black'], 1, cv2.LINE_AA)

    # 5. 각도 및 바늘
    minute_angle = math.radians((minute * 6 + 270) % 360)
    hour_angle = math.radians((hour * 30 + minute * 0.5 + 270) % 360)

    cv2.line(image, center,
             (round(center[0] + minute_length * math.cos(minute_angle)),
              round(center[1] + minute_length * math.sin(minute_angle))),
             colors['black'], 3)
    
    cv2.line(image, center,
             (round(center[0] + hour_length * math.cos(hour_angle)),
              round(center[1] + hour_length * math.sin(hour_angle))),
             colors['black'], 5)

    # 중심 원
    cv2.circle(image, center, 10, colors['dark_gray'], -1)
    cv2.imshow("Original Clock", image)

    return image, color

def translate_clock(image, bg_color, clock_box, colors,shift_range=(0.25, 0.5)):
    """
    image: numpy.ndarray (OpenCV 이미지, BGR)
    clock_box: ((x_min, y_min), (x_max, y_max)) 형태의 튜플
    shift_range: 시계 크기에 대한 이동 비율 범위 (최소, 최대)
    """
    img_h, img_w = image.shape[:2]
    x_min, y_min = clock_box[0]
    x_max, y_max = clock_box[1]
    clock_w = x_max - x_min
    clock_h = y_max - y_min

    # 이동 방향 및 거리
    direction = random.choice(['up', 'down', 'left', 'right'])
    shift_frac = random.uniform(*shift_range)
    shift_x, shift_y = 0, 0

    if direction == 'up':
        shift_y = -int(clock_h * shift_frac)
    elif direction == 'down':
        shift_y = int(clock_h * shift_frac)
    elif direction == 'left':
        shift_x = -int(clock_w * shift_frac)
    elif direction == 'right':
        shift_x = int(clock_w * shift_frac)

    # 새 좌표 계산
    new_x_min = np.clip(x_min + shift_x, 0, img_w - clock_w)
    new_y_min = np.clip(y_min + shift_y, 0, img_h - clock_h)
    new_x_max = new_x_min + clock_w
    new_y_max = new_y_min + clock_h

    # 시계 영역 잘라내기
    clock_crop = image[y_min:y_max, x_min:x_max].copy()

    # 시계 붙여넣기 (기존 위치는 0으로 지우고 싶다면 아래 줄 추가 가능)
    image[y_min:y_max, x_min:x_max] = colors[bg_color]
    image[new_y_min:new_y_max, new_x_min:new_x_max] = clock_crop

    return image

def parse_time(filepath):
    basename = os.path.basename(filepath)
    name, _ = os.path.splitext(basename)
    _, _, hour_str, minute_str, _ = name.split('_')

    return int(hour_str), int(minute_str)

class ClockDataset(Dataset):
    def __init__(self, image_paths, transform = None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        hour, minute = parse_time(img_path)
        return image, torch.tensor([hour, minute])
    
class TwoHeadResNet(nn.Module):
    def __init__(self, num_hour_class=12 , num_minute_class=60, pretrained=True):
        super().__init__()
        # ResNet18의 BackBone
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 분기된 FCN
        self.fc_hour = nn.Linear(num_ftrs, num_hour_class)
        self.fc_minute = nn.Linear(num_ftrs, num_minute_class)

    def forward(self, x):
        features = self.backbone(x)
        hour_logits = self.fc_hour(features)
        minute_logits = self.fc_minute(features)
        return hour_logits, minute_logits
