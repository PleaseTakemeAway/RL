"""
Example to show how to draw an analog clock OpenCV
"""

# Import required packages:
import cv2
import numpy as np
import math
import random 
from utils import noise_image, draw_clock, resize_clock, translate_clock
import os 
import matplotlib.pyplot as plt 

# Define some arguments 
image_size = (227, 227, 3)
center = (113, 113)  # 이미지 중앙
rectangle_size = ((5, 5), (222, 222))
radius = 80  # 숫자 위치 반지름
minute_length = 60
hour_length = 40
font = cv2.FONT_HERSHEY_SIMPLEX 
font_scale = 0.5 
font_thickness = 1


# Get current date:
print("Start Generating Images")

# hour, minute = input("hour and miniute: ").split()
# hour = int(hour)
# minute = int(minute)
# print("hour:'{}' minute:'{}' ".format(hour, minute))

# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

 
# Coordinates to define the origin for the hour markings:
number_coords = {
    1: (160, 46),
    2: (183, 73),
    3: (193, 113),
    4: (183, 153),
    5: (156, 180),
    6: (113, 193),
    7: (70, 180),
    8: (43, 153),
    9: (33, 113),
    10: (43, 73),
    11: (70, 46),
    12: (113, 33)
}

number = 0
# Now, we draw the "dynamic" information:
while True:
    
    # Random Time Generator
    hour = np.random.randint(0, 12)
    minute = np.random.randint(0, 60)
    
    # Original Image Drawing:
    img, color = draw_clock(hour, minute, number_coords, colors, image_size, rectangle_size)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # Noise Image Drawing:
    noise_img, img_original = noise_image(number_coords, colors, hour, minute, img, image_size, 
                            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1,
                            minute_length=60, hour_length=40, center=(113, 113))
    # plt.imshow(noise_img)
    # plt.axis('off')
    # plt.show()

    # Resized Image Drawing:
    resized_img = resize_clock(img, rectangle_size, color, colors)

    # plt.imshow(resized_img)
    # plt.axis('off')
    # plt.show()

    # Translated Image Drawing:
    translated_img = translate_clock(resized_img, color,rectangle_size, colors)
    # cv2.imshow("Translated Resized Image",translated_img)

    # plt.imshow(translated_img)
    # plt.axis('off')
    # plt.show()

    # 저장 경로와 파일 이름 설정
    # Collecting Imagesets
    if number < 100000:
        if number < 80000:
            save_dir = "data/images/train"
        else:
            save_dir = "data/images/val"
    else:
        break 
    file_name = f"train_image_{number}.png"
    save_path = os.path.join(save_dir, file_name)
    
    # 경로가 없으면 생성
    os.makedirs(save_dir, exist_ok= True)

    # 이미지 저장
    # cv2.imwrite(os.path.join(save_dir, f"img_{number:05d}_{hour}_{minute}_original.png"), img_original)
    # number += 1
    # cv2.imwrite(os.path.join(save_dir, f"img_{number:05d}_{hour}_{minute}_noise.png"), noise_img)
    # number += 1
    # cv2.imwrite(os.path.join(save_dir, f"img_{number:05d}_{hour}_{minute}_resized.png"), resized_img)
    # number += 1
    cv2.imwrite(os.path.join(save_dir, f"img_{number:05d}_{hour}_{minute}_translated.png"), translated_img)
    number += 1

    # A wait of 500 milliseconds is performed (to see the displayed image)
    # Press q on keyboard to exit the program:
    key = cv2.waitKey(500) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    
# Release everything:
cv2.destroyAllWindows()
