"""
Example to show how to draw an analog clock OpenCV
"""

# Import required packages:
import cv2
import numpy as np
import math
import random 
from utils import noise_image, draw_clock, resize_clock, translate_clock

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
hour, minute = input("hour and miniute: ").split()
hour = int(hour)
minute = int(minute)
print("hour:'{}' minute:'{}' ".format(hour, minute))

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

# Now, we draw the "dynamic" information:
while True:
    
    # Original Image Drawing:
    img = draw_clock(hour, minute, number_coords, colors, image_size, rectangle_size)

    noise_img, img_original = noise_image(number_coords, colors, hour, minute, img, image_size, 
                            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1,
                            minute_length=60, hour_length=40, center=(113, 113))
    
    # Resized Image Drawing:
    translated_img = resize_clock(img, rectangle_size)

    # Translated Image Drawing:
    translated_img = translate_clock(img, rectangle_size)

    cv2.imshow("Translated Resized Image",translated_img)

    # A wait of 500 milliseconds is performed (to see the displayed image)
    # Press q on keyboard to exit the program:
    key = cv2.waitKey(500) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

# Release everything:
cv2.destroyAllWindows()
