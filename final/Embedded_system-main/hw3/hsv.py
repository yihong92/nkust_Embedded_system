import cv2
import numpy as np


def hsv(image, lower, upper):

    # 讀取影像
    if image is None:
        raise ValueError(f"Error: unable to load image '{image}'")

    # 將影像轉換為 HSV 色彩空間
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 建立遮罩，將範圍內的深色設為白色(255)，其他顏色為黑色(0)
    mask = cv2.inRange(hsv_image, lower, upper)
    
    # 反轉遮罩，讓非深色部分為白色(255)，深色部分為黑色(0)
    non_mask = cv2.bitwise_not(mask)

    # 將遮罩應用到原始影像，僅保留非天空的部分
    filtered_image = cv2.bitwise_and(image, image, mask=non_mask)

    return filtered_image
