import cv2
import numpy as np

def label_similar_areas(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    根據遮罩上色影像中的相似區域。

    參數：
        image (numpy.ndarray): 原始影像，使用 BGR 格式。
        mask (numpy.ndarray): 二值遮罩，值為 1 的地方表示需要上色的區域。
        color (tuple): 上色使用的顏色，預設為綠色 (0, 255, 0)。
        alpha (float): 上色透明度，0 到 1 之間，預設為 0.5。
    
    回傳：
        numpy.ndarray: 上色後的影像。
    """
    # 創建一個彩色遮罩
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color

    # 將彩色遮罩疊加到原始影像上
    colored_image = cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)
    
    return colored_image