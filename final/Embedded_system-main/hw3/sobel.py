
import cv2 
import numpy as np
from matplotlib import pyplot as plt

def sobel_and_morphology(img):
    """
    Sobel operator with morphological processing
    
    Parameters:
        img: Input image (grayscale)
    Returns:
        sobel_combined: Combined Sobel edges
    """
    # 1. 先進行高斯模糊以減少噪聲
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 2. 計算 Sobel 梯度
    # X 方向邊緣檢測
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)  # 轉換為 uint8
    
    # Y 方向邊緣檢測
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = cv2.convertScaleAbs(sobely)
    
    # 3. 合併兩個方向
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    
    return sobel_combined

