
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

def sobel(img):
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
      
    #return sobel_combined
    return  abs_sobelx ,abs_sobely


if __name__ == "__main__":
 
    # 讀取輸入圖像
    img = cv2.imread('./cropped/cropped_image_23.jpg')  # 讀取名稱為'test.jpg'的圖像文件

    # 將圖像轉換為灰度圖像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 將彩色圖像轉換為灰度圖像
    kernel = np.ones((3, 3), np.uint8)
    # 取得 Sobel 邊緣圖像
    sobelx, sobely = sobel(img)

    # 保存輸出圖像
    cv2.imwrite('./photo/sobelx.jpg', sobelx)  
    cv2.imwrite('./photo/sobely.jpg', sobely)  

