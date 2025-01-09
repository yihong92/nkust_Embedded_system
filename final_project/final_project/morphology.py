import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def morphological_opening(image, kernel):
    """
    模擬 cv2.morphologyEx 的開運算功能
    :param image: numpy array, 二值化圖像
    :param kernel: numpy array, 結構元素（核）
    :return: numpy array, 開運算後的圖像
    """
    # 確保輸入是二值圖像
    if len(image.shape) != 2:
        raise ValueError("輸入圖像必須是二維的二值化圖像")
    

    # 腐蝕操作
    eroded = binary_erosion(image, structure=kernel).astype(np.uint8)
    
    # 膨脹操作
    opened = binary_dilation(eroded, structure=kernel).astype(np.uint8)

    return (opened * 255).astype(np.uint8)

# 測試
if __name__ == "__main__":
    import cv2
    
    # 讀取灰階圖像並進行二值化
    gray = cv2.imread('./photo/Binary.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 創建一個 5x5 的結構元素
    kernel = np.ones((41, 41), np.uint8)
    
    # 自製開運算
    opened_custom = morphological_opening(thresh // 255, kernel)
    
    # 使用 OpenCV 開運算
    opened_cv = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 顯示結果
    cv2.imwrite('morphology.jpg',opened_custom)

