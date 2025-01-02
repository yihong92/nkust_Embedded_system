import numpy as np

def threshold(image, thresh, maxval):
    """
    模擬 cv2.threshold 的二值化功能
    :param image: numpy array, 灰階圖像
    :param thresh: int, 閾值
    :param maxval: int, 設定的最大值
    :return: numpy array, 二值化後的圖像
    """
    # 檢查輸入是否為灰階圖像
    if len(image.shape) != 2:
        raise ValueError("輸入圖像必須為灰階圖像")
    
    # 創建二值化結果陣列
    binary_image = np.zeros_like(image, dtype=np.uint8)
    
    # 運用條件設置像素值
    binary_image[image > thresh] = maxval
    
    return binary_image

# 測試
if __name__ == "__main__":
    import cv2
    # 讀取圖像並轉為灰階
    gray = cv2.imread('./photo/test_2.jpg', cv2.IMREAD_GRAYSCALE)
    
    # 自製二值化函數
    binary = threshold(gray, 127, 255)
     
    # 顯示結果
    cv2.imwrite("./photo/Binary.jpg", binary)
