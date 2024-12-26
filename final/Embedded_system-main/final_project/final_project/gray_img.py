import numpy as np
import cv2

def bgr_to_gray(img):
    """
    將 BGR 彩色圖像轉換為灰階圖像
    :param img: numpy array, BGR 圖像
    :return: numpy array, 灰階圖像
    """
    # 確保圖像是三通道的彩色圖像
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("輸入圖像必須是 BGR 格式的彩色圖像")
    
    # 提取 B, G, R 通道
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    # 計算加權平均
    gray = 0.114 * B + 0.587 * G + 0.299 * R
    
    # 將結果轉換為整數類型
    return gray.astype(np.uint8)

# 測試
if __name__ == "__main__":
    import cv2
    # 讀取圖片
    img = cv2.imread('./photo/test.jpg')
    
    # 自製函數轉換
    #gray_img = bgr_to_gray(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 顯示結果
    cv2.imshow("Gray Image", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
