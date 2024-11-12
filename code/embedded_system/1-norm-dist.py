import numpy as np
import cv2

def calculate_histogram(image_path, num_bins=256):
    """
    計算灰度直方圖
    :param image_path: 圖像路徑
    :param num_bins: 直方圖分箱數
    :return: 直方圖 (numpy array)
    """
    # 讀取灰度圖像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid format.")
    
    # 計算直方圖
    hist = cv2.calcHist([img], [0], None, [num_bins], [0, 256])
    return hist.ravel()  # 展平為一維向量

def calculate_1_norm_distance_histogram(hist1, hist2):
    """
    計算兩個直方圖之間的 1-Norm Distance
    :param hist1: 第一個直方圖
    :param hist2: 第二個直方圖
    :return: 1-Norm Distance (float)
    """
    # 確保兩個直方圖長度一致
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same number of bins.")
    
    # 計算 1-Norm Distance
    distance = np.sum(np.abs(hist1 - hist2))
    return distance

# 範例使用
# 假設你有兩張 LBP 結果圖片
hist1 = calculate_histogram('lbp2_result.jpg')  # 第一張圖片的直方圖
hist2 = calculate_histogram('sobel_result.jpg')  # 第二張圖片的直方圖

distance = calculate_1_norm_distance_histogram(hist1, hist2)
print(f"1-Norm Distance between the two histograms: {distance}")